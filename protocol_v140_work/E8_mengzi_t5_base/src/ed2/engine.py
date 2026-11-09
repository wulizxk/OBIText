import math
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from benchmark_protocol_v140 import FORMAL_TEST_SUITES, compatible_source, evaluation_config

import numpy as np
import torch

from .data import Codec, SpanSampler, benchmark_public, benchmark_gold
from .decoding import predict_encoded_batch
from .io import digest, read, sha, source_fingerprint, verify_data, write, write_rows
from .metrics import aggregate, better, score_prediction
from .model import autocast, load_pretrained, optimizer_for, per_sample_losses, verify_weights, scaler_for
from .protocol import FORMAL_CONFIG, validate_formal_config
from .runtime import configure_execution, runtime_versions, require_runtime, evaluation_environment, require_cuda_precision
from .run_state import SCHEMA_VERSION, stage_best, reconcile_records, complete_run, require_completed_run, validate_history
from .run_state import completed_training_batch
from .inference_graph import EvaluationGraphSession


def seed_all(seed):
    configure_execution(seed)


def rng_state(sampler):
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'sampler': sampler.rng.getstate()}


def restore_rng(state, sampler):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume with the same CUDA_VISIBLE_DEVICES count')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    sampler.rng.setstate(state['sampler'])


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def scheduler_for(optimizer, cfg):
    per_epoch = math.ceil(cfg['samples_per_epoch'] / cfg['effective_batch_size'])
    total = cfg['max_epochs'] * per_epoch
    warmup = round(cfg['warmup_epochs'] * per_epoch)
    if not 0 < warmup < total:
        raise ValueError('Warmup must be positive and shorter than training')

    def scale(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (total - step) / (total - warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def update(model, codec, questions, cfg, device, optimizer, scaler, targets=None):
    """Accumulate sample sums / actual effective batch size, including the tail."""
    model.train()
    if targets is None:
        targets = torch.tensor(codec.candidate_ids, dtype=torch.long, device=device)
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    attempt = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        for start in range(0, len(questions), cfg['micro_batch_size']):
            batch = codec.collate(questions[start:start + cfg['micro_batch_size']], device)
            with autocast(device, cfg):
                losses = per_sample_losses(model, batch, targets)
                loss = losses.sum() / len(questions)
            if not bool(torch.isfinite(losses).all() & torch.isfinite(loss)):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError('Non-finite forward loss before backward; no optimizer step committed')
            scaler.scale(loss).backward()
            loss_sum += losses.detach().float().sum()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=False)
        finite_loss, finite_norm = torch.isfinite(loss_sum), torch.isfinite(norm)
        if bool(finite_loss & finite_norm):
            scaler.step(optimizer)
            scaler.update()
            return float(loss_sum), float(norm)
        if not bool(finite_loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Non-finite training loss; no checkpoint committed')
        if not scaler.is_enabled() or scaler.get_scale() <= 1:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Non-finite unscaled gradients; no optimizer step committed')
        # Retry the SAME examples/dropout masks at a lower scale. No optimizer or
        # scheduler step is consumed. update(new_scale=...) also clears unscale state.
        lower_scale = max(1.0, scaler.get_scale() * .5)
        scaler.update(new_scale=lower_scale)
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, device)
        attempt += 1
        print(f'AMP_RETRY attempt={attempt} scale={lower_scale}', flush=True)


def evaluate(model, codec, public, answers, cfg, device, encoded=None, execution_report=None):
    model.eval()
    if encoded is None:
        encoded = [codec.encode(q) for q in public]
    if len(encoded) != len(public):
        raise ValueError('Encoded evaluation questions do not match the public suite')
    # Group equal gap lengths and similar input lengths to avoid padding and
    # inactive beam slots. Restore canonical public-file order before scoring.
    batches = []
    for gap_length in sorted({q['gap_length'] for q in public}):
        group = sorted((i for i, q in enumerate(public) if q['gap_length'] == gap_length),
                       key=lambda i: (len(encoded[i]['input_ids']), i))
        batches.extend(group[start:start + cfg['eval_question_batch_size']]
                       for start in range(0, len(group), cfg['eval_question_batch_size']))
    raw_predictions = [None] * len(public)
    targets = torch.tensor(codec.target_ids, dtype=torch.long, device=device)
    completed = 0
    with EvaluationGraphSession(model, cfg, device) as execution:
        for batch_index, indices in enumerate(batches):
            encoded_batch = [encoded[i] for i in indices]
            # Inference cannot see the gold dictionary. Gold is joined only after prediction.
            options = {'executor': execution} if execution.enabled else {}
            predictions = predict_encoded_batch(model, codec, encoded_batch, cfg, device, targets, **options)
            for i, prediction in zip(indices, predictions):
                raw_predictions[i] = prediction
            completed += len(indices)
            if (batch_index + 1) % 100 == 0 or completed == len(public):
                print(f'EVAL {completed}/{len(public)}', flush=True)
    if execution_report is not None:
        execution_report.update(execution.stats)
    if execution.enabled:
        print('EVAL_EXECUTION ' + json.dumps(execution.stats), flush=True)
    # Formal callers pass a loader, so gold is opened only after every public
    # question has a committed model prediction in canonical order.
    if callable(answers):
        answers = answers()
    values = [score_prediction(q, prediction, answers[q['example_id']], codec.units)
              for q, prediction in zip(public, raw_predictions)]
    return aggregate(values), values


def validate_config(cfg):
    return validate_formal_config(cfg)


def require_formal_cuda(device, cfg, stage):
    """Formal training/Test must use the frozen native CUDA BF16 protocol."""
    device = torch.device(device)
    if cfg.get('precision') != FORMAL_CONFIG['precision'] or device.type != 'cuda':
        raise RuntimeError(f'Formal ED2-A {stage} requires a CUDA device with BF16; CPU is diagnostic only')
    require_cuda_precision(device, cfg)
    return device


def identity(project, data, weights, cfg, seed):
    policy = configure_execution()
    return {'cfg': cfg, 'seed': seed, 'data_sha256': verify_data(data),
            'code_sha256': source_fingerprint(project), 'weights_sha256': verify_weights(weights, cfg),
            'runtime': runtime_versions(), 'execution_policy': policy}


def model_from_checkpoint(ckpt, device):
    from transformers import T5Config, T5ForConditionalGeneration
    if ckpt['model_config'].get('model_type') != 't5':
        raise ValueError('ED2 requires a T5 checkpoint; mT5/E7 checkpoints are not compatible')
    if ckpt['model_config'].get('tie_word_embeddings') is not False:
        raise ValueError('ED2 output head must remain untied')
    config = T5Config.from_dict(ckpt['model_config'])
    config._attn_implementation = 'eager'
    config.use_cache = False
    model = T5ForConditionalGeneration(config)
    model.load_state_dict(ckpt['model'], strict=True)
    model.tie_weights()
    return model.to(device)


def load_checkpoint(path):
    # Only tensors and primitive containers are serialized: never arbitrary pickle objects.
    return torch.load(path, map_location='cpu', weights_only=True)


def train(project, data, weights, output, cfg, seed, device, resume=False):
    validate_config(cfg)
    device = require_formal_cuda(device, cfg, 'training')
    output = Path(output)
    ident = identity(project, data, weights, cfg, seed)
    if output.exists() and not resume:
        raise FileExistsError('Output exists; choose a new run or explicitly --resume')
    if resume and not (output / 'last.pt').is_file():
        raise FileNotFoundError('No epoch-boundary last.pt to resume')
    output.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent jobs from writing the same run directory. Kernel lock is released on crash.
    import fcntl
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train_locked(project, data, weights, output, cfg, seed, device, resume, ident)


def _train_locked(project, data, weights, output, cfg, seed, device, resume, ident):
    initialization_started = time.perf_counter()
    print(f'TRAIN_INIT seed={seed} stage=codec_and_train_data resume={resume}', flush=True)
    seed_all(seed)
    codec = Codec(data, cfg)
    sampler = SpanSampler(data, seed)
    public = benchmark_public(data, 'dev_full')
    answers = lambda: benchmark_gold(data, 'dev_full', public)
    dev_encoded = [codec.encode(q) for q in public]
    counts = Counter(str(q['gap_length']) for q in public)
    if dict(counts) != read(Path(data) / 'benchmark/protocol.json')['validation_length_counts']:
        raise ValueError('Full-validation length counts changed')
    ckpt = load_checkpoint(output / 'last.pt') if resume else None
    if ckpt and ckpt['identity'] != ident:
        raise ValueError('Resume identity mismatch: config/data/code/weights/runtime/seed')
    if ckpt:
        if ckpt.get('schema_version') != SCHEMA_VERSION or ckpt.get('kind') != 'training':
            raise ValueError('Unsupported resume schema; legacy smoke/checkpoints are not converted automatically')
        reconcile_records(output, ident, ckpt['epoch'], ckpt['step'], ckpt['history'], ckpt['best'])
        model = model_from_checkpoint(ckpt, device)
        epoch, step, history, best = ckpt['epoch'], ckpt['step'], ckpt['history'], ckpt['best']
    else:
        print(f'TRAIN_INIT seed={seed} stage=fixed_pretrained_weights', flush=True)
        model, info = load_pretrained(weights, codec, cfg)
        model.to(device)
        write(output / 'pretrained_loading.json', info)
        epoch, step, history, best = 0, 0, [], None
    optimizer = optimizer_for(model, cfg)
    scheduler = scheduler_for(optimizer, cfg)
    scaler = scaler_for(device, cfg)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        restore_rng(ckpt['rng'], sampler)
        del ckpt
    write(output / 'run.json', {'kind': 'formal', 'identity': ident, 'parameter_count': sum(p.numel() for p in model.parameters()),
                                'resume_granularity': 'completed_epoch', 'test_used_for_selection': False,
                                'training_execution': evaluation_environment(device, cfg)})

    def state():
        validate_history(cfg, epoch, step, history, best)
        return {'kind': 'training', 'schema_version': SCHEMA_VERSION, 'identity': ident, 'model_config': model.config.to_dict(),
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'rng': rng_state(sampler), 'epoch': epoch, 'step': step,
                'history': history, 'best': best}

    if not resume:
        save_checkpoint(output / 'last.pt', state())
        reconcile_records(output, ident, epoch, step, history, best)
    print(f'TRAIN_READY seed={seed} initialization_seconds={time.perf_counter()-initialization_started:.2f} '
          f'effective_batch={cfg["effective_batch_size"]} micro_batch={cfg["micro_batch_size"]} '
          f'completed_epoch={epoch} completed_step={step}', flush=True)
    if epoch == cfg['max_epochs']:
        return complete_run(output, ident, epoch, step, history, best, 'max_epochs')
    if epoch and epoch - best['epoch'] == cfg['early_stopping_patience']:
        return complete_run(output, ident, epoch, step, history, best, 'early_stopping')
    if epoch and epoch - best['epoch'] > cfg['early_stopping_patience']:
        raise ValueError('Resume checkpoint has already exceeded early-stopping patience')
    targets = torch.tensor(codec.candidate_ids, dtype=torch.long, device=device)
    stopping_reason = None
    for next_epoch in range(epoch + 1, cfg['max_epochs'] + 1):
        started, loss_sum, length_counts = time.time(), 0.0, Counter()
        samples = cfg['samples_per_epoch']
        print(f'EPOCH_START epoch={next_epoch} samples={samples}', flush=True)
        for offset in range(0, samples, cfg['effective_batch_size']):
            questions = [sampler.sample() for _ in range(min(cfg['effective_batch_size'], samples - offset))]
            length_counts.update(q['gap_length'] for q in questions)
            loss, grad = update(model, codec, questions, cfg, device, optimizer, scaler, targets)
            loss_sum += loss
            step += 1
            scheduler.step()  # Only after a finite, successful optimizer update.
            if step % 100 == 0:
                print(f'TRAIN epoch={next_epoch} samples={offset + len(questions)}/{samples} '
                      f'loss={loss_sum/(offset+len(questions)):.5f} lr={scheduler.get_last_lr()[0]:.8f} '
                      f'elapsed_seconds={time.time()-started:.2f}', flush=True)
        train_seconds = time.time() - started
        validation_started = time.time()
        print(f'VALIDATION_START epoch={next_epoch} questions={len(public)} '
              f'train_seconds={train_seconds:.2f}', flush=True)
        metrics, predictions = evaluate(model, codec, public, answers, cfg, device, dev_encoded)
        validation_seconds = time.time() - validation_started
        epoch = next_epoch
        improved = better(metrics, best['metrics'] if best else None)
        if improved:
            candidate = {'kind': 'best', 'schema_version': SCHEMA_VERSION, 'identity': ident,
                         'model_config': model.config.to_dict(), 'model': model.state_dict(), 'epoch': epoch, 'metrics': metrics}
            best = stage_best(output, candidate, predictions, save_checkpoint)
        history.append({'epoch': epoch, 'step': step, 'train_loss': loss_sum / samples,
                        'sampled_lengths': dict(length_counts), 'validation': metrics,
                        'seconds': time.time() - started, 'train_seconds': train_seconds,
                        'validation_seconds': validation_seconds, 'new_best': improved})
        save_checkpoint(output / 'last.pt', state())
        reconcile_records(output, ident, epoch, step, history, best)
        print(f'EPOCH_DONE epoch={epoch} Macro5-EM@1={metrics["macro5_em1"]:.6f} '
              f'Macro5-Hit@5={metrics["macro5_hit5"]:.6f} best_epoch={best["epoch"]}', flush=True)
        if epoch < cfg['max_epochs'] and epoch - best['epoch'] == cfg['early_stopping_patience']:
            stopping_reason = 'early_stopping'
            print(f'EARLY_STOP epoch={epoch} best_epoch={best["epoch"]} '
                  f'patience={cfg["early_stopping_patience"]}', flush=True)
            break
    if stopping_reason is None:
        stopping_reason = 'max_epochs'
    complete_run(output, ident, epoch, step, history, best, stopping_reason)


def inspect_checkpoint(project, data, path, device, checkpoint=None):
    ckpt = load_checkpoint(path) if checkpoint is None else checkpoint
    cfg = evaluation_config(ckpt['identity']['cfg'])
    validate_config(cfg)
    if torch.device(device).type == 'cuda':
        require_cuda_precision(device, cfg)
    require_runtime(ckpt['identity'])
    if ckpt['identity']['data_sha256'] != verify_data(data):
        raise ValueError('Checkpoint/data mismatch')
    if not compatible_source(project, ckpt['identity']['code_sha256'], source_fingerprint(project)):
        raise ValueError('Evaluation code differs from training code; do not silently change protocol')
    codec = Codec(data, cfg)
    model = model_from_checkpoint(ckpt, device)
    model.eval()
    return ckpt, cfg, codec, model


def evaluate_checkpoint(project, data, path, output, device, suites, confirm_final=False, training_runs=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new evaluation output directory')
    is_test = any(s.startswith('test') for s in suites)
    if not suites or len(suites) != len(set(suites)) or not set(suites) <= {'dev_full', *FORMAL_TEST_SUITES}:
        raise ValueError('Unsupported or duplicate evaluation suites')
    if is_test:
        if not confirm_final:
            raise ValueError('Formal Test requires explicit confirm_final after all model selection')
        device = require_formal_cuda(device, FORMAL_CONFIG, 'Test')
        if suites != FORMAL_CONFIG['test_suites']:
            raise ValueError('Formal Test must evaluate the current manuscript suites in fixed order')
        # Batch gate applies to direct CLI/API calls too, not just the controller.
        # All three shared locks remain held until generation/acceptance ends.
        with completed_training_batch(path, training_runs) as (checkpoint, completion, batch):
            return _evaluate_checkpoint(project, data, path, output, device, suites, checkpoint, completion, batch)
    return _evaluate_checkpoint(project, data, path, output, device, suites)


def _evaluate_checkpoint(project, data, path, output, device, suites, checkpoint=None, completion=None, training_batch=None):
    ckpt, cfg, codec, model = inspect_checkpoint(project, data, path, device, checkpoint)
    eval_runtime = evaluation_environment(device, cfg)
    output.mkdir(parents=True)
    results = {}
    for suite in suites:
        public = benchmark_public(data, suite)
        answers = lambda suite=suite, public=public: benchmark_gold(data, suite, public)
        metrics, predictions = evaluate(model, codec, public, answers, cfg, device)
        results[suite] = metrics
        write_rows(output / (suite + '_predictions.jsonl'), predictions)
    report = {'seed': ckpt['identity']['seed'], 'epoch': ckpt['epoch'],
                                  'identity': ckpt['identity'], 'checkpoint_sha256': sha(path), 'suites': results,
                                  'evaluation_runtime': eval_runtime,
                                  'completion_sha256': digest(completion) if completion else None,
                                  'training_completion': completion}
    if completion is not None:
        from .evaluation_artifacts import ARTIFACT_SCHEMA, prediction_manifest, require_test_artifacts
        report.update(test_artifact_schema_version=ARTIFACT_SCHEMA,
                      training_batch_completion=training_batch, training_batch_sha256=digest(training_batch),
                      prediction_artifacts=prediction_manifest(output, data))
    write(output / 'metrics.json', report)
    if completion is not None:
        require_test_artifacts(output, data=data, project=project)
    print(results, flush=True)
