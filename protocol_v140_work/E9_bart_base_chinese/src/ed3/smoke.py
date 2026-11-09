"""Bounded implementation checks; never a formal experimental result."""
from pathlib import Path
import gc

import torch

from .data import Codec, SpanSampler
from .engine import (seed_all, identity, update, evaluate, rng_state, restore_rng,
                     save_checkpoint, inspect_checkpoint, scheduler_for)
from .io import write, write_rows, sha
from .model import load_pretrained, optimizer_for, scaler_for, autocast, per_sample_losses
from .runtime import evaluation_environment, require_cuda_precision
from .diagnostics import (GenerationProfile, generate_public, train_fixtures, compare_predictions,
                          require_identical, native_forward_reference)


def smoke(project, data, weights, output, cfg, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Smoke output exists; choose a new directory')
    if device.type == 'cuda':
        from .formal_controller import require_training_gpus_free
        require_training_gpus_free()
        from .shared_gpu import apply_shared_budget
        apply_shared_budget(device)
        require_cuda_precision(device, cfg)
    ident = identity(project, data, weights, cfg, 42)
    output.mkdir(parents=True)
    seed_all(42)
    codec, sampler = Codec(data, cfg), SpanSampler(data, 42)
    model, info = load_pretrained(weights, codec, cfg)
    model.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = optimizer_for(model, cfg)
    scaler = scaler_for(device, cfg)
    scheduler = scheduler_for(optimizer, cfg)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise AssertionError('All master parameters must remain FP32')
    questions = [sampler.sample() for _ in range(cfg['effective_batch_size'])]
    new_rows = {codec.mapping['target_unit_to_model_id'][t] for q in questions for t in q['gold_units']
                if codec.mapping['target_unit_to_model_id'][t] >= 51271}
    if not new_rows:
        raise AssertionError('Smoke must include a newly added target embedding')
    before = model.get_input_embeddings().weight[min(new_rows)].detach().clone()
    output_before = model.get_output_embeddings().weight[min(new_rows)].detach().clone()
    losses, gradient_norms = [], []
    for _ in range(2):
        loss, grad = update(model, codec, questions, cfg, device, optimizer, scaler)
        scheduler.step()
        losses.append(loss / len(questions))
        gradient_norms.append(grad)
    row_changed = not torch.equal(before, model.get_input_embeddings().weight[min(new_rows)].detach())
    if not row_changed:
        raise AssertionError('New target embedding failed to update')
    output_changed = not torch.equal(output_before, model.get_output_embeddings().weight[min(new_rows)].detach())
    if not output_changed:
        raise AssertionError('New tied output head row failed to update')
    # Shape-only stress check at the full supported input/answer lengths.
    # Uses a TRAIN target repeated as synthetic context, not Test or Dev gold.
    # Backpropagate without an optimizer/scheduler step: the two update checks
    # and checkpoint round-trip below still describe exactly two updates.
    stress_token = questions[0]['gold_units'][0]
    stress = {'prefix_units': [stress_token], 'suffix_units': [], 'gap_length': 5,
              'gold_units': [stress_token] * 5}
    overhead = len(codec.encode(stress)['input_ids']) - 1
    stress['prefix_units'] *= cfg['max_input_length'] - overhead
    stress_length = len(codec.encode(stress)['input_ids'])
    if stress_length != cfg['max_input_length']:
        raise AssertionError('Stress input must reach the supported maximum')
    optimizer.zero_grad(set_to_none=True)
    stress_batch = codec.collate([stress] * cfg['micro_batch_size'], device)
    stress_targets = torch.tensor(codec.candidate_ids, dtype=torch.long, device=device)
    with autocast(device, cfg):
        stress_losses = per_sample_losses(model, stress_batch, stress_targets)
        stress_loss = stress_losses.mean()
    if not bool(torch.isfinite(stress_losses).all() & torch.isfinite(stress_loss)):
        raise FloatingPointError('Non-finite maximum-shape smoke loss')
    stress_loss.backward()
    stress_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'],
                                               error_if_nonfinite=True)
    stress_norm_value = float(stress_norm)
    optimizer.zero_grad(set_to_none=True)
    del stress_batch, stress_targets, stress_losses, stress_loss, stress_norm
    # All bounded smoke questions come from TRAIN only. Never read Test gold.
    chosen, answers = train_fixtures(sampler, codec, cfg, per_length=32, include_stress=True)
    native_questions = [dict(next(q for q in chosen if q['gap_length'] == length)) for length in range(1, 6)]
    for q in native_questions:
        q['gold_units'] = answers[q['example_id']]
    native_reference = native_forward_reference(model, codec, native_questions, cfg, device)
    encoded = [codec.encode(q) for q in chosen]
    profile = GenerationProfile(device)
    with profile.measure('total'):
        raw_before = generate_public(model, codec, chosen, encoded, cfg, device, profile)
    shape_evidence = dict(profile.report(), question_count=len(chosen), gap_lengths=[1, 2, 3, 4, 5],
                          maximum_gap_length=5, boundary_sides=['initial', 'final'], test_answers_used=False)
    if (shape_evidence['max_encoder_shape'] != [cfg['eval_question_batch_size'], cfg['max_input_length']]
            or shape_evidence['max_decoder_shape'] != [cfg['eval_forward_batch_size'], 8]):
        raise AssertionError('Smoke did not actually reach the formal generation shapes')
    gold = lambda: answers
    graph_before = {}
    raw_eager = []
    metrics, predictions = evaluate(model, codec, chosen, gold, cfg, device,
                                    execution_report=graph_before, raw_output=raw_eager)
    eager_comparison = compare_predictions(raw_before, raw_eager, chosen, answers, codec.units)
    require_identical(eager_comparison)
    state = rng_state(sampler)
    next_questions = [sampler.sample() for _ in range(8)]
    restore_rng(state, sampler)
    if next_questions != [sampler.sample() for _ in range(8)]:
        raise AssertionError('Sampler RNG restoration failed')
    # Real-weight checkpoint serialization and evaluation round trip.
    checkpoint = {'kind': 'smoke', 'identity': ident, 'model_config': model.config.to_dict(),
                  'model': model.state_dict(), 'rng': state, 'epoch': 0}
    save_checkpoint(output / 'smoke.pt', checkpoint)
    # Do not retain the first model and Adam states while loading a second copy.
    scheduler_steps = scheduler.last_epoch
    del checkpoint, model, optimizer, scheduler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    _, _, restored_codec, restored = inspect_checkpoint(project, data, output / 'smoke.pt', device)
    graph_after = {}
    raw_after = []
    _, after_predictions = evaluate(restored, restored_codec, chosen, gold, cfg, device,
                                    execution_report=graph_after, raw_output=raw_after)
    checkpoint_comparison = compare_predictions(raw_before, raw_after, chosen, answers, codec.units)
    require_identical(checkpoint_comparison)
    write_rows(output / 'train_public.jsonl', chosen)
    write(output / 'train_gold.json', answers)
    for name, values in [('generation_before.jsonl', raw_before), ('generation_eager.jsonl', raw_eager),
                         ('generation_after.jsonl', raw_after)]:
        write_rows(output / name, values)
    write(output / 'native_forward_reference.json', native_reference)
    names = ('train_public.jsonl', 'train_gold.json', 'generation_before.jsonl', 'generation_eager.jsonl',
             'generation_after.jsonl', 'native_forward_reference.json')
    from .shared_gpu import applied_budget
    report = {'kind': 'smoke_only_not_paper_metrics', 'status': 'PASS', 'optimizer_updates': 2,
              'gpu_resource_policy': applied_budget(),
              'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else None,
              'train_samples_per_update': len(questions), 'losses': losses,
              'gradient_norms': gradient_norms, 'scheduler_steps': scheduler_steps,
              'grad_scaler_enabled': scaler.is_enabled(), 'parameter_dtype': 'torch.float32',
              'new_target_embedding_updated': row_changed, 'new_target_id_checked': min(new_rows),
              'new_tied_output_row_updated': output_changed,
              'max_shape_forward_backward_passed': True,
              'max_shape_batch_size': cfg['micro_batch_size'],
              'max_shape_input_length': stress_length, 'max_shape_gap_length': 5,
              'max_shape_gradient_norm': stress_norm_value,
              'max_shape_optimizer_updates': 0,
              'train_questions_checked': len(chosen), 'boundary_questions_checked': 2,
              'diagnostic_evidence_schema': 2,
              'projection_cache_reference': 'uncached_public_reference_vs_formal_cached_evaluation',
              'native_forward_reference': native_reference,
              'generation_shape_evidence': shape_evidence,
              'full_prediction_eager_comparison': eager_comparison,
              'full_prediction_checkpoint_comparison': checkpoint_comparison,
              'diagnostic_artifacts': {name: sha(output / name) for name in names},
              'test_answers_used': False,
              'gap_lengths_checked': [1, 2, 3, 4, 5],
              'boundary_sides_checked': ['initial', 'final'],
              'checkpoint_predictions_identical': True, 'sampler_restore_identical': True,
              'cuda_graph_eager_predictions_identical': True,
              'evaluation_execution_before_reload': graph_before,
              'evaluation_execution_after_reload': graph_after,
              'pretrained_loading': info, 'device': str(device),
              'peak_cuda_memory_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
              'identity': ident, 'evaluation_runtime': evaluation_environment(device, cfg)}
    write(output / 'smoke_report.json', report)
    print(report, flush=True)
