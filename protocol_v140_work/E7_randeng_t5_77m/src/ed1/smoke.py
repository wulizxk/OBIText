"""Bounded implementation checks; never a formal experimental result."""
from pathlib import Path

import torch

from .data import Codec, SpanSampler
from .engine import (seed_all, identity, update, evaluate, rng_state, restore_rng,
                     save_checkpoint, inspect_checkpoint, scheduler_for)
from .io import write
from .model import load_pretrained, optimizer_for, scaler_for
from .runtime import evaluation_environment, require_cuda_precision


def smoke(project, data, weights, output, cfg, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Smoke output exists; choose a new directory')
    if device.type == 'cuda':
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
                if codec.mapping['target_unit_to_model_id'][t] >= 32596}
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
        raise AssertionError('New untied output head row failed to update')
    # All bounded smoke questions come from TRAIN only. Never read Test gold.
    chosen = []
    for length in range(1, 6):
        item = sampler.pools[length][0]
        row, start = sampler.rows[item['source_id']], item['starts'][0]
        chosen.append({'source_id': item['source_id'], 'group_id': row['group_id'], 'gap_length': length,
                       'prefix_units': row['model_tokens'][:start], 'suffix_units': row['model_tokens'][start+length:],
                       'gold_units': row['model_tokens'][start:start+length]})
    for side in ('initial', 'final'):
        for item in sampler.pools[1]:
            row = sampler.rows[item['source_id']]
            start = 0 if side == 'initial' else len(row['model_tokens']) - 1
            if start in item['starts']:
                chosen.append({'source_id': item['source_id'], 'group_id': row['group_id'], 'gap_length': 1,
                               'prefix_units': row['model_tokens'][:start], 'suffix_units': row['model_tokens'][start+1:],
                               'gold_units': row['model_tokens'][start:start+1], 'boundary_side': side})
                break
        else:
            raise AssertionError('Train has no eligible boundary fixture')
    answers = {}
    for index, q in enumerate(chosen):
        q['example_id'] = f'TRAIN-SMOKE-{index}'
        answers[q['example_id']] = q.pop('gold_units')
    gold = lambda: answers
    metrics, predictions = evaluate(model, codec, chosen, gold, cfg, device)
    state = rng_state(sampler)
    next_questions = [sampler.sample() for _ in range(8)]
    restore_rng(state, sampler)
    if next_questions != [sampler.sample() for _ in range(8)]:
        raise AssertionError('Sampler RNG restoration failed')
    # Real-weight checkpoint serialization and evaluation round trip.
    checkpoint = {'kind': 'smoke', 'identity': ident, 'model_config': model.config.to_dict(),
                  'model': model.state_dict(), 'rng': state, 'epoch': 0}
    save_checkpoint(output / 'smoke.pt', checkpoint)
    _, _, restored_codec, restored = inspect_checkpoint(project, data, output / 'smoke.pt', device)
    _, after_predictions = evaluate(restored, restored_codec, chosen, gold, cfg, device)
    if predictions != after_predictions:
        raise AssertionError('Checkpoint predictions changed after reload')
    report = {'kind': 'smoke_only_not_paper_metrics', 'status': 'PASS', 'optimizer_updates': 2,
              'train_samples_per_update': len(questions), 'losses': losses,
              'gradient_norms': gradient_norms, 'scheduler_steps': scheduler.last_epoch,
              'grad_scaler_enabled': scaler.is_enabled(), 'parameter_dtype': 'torch.float32',
              'new_target_embedding_updated': row_changed, 'new_target_id_checked': min(new_rows),
              'new_untied_output_row_updated': output_changed,
              'train_questions_checked': 7, 'boundary_questions_checked': 2,
              'test_answers_used': False,
              'gap_lengths_checked': [1, 2, 3, 4, 5],
              'boundary_sides_checked': ['initial', 'final'],
              'checkpoint_predictions_identical': True, 'sampler_restore_identical': True,
              'pretrained_loading': info, 'device': str(device),
              'peak_cuda_memory_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
              'identity': ident, 'evaluation_runtime': evaluation_environment(device, cfg)}
    write(output / 'smoke_report.json', report)
    print(report, flush=True)
