"""Bounded implementation checks; never a formal experimental result."""
from pathlib import Path
import time

import torch

from .data import Codec, SpanSampler, benchmark
from .engine import (seed_all, identity, update, evaluate, rng_state, restore_rng,
                     save_checkpoint, inspect_checkpoint)
from .io import write
from .model import load_pretrained, optimizer_for
from .runtime import evaluation_environment


def smoke(project, data, weights, output, cfg, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Smoke output exists; choose a new directory')
    ident = identity(project, data, weights, cfg, 42)
    output.mkdir(parents=True)
    seed_all(42)
    codec, sampler = Codec(data, cfg), SpanSampler(data, 42)
    model, info = load_pretrained(weights, codec, cfg)
    model.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['precision'] == 'fp16', init_scale=1024.0)
    questions = [sampler.sample() for _ in range(cfg['effective_batch_size'])]
    new_rows = {codec.mapping['target_unit_to_model_id'][t] for q in questions for t in q['gold_units']
                if codec.mapping['target_unit_to_model_id'][t] >= 23292}
    if not new_rows:
        raise AssertionError('Smoke must include a newly added target embedding')
    before = model.get_input_embeddings().weight[min(new_rows)].detach().clone()
    losses = []
    for _ in range(2):
        loss, grad = update(model, codec, questions, cfg, device, optimizer, scaler)
        losses.append(loss / len(questions))
    row_changed = not torch.equal(before, model.get_input_embeddings().weight[min(new_rows)].detach())
    if not row_changed:
        raise AssertionError('New target embedding failed to update')
    public, gold = benchmark(data, 'dev_full')
    prefix_public, prefix_gold = benchmark(data, 'test_prefix')
    suffix_public, suffix_gold = benchmark(data, 'test_suffix')
    chosen = [next(q for q in public if q['gap_length'] == length) for length in range(1, 6)]
    chosen += [next(q for q in prefix_public if q['boundary_side'] == 'prefix'),
               next(q for q in suffix_public if q['boundary_side'] == 'suffix')]
    gold = {**gold, **prefix_gold, **suffix_gold}
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
              'new_target_embedding_updated': row_changed, 'new_target_id_checked': min(new_rows),
              'dev_questions_checked': 5, 'boundary_questions_checked': 2,
              'gap_lengths_checked': [1, 2, 3, 4, 5],
              'boundary_sides_checked': ['prefix', 'suffix'],
              'checkpoint_predictions_identical': True, 'sampler_restore_identical': True,
              'pretrained_loading': info, 'device': str(device),
              'peak_cuda_memory_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
              'identity': ident, 'evaluation_runtime': evaluation_environment(device, cfg)}
    write(output / 'smoke_report.json', report)
    print(report, flush=True)
