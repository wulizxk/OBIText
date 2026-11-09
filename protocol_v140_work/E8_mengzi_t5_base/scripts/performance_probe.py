"""Bounded TRAIN-only throughput/memory probe; never a formal experiment.

Compares batch shapes with identical starting weights and sampled examples.
No Test/Dev answers, formal checkpoints, optimizer/precision changes or jobs.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'shared'))

import torch
from ed2.data import Codec, SpanSampler
from ed2.engine import update, seed_all
from ed2.io import read, write, digest, source_fingerprint
from ed2.model import load_pretrained, optimizer_for, scaler_for
from ed2.runtime import evaluation_environment
from ed2.decoding import predict_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    if not 2 <= args.steps <= 16 or not 1 <= args.warmup <= 3:
        raise ValueError('Probe must remain bounded')
    if args.output.exists():
        raise FileExistsError(args.output)
    if any((ROOT / 'outputs').glob('ed2a_*')):
        raise RuntimeError('Formal E8 outputs exist; do not run this probe alongside or inside a formal run')
    occupied = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                               '--format=csv,noheader,nounits'], text=True,
                              capture_output=True, timeout=15, check=True).stdout.strip()
    if occupied:
        raise RuntimeError('GPU already occupied: ' + occupied)
    args.output.mkdir(parents=True)
    cfg = read(ROOT / 'configs/ed2a.json')
    data = ROOT.parent / 'shared/data/v1.5.0'
    device = torch.device('cuda:0')
    seed_all(42)
    print('LOADING_CODEC_AND_TRAIN', flush=True)
    codec, sampler = Codec(data, cfg), SpanSampler(data, 42)
    batches = [[sampler.sample() for _ in range(64)]
               for _ in range(args.steps + args.warmup)]
    fixture_hash = digest(batches)
    # Real longest TRAIN source with an eligible five-character gap.
    longest = max(sampler.pools[5], key=lambda item: len(sampler.rows[item['source_id']]['model_tokens']))
    row = sampler.rows[longest['source_id']]
    start = longest['starts'][0]
    stress = {'source_id': longest['source_id'], 'gap_length': 5,
              'prefix_units': row['model_tokens'][:start],
              'suffix_units': row['model_tokens'][start + 5:],
              'gold_units': row['model_tokens'][start:start + 5]}
    print('LOADING_REAL_FIXED_WEIGHTS', flush=True)
    model, info = load_pretrained(ROOT / 'weights/ED2', codec, cfg)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.to(device)
    report = {'kind': 'TRAIN_ONLY_PERFORMANCE_NOT_FORMAL_METRICS',
              'code_sha256': source_fingerprint(ROOT), 'base_config': cfg,
              'runtime': evaluation_environment(device, cfg), 'fixture_sha256': fixture_hash,
              'fixed_weights_loading': info, 'test_answers_used': False,
              'training': [], 'evaluation': [], 'steps': args.steps, 'warmup': args.warmup,
              'stress_encoded_length': len(codec.encode(stress)['input_ids'])}

    for micro in (8, 16, 32, 64):
        config = dict(cfg, micro_batch_size=micro)
        seed_all(42)
        model.load_state_dict(initial)
        optimizer, scaler = optimizer_for(model, config), scaler_for(device, config)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seconds, losses, norms = [], [], []
        print('TRAIN_VARIANT_START', micro, flush=True)
        try:
            for index, questions in enumerate(batches):
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss, norm = update(model, codec, questions, config, device, optimizer, scaler)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if index >= args.warmup:
                    seconds.append(elapsed); losses.append(loss / 64); norms.append(norm)
            representative_peak = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
            stress_loss, stress_norm = update(model, codec, [stress] * 64, config,
                                               device, optimizer, scaler)
            torch.cuda.synchronize()
            entry = {'micro_batch_size': micro, 'status': 'PASS', 'effective_batch_size': 64,
                     'mean_update_seconds': statistics.mean(seconds),
                     'median_update_seconds': statistics.median(seconds),
                     'samples_per_second': 64 / statistics.mean(seconds),
                     'individual_seconds': seconds, 'losses': losses, 'gradient_norms': norms,
                     'representative_peak_mib': representative_peak,
                     'stress_peak_mib': torch.cuda.max_memory_allocated() / 2**20,
                     'stress_loss': stress_loss / 64, 'stress_gradient_norm': stress_norm}
        except torch.cuda.OutOfMemoryError as exc:
            entry = {'micro_batch_size': micro, 'status': 'OOM', 'error': str(exc)}
        report['training'].append(entry)
        print('TRAIN_VARIANT_RESULT', json.dumps(entry), flush=True)
        write(args.output / 'performance.json', report)
        model.zero_grad(set_to_none=True)
        del optimizer, scaler
        gc.collect(); torch.cuda.empty_cache()

    # Public questions generated from TRAIN only, balanced length 1..5.
    public = []
    for length in range(1, 6):
        for item in sampler.pools[length][:128]:
            row = sampler.rows[item['source_id']]; start = item['starts'][0]
            public.append({'source_id': item['source_id'], 'gap_length': length,
                           'prefix_units': row['model_tokens'][:start],
                           'suffix_units': row['model_tokens'][start+length:]})
    seed_all(42); model.load_state_dict(initial); model.eval()
    baseline = None
    for question_batch, forward_batch in ((32, 64), (64, 128), (128, 256)):
        config = dict(cfg, eval_question_batch_size=question_batch, eval_forward_batch_size=forward_batch)
        torch.cuda.reset_peak_memory_stats()
        predictions = []
        torch.cuda.synchronize(); started = time.perf_counter()
        try:
            for length in range(1, 6):
                group = sorted([q for q in public if q['gap_length'] == length],
                               key=lambda q: len(codec.encode(q)['input_ids']))
                for start in range(0, len(group), question_batch):
                    predictions.extend(predict_batch(model, codec, group[start:start+question_batch], config, device))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if baseline is None:
                baseline = copy.deepcopy(predictions)
            same_top1 = sum(a['paths'][0] == b['paths'][0] for a, b in zip(baseline, predictions))
            same_topk = sum(a['paths'] == b['paths'] for a, b in zip(baseline, predictions))
            entry = {'question_batch': question_batch, 'forward_batch': forward_batch,
                     'status': 'PASS', 'questions': len(public), 'seconds': elapsed,
                     'questions_per_second': len(public) / elapsed,
                     'peak_mib': torch.cuda.max_memory_allocated() / 2**20,
                     'same_top1_as_baseline': same_top1, 'same_topk_as_baseline': same_topk}
        except torch.cuda.OutOfMemoryError as exc:
            entry = {'question_batch': question_batch, 'forward_batch': forward_batch,
                     'status': 'OOM', 'error': str(exc)}
        report['evaluation'].append(entry)
        print('EVAL_VARIANT_RESULT', json.dumps(entry), flush=True)
        write(args.output / 'performance.json', report)
        gc.collect(); torch.cuda.empty_cache()
    report['status'] = 'COMPLETE'
    write(args.output / 'performance.json', report)
    print('PERFORMANCE_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
