"""Bounded TRAIN-only throughput/profile; explicit GPU consent and no occupied GPU.

Execution success is not lossless equivalence. Changed scores/ranks are reported
as CHANGED. No Test/Dev data, formal outputs, model selection or automatic tuning.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]

import torch
from ed3.data import Codec, SpanSampler
from ed3.engine import update, seed_all
from ed3.io import read, write, write_rows, digest, source_fingerprint, verify_data, sha
from ed3.model import load_pretrained, optimizer_for, scaler_for
from ed3.runtime import evaluation_environment, require_cuda_precision
from ed3.protocol import validate_formal_config
from ed3.formal_controller import require_training_gpus_free
from ed3.diagnostics import train_fixtures, generate_public, compare_predictions, GenerationProfile
from ed3.projection_cache import EvaluationProjectionCache


def describe(values):
    if not values or any(v <= 0 for v in values):
        raise ValueError('Timing samples must be positive')
    return {'repeats': len(values), 'mean_seconds': statistics.mean(values),
            'median_seconds': statistics.median(values), 'stdev_seconds': statistics.pstdev(values),
            'min_seconds': min(values), 'max_seconds': max(values), 'individual_seconds': values}


def memory():
    return {'allocated_mib': torch.cuda.memory_allocated() / 2**20,
            'reserved_mib': torch.cuda.memory_reserved() / 2**20,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}


def resources():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.total,memory.used,memory.free',
                             '--format=csv'], text=True, capture_output=True, timeout=15, check=True)
    return {'gpu': result.stdout, 'meminfo': Path('/proc/meminfo').read_text()}


def checkpoint_report(path, report):
    write(path / 'performance.json', report)


def run_probe(args):
    # Repeat ALL worker gates; internal worker dispatch cannot skip resource checks.
    if args.output.exists():
        raise FileExistsError(args.output)
    if any((ROOT / 'outputs').glob('ed3a_*')):
        raise RuntimeError('Formal E9 outputs exist; do not run this probe alongside a formal run')
    require_training_gpus_free()
    cfg = validate_formal_config(read(ROOT / 'configs/ed3a.json'))
    data = ROOT.parent / 'shared/data/v1.5.0'
    data_sha = verify_data(data)
    device = torch.device('cuda:0')
    require_cuda_precision(device, cfg)
    args.output.mkdir(parents=True)
    report = {'kind': 'TRAIN_ONLY_PERFORMANCE_NOT_FORMAL_METRICS', 'status': 'RUNNING',
              'code_sha256': source_fingerprint(ROOT), 'probe_sha256': sha(Path(__file__)),
              'data_sha256': data_sha, 'base_config': cfg, 'training': [], 'evaluation': [],
              'test_answers_used': False, 'steps': args.steps, 'warmup': args.warmup,
              'eval_repeats': args.repeats, 'wall_budget_seconds': args.max_seconds,
              'resources_before': resources(),
              'training_equivalence': 'NOT_CLAIMED: changing micro-batch changes dropout trajectories'}
    checkpoint_report(args.output, report)
    try:
        seed_all(42)
        codec, sampler = Codec(data, cfg), SpanSampler(data, 42)
        batches = [[sampler.sample() for _ in range(64)] for _ in range(args.steps + args.warmup)]
        report['train_fixture_sha256'] = digest(batches)
        public, answers = train_fixtures(sampler, codec, cfg, args.questions_per_length, include_stress=False)
        encoding_started = time.perf_counter()
        encoded = [codec.encode(q) for q in public]
        report['public_encoding_seconds_excluded_from_throughput'] = time.perf_counter() - encoding_started
        report['public_fixture_sha256'] = digest(public)
        write_rows(args.output / 'train_public.jsonl', public)
        write(args.output / 'train_answers.json', answers)
        # Synthetic max128 stress made solely from a TRAIN target, no extra data.
        token = batches[0][0]['gold_units'][0]
        stress = {'prefix_units': [token], 'suffix_units': [], 'gap_length': 5, 'gold_units': [token]*5}
        overhead = len(codec.encode(stress)['input_ids']) - 1
        stress['prefix_units'] *= cfg['max_input_length'] - overhead
        assert len(codec.encode(stress)['input_ids']) == 128
        model, info = load_pretrained(ROOT / 'weights/ED3', codec, cfg)
        initial = copy.deepcopy(model.state_dict())  # Preserve alias sharing; do not clone four tied tables.
        model.to(device)
        report.update(fixed_weights_loading=info, runtime=evaluation_environment(device, cfg),
                      stress_encoded_length=128)
        targets = torch.tensor(codec.target_ids, dtype=torch.long, device=device)
        candidates = torch.tensor(codec.candidate_ids, dtype=torch.long, device=device)
        for micro in (8, 16, 32, 64):
            config = dict(cfg, micro_batch_size=micro)
            seed_all(42); model.load_state_dict(initial)
            optimizer, scaler = optimizer_for(model, config), scaler_for(device, config)
            seconds, losses, norms = [], [], []
            entry = {'micro_batch_size': micro, 'effective_batch_size': 64,
                     'correctness_status': 'FINITE_UPDATES_ONLY_NOT_TRAINING_TRAJECTORY_EQUIVALENCE'}
            try:
                for index, questions in enumerate(batches):
                    if index == args.warmup:
                        torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize(); started = time.perf_counter()
                    loss, norm = update(model, codec, questions, config, device, optimizer, scaler, candidates)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                    if index >= args.warmup:
                        seconds.append(elapsed); losses.append(loss/64); norms.append(norm)
                entry.update(execution_status='COMPLETE', timing=describe(seconds),
                             samples_per_second=64/statistics.mean(seconds), losses=losses, gradient_norms=norms,
                             representative_memory=memory())
                torch.cuda.reset_peak_memory_stats()
                loss, norm = update(model, codec, [stress]*64, config, device, optimizer, scaler, candidates)
                torch.cuda.synchronize()
                entry.update(stress_memory=memory(), stress_loss=loss/64, stress_gradient_norm=norm)
            except torch.cuda.OutOfMemoryError as exc:
                entry.update(execution_status='OOM', error=str(exc))
            finally:
                model.zero_grad(set_to_none=True)
                del optimizer, scaler
                gc.collect(); torch.cuda.empty_cache()
            report['training'].append(entry)
            checkpoint_report(args.output, report)
            print('TRAIN_VARIANT_RESULT', json.dumps(entry), flush=True)

        # Evaluation variants all use exactly the same original, untrained weights.
        seed_all(42); model.load_state_dict(initial); model.eval()
        # Reference keeps the old uncached projection path at frozen Q32/F64.
        # The reference run is not included in timing or used to tune on Test.
        baseline = generate_public(model, codec, public, encoded, cfg, device,
                                   targets=targets, candidates=candidates, use_projection_cache=False)
        write_rows(args.output / 'uncached_reference.jsonl', baseline)
        report['uncached_reference_sha256'] = sha(args.output / 'uncached_reference.jsonl')
        groups = {L: [i for i, q in enumerate(public) if q['gap_length'] == L] for L in range(1, 6)}
        for qb, fb in ((32, 64), (64, 128), (128, 256)):
            config = dict(cfg, eval_question_batch_size=qb, eval_forward_batch_size=fb)
            entry = {'question_batch': qb, 'forward_batch': fb, 'comparisons': [], 'timed_runs': [],
                     'correctness_status': 'NOT_CHECKED'}
            per_length = {L: [] for L in groups}
            try:
                # Each variant gets its own shape warmup; no warmup goes into timers.
                for _ in range(args.warmup):
                    for L, indices in groups.items():
                        ix = indices[:qb]
                        generate_public(model, codec, [public[i] for i in ix], [encoded[i] for i in ix],
                                        config, device, targets=targets, candidates=candidates, use_projection_cache=True)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                for repeat in range(args.repeats):
                    predictions = [None] * len(public)
                    total_started = time.perf_counter()
                    with EvaluationProjectionCache(model) as projection_cache:
                        for L, indices in groups.items():
                            torch.cuda.synchronize(); started = time.perf_counter()
                            output = generate_public(model, codec, [public[i] for i in indices], [encoded[i] for i in indices],
                                                     config, device, targets=targets, candidates=candidates,
                                                     use_projection_cache=True, projection_cache=projection_cache)
                            for i, prediction in zip(indices, output):
                                predictions[i] = prediction
                            torch.cuda.synchronize()
                            per_length[L].append(time.perf_counter()-started)
                    elapsed = time.perf_counter()-total_started
                    comparison = compare_predictions(baseline, predictions, public, answers, codec.units)
                    name = f'predictions_q{qb}_f{fb}_r{repeat}.jsonl'
                    write_rows(args.output / name, predictions)
                    entry['comparisons'].append(comparison)
                    entry['timed_runs'].append({'seconds': elapsed, 'prediction_file': name,
                                                'prediction_sha256': sha(args.output / name),
                                                'output_projection_cache': projection_cache.report()})
                entry.update(execution_status='COMPLETE',
                    correctness_status='IDENTICAL' if all(x['status']=='IDENTICAL' for x in entry['comparisons']) else 'CHANGED',
                    timing=describe([x['seconds'] for x in entry['timed_runs']]), memory=memory(),
                    questions=len(public), questions_per_second=len(public)/statistics.mean(x['seconds'] for x in entry['timed_runs']),
                    by_length={str(L): dict(describe(times), questions=len(groups[L]),
                                             questions_per_second=len(groups[L])/statistics.mean(times)) for L, times in per_length.items()})
                # Profile separately: extra synchronizations must not inflate the throughput comparison.
                profile = GenerationProfile(device)
                with profile.measure('total'):
                    profiled = generate_public(model, codec, public, encoded, config, device, profile, targets, candidates,
                                               use_projection_cache=True)
                entry['separate_profile'] = profile.report()
                entry['profile_vs_last_timed_comparison'] = compare_predictions(predictions, profiled, public, answers, codec.units)
                if entry['profile_vs_last_timed_comparison']['status'] != 'IDENTICAL':
                    entry['correctness_status'] = 'CHANGED'
            except torch.cuda.OutOfMemoryError as exc:
                entry.update(execution_status='OOM', error=str(exc), correctness_status='NOT_FULLY_CHECKED')
            report['evaluation'].append(entry)
            checkpoint_report(args.output, report)
            print('EVAL_VARIANT_RESULT', json.dumps({'batch': [qb, fb], 'execution':entry['execution_status'],
                                                    'correctness':entry['correctness_status']}), flush=True)
            gc.collect(); torch.cuda.empty_cache()
        report['status'] = ('COMPLETE' if all(x['execution_status']=='COMPLETE' for x in report['training']+report['evaluation'])
                            else 'COMPLETE_WITH_FAILURES')
        report['lossless_eval_variants'] = [[x['question_batch'], x['forward_batch']] for x in report['evaluation']
                                            if x['correctness_status']=='IDENTICAL']
        report['resources_after'] = resources()
        checkpoint_report(args.output, report)
        return 0 if report['status']=='COMPLETE' else 1
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc))
        checkpoint_report(args.output, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-gpu', action='store_true', help='Explicit consent for this bounded GPU diagnosis')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--questions-per-length', type=int, default=128)
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.allow_gpu:
        raise ValueError('Explicit --allow-gpu is required; no GPU work started')
    if (not 2 <= args.steps <= 16 or not 1 <= args.warmup <= 3 or not 2 <= args.repeats <= 5
            or not 32 <= args.questions_per_length <= 128 or not 60 <= args.max_seconds <= 1800):
        raise ValueError('Probe must remain bounded')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args._worker:
        # Also bound direct internal-worker invocation. Supervisor below supplies
        # a hard process timeout if a native call delays Python signal handling.
        def deadline(*_):
            raise TimeoutError('Performance probe wall-clock budget exceeded')
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
        try:
            return run_probe(args)
        finally:
            signal.alarm(0)
    original_args = list(sys.argv[1:] if argv is None else argv)
    proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *original_args, '--_worker'])
    try:
        return proc.wait(timeout=args.max_seconds)
    except subprocess.TimeoutExpired:
        proc.terminate()  # Only this probe's own child, never another experiment.
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
        write(args.output / 'WALL_TIMEOUT.json', {'status':'TIMEOUT','seconds':args.max_seconds,
                                                'worker_pid':proc.pid,'formal_job':False})
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
