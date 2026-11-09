"""Frozen three-seed ED1-A orchestration, included in the source fingerprint."""
import argparse
from datetime import datetime, timezone
import os
import math
from pathlib import Path
import subprocess
import traceback

from .io import read, sha, source_fingerprint, verify_data, write
from .model import verify_weights
from .run_state import require_completed_run
from .runtime import runtime_versions, configure_execution, evaluation_environment, require_cuda_precision
from .protocol import DATA_MANIFEST_SHA256, FORMAL_CONFIG_SHA256, validate_formal_config


ROOT = Path(__file__).resolve().parents[2]
SEEDS = [(42, 0), (3407, 0), (2026, 0)]
OUTPUTS = ROOT / 'outputs'
CONTROL = OUTPUTS / 'ed1a_v150_formal_earlystop'
PYTHON = str(ROOT / '.venv/bin/python')
SUMMARY = OUTPUTS / 'ed1a_v150_test_summary.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def check_frozen(expected_code_sha):
    if not isinstance(expected_code_sha, str) or len(expected_code_sha) != 64:
        raise ValueError('The authorized source fingerprint is missing or malformed')
    if source_fingerprint(ROOT) != expected_code_sha:
        raise ValueError('Source fingerprint differs from the verified early-stop version')
    if sha(ROOT / 'configs/ed1a.json') != FORMAL_CONFIG_SHA256:
        raise ValueError('Configuration differs from the authorized early-stop protocol')
    if verify_data(ROOT.parent / 'shared/data/v1.5.0') != DATA_MANIFEST_SHA256:
        raise ValueError('Frozen data identity differs')
    cfg = validate_formal_config(read(ROOT / 'configs/ed1a.json'))
    verify_weights(ROOT.parent / 'shared/pretrained/ED1', cfg)


def require_training_gpus_free():
    occupied = {}
    for gpu in sorted({gpu for _, gpu in SEEDS}):
        result = subprocess.run(
            ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,process_name',
             '--format=csv,noheader,nounits'],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(f'Cannot verify GPU {gpu}: {result.stdout.strip()}')
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if rows:
            occupied[gpu] = rows
    if occupied:
        raise RuntimeError('Refusing to overlap another compute job on ED1 GPUs: ' + repr(occupied))


def require_gpu_smoke(expected_code_sha):
    report = read(ROOT / 'verification/gpu_smoke_v150/smoke_report.json')
    cfg = read(ROOT / 'configs/ed1a.json')
    validate_formal_config(cfg)
    ident = report.get('identity', {})
    environment = report.get('evaluation_runtime', {})
    finite_diagnostics = all(
        isinstance(report.get(key), list) and len(report[key]) == 2
        and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in report[key])
        for key in ('losses', 'gradient_norms'))
    if (report.get('status') != 'PASS' or report.get('optimizer_updates') != 2
            or report.get('kind') != 'smoke_only_not_paper_metrics' or not finite_diagnostics
            or report.get('train_samples_per_update') != cfg['effective_batch_size']
            or report.get('new_target_embedding_updated') is not True
            or report.get('new_untied_output_row_updated') is not True
            or report.get('checkpoint_predictions_identical') is not True
            or report.get('sampler_restore_identical') is not True
            or report.get('grad_scaler_enabled') is not False
            or report.get('parameter_dtype') != 'torch.float32'
            or report.get('scheduler_steps') != 2
            or report.get('test_answers_used') is not False
            or report.get('gap_lengths_checked') != [1, 2, 3, 4, 5]
            or report.get('boundary_sides_checked') != ['initial', 'final']):
        raise ValueError('Required real-weight GPU smoke checks have not all passed')
    if (ident.get('cfg') != cfg or ident.get('code_sha256') != expected_code_sha
            or ident.get('data_sha256') != DATA_MANIFEST_SHA256
            or ident.get('weights_sha256') != verify_weights(ROOT.parent / 'shared/pretrained/ED1', cfg)
            or ident.get('runtime') != runtime_versions()
            or environment.get('versions') != ident.get('runtime')
            or environment.get('execution_policy') != ident.get('execution_policy')
            or environment.get('effective_precision') != cfg['precision']
            or environment.get('hardware', {}).get('type') != 'cuda'):
        raise ValueError('GPU smoke identity/runtime/precision differs from this formal deployment')
    require_cuda_precision('cuda:0', cfg)
    configure_execution()
    if environment != evaluation_environment('cuda:0', cfg):
        raise ValueError('GPU smoke hardware/execution policy differs from the current CUDA device')


def command(stage, seed):
    base = [PYTHON, '-B', '-u', 'run.py']
    if stage == 'train':
        return base + ['train', '--seed', str(seed), '--output', str(OUTPUTS / f'ed1a_v150_s{seed}')]
    if stage == 'test':
        return base + ['test', '--checkpoint', str(OUTPUTS / f'ed1a_v150_s{seed}/best.pt'),
                       '--output', str(OUTPUTS / f'ed1a_v150_s{seed}_test'), '--confirm-final']
    if stage == 'summarize':
        return base + ['summarize', '--evaluations'] + [str(OUTPUTS / f'ed1a_v150_s{s}_test') for s, _ in SEEDS] + [
            '--output', str(SUMMARY)]
    raise ValueError('Unknown formal stage: ' + repr(stage))


def save_status(state):
    state['updated_utc'] = now()
    write(CONTROL / 'status.json', state)


def phase(stage, state):
    state['stage'] = stage
    state['jobs'][stage] = {}
    save_status(state)
    jobs = SEEDS if stage != 'summarize' else [(None, None)]
    if stage == 'test':
        for seed, _ in SEEDS:
            checkpoint, _ = require_completed_run(OUTPUTS / f'ed1a_v150_s{seed}/best.pt')
            del checkpoint
    for seed, gpu in jobs:
        if gpu is not None:
            require_training_gpus_free()
        key = str(seed) if seed is not None else 'summary'
        log_path = CONTROL / f'{stage}_{key}.log'
        cmd = command(stage, seed)
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
                   CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '')
        with log_path.open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        job = {'pid': proc.pid, 'seed': seed, 'gpu': gpu, 'command': cmd, 'log': str(log_path),
               'status': 'running', 'started_utc': now()}
        state['jobs'][stage][key] = job
        save_status(state)
        print(f'START {stage} seed={seed} GPU={gpu} pid={proc.pid}', flush=True)
        try:
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
        job.update(exit_code=code, status='succeeded' if code == 0 else 'failed', finished_utc=now())
        save_status(state)
        if code:
            return False
        if stage == 'train':
            checkpoint, _ = require_completed_run(OUTPUTS / f'ed1a_v150_s{seed}/best.pt')
            del checkpoint
    return True


def main(argv=None, expected_code_sha=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Verify frozen files and print commands; start no jobs')
    parser.add_argument('--execute', action='store_true', help='Explicitly start all three seeds, then Test and summary')
    args = parser.parse_args(argv)
    check_frozen(expected_code_sha)
    reserved = [CONTROL, SUMMARY]
    reserved += [OUTPUTS / f'ed1a_v150_s{seed}{suffix}' for seed, _ in SEEDS for suffix in ['', '_test']]
    if any(path.exists() for path in reserved):
        raise FileExistsError('A reserved output exists; never overwrite or silently resume')
    for stage in ['train', 'test', 'summarize']:
        for seed, gpu in SEEDS if stage != 'summarize' else [(None, None)]:
            print({'stage': stage, 'seed': seed, 'gpu': gpu, 'command': command(stage, seed)}, flush=True)
    if args.plan or not args.execute:
        return 0
    require_gpu_smoke(expected_code_sha)
    require_training_gpus_free()
    CONTROL.mkdir(parents=True)
    state = {'status': 'running', 'stage': 'starting', 'controller_pid': os.getpid(),
             'started_utc': now(), 'project': str(ROOT), 'code_sha256': expected_code_sha,
             'config_sha256': FORMAL_CONFIG_SHA256, 'data_sha256': DATA_MANIFEST_SHA256,
             'config': read(ROOT / 'configs/ed1a.json'), 'jobs': {},
             'policy': ('Each seed runs at most 50 epochs and stops after 5 consecutive full-Dev '
                        'non-improvements. All three valid completions precede any final Test.')}
    save_status(state)
    try:
        for stage in ['train', 'test', 'summarize']:
            check_frozen(expected_code_sha)
            if not phase(stage, state):
                state.update(status='failed', failed_stage=stage, finished_utc=now())
                save_status(state)
                return 1
        report = read(SUMMARY)
        if not report['formal_three_seed_complete']:
            raise ValueError('Summary does not cover the required three distinct seeds')
        state.update(status='complete', stage='complete', summary=str(SUMMARY), finished_utc=now())
        save_status(state)
        print('FORMAL_TRAIN_TEST_SUMMARY_COMPLETE', flush=True)
        return 0
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_utc=now())
        save_status(state)
        traceback.print_exc()
        raise
