"""Frozen three-seed E3-A orchestration, included in the source fingerprint."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
import traceback

from .io import read, sha, source_fingerprint, verify_data, write
from .protocol import DATA_MANIFEST_SHA256, FORMAL_CONFIG_SHA256, validate_formal_config


ROOT = Path(__file__).resolve().parents[2]
SEEDS = [(42, 0), (3407, 1), (2026, 2)]
OUTPUTS = ROOT / 'outputs'
CONTROL = OUTPUTS / 'e3a_v150_formal_earlystop'
PYTHON = str(ROOT / '.venv/bin/python')
SUMMARY = OUTPUTS / 'e3a_v150_test_summary.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def check_frozen(expected_code_sha):
    if not isinstance(expected_code_sha, str) or len(expected_code_sha) != 64:
        raise ValueError('The authorized source fingerprint is missing or malformed')
    if source_fingerprint(ROOT) != expected_code_sha:
        raise ValueError('Source fingerprint differs from the verified early-stop version')
    if sha(ROOT / 'configs/e3a.json') != FORMAL_CONFIG_SHA256:
        raise ValueError('Configuration differs from the authorized early-stop protocol')
    if verify_data(ROOT.parent / 'shared/data/v1.5.0') != DATA_MANIFEST_SHA256:
        raise ValueError('Frozen data identity differs')
    validate_formal_config(read(ROOT / 'configs/e3a.json'))


def require_training_gpus_free():
    occupied = {}
    for _, gpu in SEEDS:
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
        raise RuntimeError('Refusing to overlap another compute job on E3 GPUs: ' + repr(occupied))


def command(stage, seed):
    base = [PYTHON, '-B', '-u', 'run.py']
    if stage == 'train':
        return base + ['train', '--seed', str(seed), '--output', str(OUTPUTS / f'e3a_v150_s{seed}')]
    if stage == 'test':
        return base + ['test', '--checkpoint', str(OUTPUTS / f'e3a_v150_s{seed}/best.pt'),
                       '--output', str(OUTPUTS / f'e3a_v150_s{seed}_test'), '--confirm-final']
    if stage == 'summarize':
        return base + ['summarize', '--evaluations'] + [str(OUTPUTS / f'e3a_v150_s{s}_test') for s, _ in SEEDS] + [
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
    active = []
    try:
        for seed, gpu in jobs:
            key = str(seed) if seed is not None else 'summary'
            log_path = CONTROL / f'{stage}_{key}.log'
            cmd = command(stage, seed)
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            with log_path.open('x') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            active.append((key, proc))
            state['jobs'][stage][key] = {'pid': proc.pid, 'seed': seed, 'gpu': gpu,
                                         'command': cmd, 'log': str(log_path),
                                         'status': 'running', 'started_utc': now()}
            save_status(state)
            print(f'START {stage} seed={seed} GPU={gpu} pid={proc.pid}', flush=True)
        while active:
            for key, proc in list(active):
                code = proc.poll()
                if code is None:
                    continue
                job = state['jobs'][stage][key]
                job.update(exit_code=code, status='succeeded' if code == 0 else 'failed', finished_utc=now())
                active.remove((key, proc))
                save_status(state)
                print(f'FINISH {stage} key={key} exit={code}', flush=True)
            if active:
                time.sleep(5)
    except BaseException:
        # Only terminate children created by this controller, never unrelated jobs.
        for _, proc in active:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in active:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    return all(job['exit_code'] == 0 for job in state['jobs'][stage].values())


def main(argv=None, expected_code_sha=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Verify frozen files and print commands; start no jobs')
    args = parser.parse_args(argv)
    check_frozen(expected_code_sha)
    reserved = [CONTROL, SUMMARY]
    reserved += [OUTPUTS / f'e3a_v150_s{seed}{suffix}' for seed, _ in SEEDS for suffix in ['', '_test']]
    if any(path.exists() for path in reserved):
        raise FileExistsError('A reserved output exists; never overwrite or silently resume')
    for stage in ['train', 'test', 'summarize']:
        for seed, gpu in SEEDS if stage != 'summarize' else [(None, None)]:
            print({'stage': stage, 'seed': seed, 'gpu': gpu, 'command': command(stage, seed)}, flush=True)
    if args.plan:
        return 0
    require_training_gpus_free()
    CONTROL.mkdir(parents=True)
    state = {'status': 'running', 'stage': 'starting', 'controller_pid': os.getpid(),
             'started_utc': now(), 'project': str(ROOT), 'code_sha256': expected_code_sha,
             'config_sha256': FORMAL_CONFIG_SHA256, 'data_sha256': DATA_MANIFEST_SHA256,
             'config': read(ROOT / 'configs/e3a.json'), 'jobs': {},
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
