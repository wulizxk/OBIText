"""Run the frozen three-seed E1-A protocol: train, final Test, summarize."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]
from e1.io import read, sha, source_fingerprint, verify_data, write

CODE_SHA = '086114d5666a506e4ca51bd9cf412e56bfdc61d271eb8e28886d1167251c3958'
CONFIG_SHA = '128ebb180eb7d331017050506abe233d510a2c22f7307c0d9d37444366f1ad7a'
DATA_SHA = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'
SEEDS = [(42, 0), (3407, 1), (2026, 2)]
OUTPUTS = ROOT / 'outputs'
CONTROL = OUTPUTS / 'e1a_v150_formal_earlystop'
PYTHON = str(ROOT / '.venv/bin/python')
SUMMARY = OUTPUTS / 'e1a_v150_test_summary.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def check_frozen():
    if source_fingerprint(ROOT) != CODE_SHA:
        raise ValueError('Source fingerprint differs from the verified early-stop version')
    if sha(ROOT / 'configs/e1a.json') != CONFIG_SHA:
        raise ValueError('Configuration differs from the authorized early-stop protocol')
    if verify_data(ROOT.parent / 'shared/data/v1.5.0') != DATA_SHA:
        raise ValueError('Frozen data identity differs')
    cfg = read(ROOT / 'configs/e1a.json')
    if (cfg.get('max_epochs'), cfg.get('early_stopping'), cfg.get('early_stopping_patience')) != (50, True, 5):
        raise ValueError('Expected max_epochs=50, early_stopping=true, patience=5')


def command(stage, seed):
    base = [PYTHON, '-B', '-u', 'run.py']
    if stage == 'train':
        return base + ['train', '--seed', str(seed), '--output', str(OUTPUTS / f'e1a_v150_s{seed}')]
    if stage == 'test':
        return base + ['test', '--checkpoint', str(OUTPUTS / f'e1a_v150_s{seed}/best.pt'),
                       '--output', str(OUTPUTS / f'e1a_v150_s{seed}_test'), '--confirm-final']
    return base + ['summarize', '--evaluations'] + [str(OUTPUTS / f'e1a_v150_s{s}_test') for s, _ in SEEDS] + ['--output', str(SUMMARY)]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Verify frozen files and print commands; start no jobs')
    args = parser.parse_args()
    check_frozen()
    reserved = [CONTROL, SUMMARY]
    reserved += [OUTPUTS / f'e1a_v150_s{seed}{suffix}' for seed, _ in SEEDS for suffix in ['', '_test']]
    if any(path.exists() for path in reserved):
        raise FileExistsError('A reserved output exists; never overwrite or silently resume')
    for stage in ['train', 'test', 'summarize']:
        for seed, gpu in SEEDS if stage != 'summarize' else [(None, None)]:
            print({'stage': stage, 'seed': seed, 'gpu': gpu, 'command': command(stage, seed)}, flush=True)
    if args.plan:
        return 0
    CONTROL.mkdir(parents=True)
    state = {'status': 'running', 'stage': 'starting', 'controller_pid': os.getpid(),
             'started_utc': now(), 'project': str(ROOT), 'code_sha256': CODE_SHA,
             'config_sha256': CONFIG_SHA, 'data_sha256': DATA_SHA,
             'config': read(ROOT / 'configs/e1a.json'), 'jobs': {},
             'policy': ('Each seed runs at most 50 epochs and stops after 5 consecutive full-Dev '
                        'non-improvements. All three valid completions precede any final Test.')}
    save_status(state)
    try:
        for stage in ['train', 'test', 'summarize']:
            check_frozen()
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


if __name__ == '__main__':
    sys.exit(main())
