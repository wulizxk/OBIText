"""Resource-admitted parallel seeds; unchanged frozen ED1-A train/Test code.

Default is plan-only. This launch-only wrapper has a separately verified SHA;
it does not change the research-code fingerprint or any model hyperparameter.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from ed1 import formal_controller as frozen
from ed1.io import read, sha

MAX_WORKERS = 3
JOB_GPU_BUDGET_MIB = 8192
GPU_HEADROOM_MIB = 4096
JOB_HOST_BUDGET_MIB = 8192
HOST_HEADROOM_MIB = 8192
STAGGER_SECONDS = 45
POLL_SECONDS = 10
DRIVER = Path(__file__).resolve()
EVIDENCE = ROOT / 'verification/parallel_launcher_verified.json'


def parse_gpu_processes(output):
    result = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, used = [part.strip() for part in line.split(',')]
        result[int(pid)] = int(used)  # Reject N/A rather than assume zero.
    return result


def resource_snapshot():
    def query(args):
        return subprocess.run(['nvidia-smi', '-i', '0'] + args, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, timeout=15).stdout.strip()
    free_gpu = int(query(['--query-gpu=memory.free', '--format=csv,noheader,nounits']))
    processes = parse_gpu_processes(query(['--query-compute-apps=pid,used_gpu_memory',
                                           '--format=csv,noheader,nounits']))
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    return {'free_gpu_mib': free_gpu, 'processes': processes,
            'available_host_mib': mem['MemAvailable'] // 1024}


def admission(snapshot, active_pids):
    """Reserve the unallocated portion of every admitted child's GPU budget."""
    if len(active_pids) >= MAX_WORKERS:
        return False, 'max_parallel_workers'
    foreign = set(snapshot['processes']) - set(active_pids)
    if foreign:
        return False, 'external_gpu_processes:' + ','.join(map(str, sorted(foreign)))
    future_growth = sum(max(0, JOB_GPU_BUDGET_MIB - snapshot['processes'].get(pid, 0))
                        for pid in active_pids)
    need_gpu = GPU_HEADROOM_MIB + JOB_GPU_BUDGET_MIB + future_growth
    if snapshot['free_gpu_mib'] < need_gpu:
        return False, f'gpu_reserve_requires_{need_gpu}_MiB'
    # Deliberately conservative: account for all active jobs' future host growth.
    need_host = HOST_HEADROOM_MIB + JOB_HOST_BUDGET_MIB * (len(active_pids) + 1)
    if snapshot['available_host_mib'] < need_host:
        return False, f'host_reserve_requires_{need_host}_MiB'
    return True, 'resources_available'


def ready_for_overlap(job):
    """Do not overlap a still-loading child: require its first progress record."""
    path = Path(job['log'])
    if not path.exists():
        return False
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        recent = stream.read().decode('utf-8', errors='replace')
    return 'TRAIN epoch=' in recent or 'EPOCH_DONE epoch=' in recent


def verify_driver_and_core(expected_code, expected_driver):
    if sha(DRIVER) != expected_driver:
        raise ValueError('Parallel launcher changed after verification')
    frozen.check_frozen(expected_code)


def validate_seed(seed):
    checkpoint, marker = frozen.require_completed_run(frozen.OUTPUTS / f'ed1a_v150_s{seed}/best.pt')
    if checkpoint['identity']['seed'] != seed:
        raise ValueError('Completed seed does not match its requested output directory')
    del checkpoint
    return {'epochs_ran': marker['epochs_ran'], 'best_epoch': marker['best']['epoch'],
            'stopping': marker['stopping']}


def launch_seed(seed, state):
    command = frozen.command('train', seed)
    log_path = frozen.CONTROL / f'train_{seed}.log'
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
    with log_path.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
    state['jobs']['train'][str(seed)] = {
        'pid': process.pid, 'seed': seed, 'gpu': 0, 'command': command,
        'log': str(log_path), 'status': 'running', 'started_utc': frozen.now()}
    frozen.save_status(state)
    print(f'START train seed={seed} GPU=0 pid={process.pid}', flush=True)
    return process


def train_phase(state, expected_code, expected_driver):
    state['stage'] = 'train'
    state['jobs']['train'] = {}
    pending = [seed for seed, _ in frozen.SEEDS]
    active = {}
    failed = False
    last_launch = float('-inf')
    try:
        while active or (pending and not failed):
            for seed, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                job = state['jobs']['train'][str(seed)]
                job.update(exit_code=code, finished_utc=frozen.now())
                try:
                    if code:
                        raise RuntimeError(f'Training exited with code {code}')
                    job['completion'] = validate_seed(seed)
                    job['status'] = 'succeeded'
                except Exception as exc:
                    failed = True
                    job.update(status='failed', error=repr(exc))
                del active[seed]
                frozen.save_status(state)
            state['pending_seeds'] = pending[:]
            state['active_seeds'] = list(active)
            if failed:
                state['admission_status'] = 'failure: no new starts; draining healthy active seeds'
            elif pending and len(active) < MAX_WORKERS:
                ready = (not active or (time.monotonic() - last_launch >= STAGGER_SECONDS
                         and all(ready_for_overlap(state['jobs']['train'][str(s)]) for s in active)))
                if ready:
                    try:
                        snapshot = resource_snapshot()
                        allowed, reason = admission(snapshot, [p.pid for p in active.values()])
                        state['resources'] = snapshot
                        state['admission_status'] = reason
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        allowed = False
                        state['admission_status'] = 'resource_probe_failed:' + repr(exc)
                    if allowed:
                        verify_driver_and_core(expected_code, expected_driver)
                        seed = pending.pop(0)
                        active[seed] = launch_seed(seed, state)
                        last_launch = time.monotonic()
                else:
                    state['admission_status'] = 'waiting_for_stagger_and_first_train_progress'
            elif not failed:
                state['admission_status'] = 'all_seeds_admitted'
            state['pending_seeds'], state['active_seeds'] = pending[:], list(active)
            frozen.save_status(state)
            if active or (pending and not failed):
                time.sleep(POLL_SECONDS)
        return not failed and not pending
    except BaseException:
        # Only this controller's still-live children; never other GPU processes.
        for process in active.values():
            if process.poll() is None:
                process.terminate()
        for process in active.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    freeze = read(ROOT / 'verification/source_freeze_v150.json')
    code_sha, driver_sha = freeze['code_sha256'], sha(DRIVER)
    verify_driver_and_core(code_sha, driver_sha)
    if not args.execute:
        print(json.dumps({'mode': 'plan_only', 'seeds': [s for s, _ in frozen.SEEDS],
                          'max_parallel_train': MAX_WORKERS, 'gpu': 0,
                          'gpu_budget_per_job_mib': JOB_GPU_BUDGET_MIB,
                          'gpu_headroom_mib': GPU_HEADROOM_MIB,
                          'stagger_seconds': STAGGER_SECONDS,
                          'test_policy': 'sequential Test only after all 3 valid training completions',
                          'code_sha256': code_sha, 'launcher_sha256': driver_sha}, indent=2))
        return 0
    evidence = read(EVIDENCE)
    if (evidence.get('status') != 'PASS' or evidence.get('launcher_sha256') != driver_sha
            or evidence.get('code_sha256') != code_sha
            or evidence.get('tests_sha256') != sha(ROOT / 'tests/test_parallel_launcher.py')):
        raise ValueError('Current parallel launcher lacks matching test evidence')
    reserved = [frozen.CONTROL, frozen.SUMMARY]
    reserved += [frozen.OUTPUTS / f'ed1a_v150_s{seed}{suffix}' for seed, _ in frozen.SEEDS for suffix in ('', '_test')]
    if any(p.exists() for p in reserved):
        raise FileExistsError('Formal output exists; never overwrite or silently resume')
    frozen.require_gpu_smoke(code_sha)
    frozen.require_training_gpus_free()
    frozen.CONTROL.mkdir(parents=True)  # Shared exclusive namespace with serial entry.
    state = {'status': 'running', 'stage': 'starting', 'controller_pid': os.getpid(),
             'started_utc': frozen.now(), 'project': str(ROOT), 'jobs': {},
             'code_sha256': code_sha, 'launcher_sha256': driver_sha,
             'config_sha256': freeze['config_sha256'], 'data_sha256': freeze['data_manifest_sha256'],
             'config': read(ROOT / 'configs/ed1a.json'),
             'orchestration': 'parallel_train_then_sequential_test',
             'policy': {'max_parallel_train': MAX_WORKERS, 'gpu': 0,
                        'per_job_gpu_budget_mib': JOB_GPU_BUDGET_MIB,
                        'gpu_headroom_mib': GPU_HEADROOM_MIB, 'host_headroom_mib': HOST_HEADROOM_MIB,
                        'stagger_seconds': STAGGER_SECONDS,
                        'on_failure': 'no new starts; healthy children finish; no Test; no automatic retry'}}
    frozen.save_status(state)
    try:
        if not train_phase(state, code_sha, driver_sha):
            state.update(status='failed', failed_stage='train', finished_utc=frozen.now())
            frozen.save_status(state)
            return 1
        for stage in ('test', 'summarize'):
            verify_driver_and_core(code_sha, driver_sha)
            if not frozen.phase(stage, state):
                state.update(status='failed', failed_stage=stage, finished_utc=frozen.now())
                frozen.save_status(state)
                return 1
        if not read(frozen.SUMMARY)['formal_three_seed_complete']:
            raise ValueError('Three-seed Test summary is incomplete')
        state.update(status='complete', stage='complete', summary=str(frozen.SUMMARY), finished_utc=frozen.now())
        frozen.save_status(state)
        print('FORMAL_TRAIN_TEST_SUMMARY_COMPLETE', flush=True)
        return 0
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_utc=frozen.now())
        frozen.save_status(state)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
