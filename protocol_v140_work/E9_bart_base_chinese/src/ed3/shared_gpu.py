"""Explicit, checksum-bound E8 coexistence; default remains exclusive GPU use.

This is an operational allocation ceiling, not a model/numerical setting.
Only the E9 process receives the PyTorch allocator limit. No peer is signalled.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

_applied = None


def load_policy():
    path, expected = os.environ.get('ED3_SHARED_GPU_POLICY'), os.environ.get('ED3_SHARED_GPU_POLICY_SHA256')
    if not path and not expected:
        return None
    if not path or not expected or not Path(path).is_absolute():
        raise ValueError('Shared GPU policy requires an absolute path and SHA256')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Shared GPU policy checksum mismatch')
    value = json.loads(raw)
    fields = {'schema', 'gpu_uuid', 'boot_id', 'uid', 'peer_project', 'peer_code_sha256',
              'allocator_limit_mib', 'context_margin_mib', 'peer_reservation_mib', 'free_margin_mib',
              'minimum_ram_mib', 'minimum_disk_mib', 'max_peer_jobs'}
    if set(value) != fields or value['schema'] != 1:
        raise ValueError('Unexpected shared GPU policy schema')
    for key in fields - {'gpu_uuid', 'boot_id', 'peer_project', 'peer_code_sha256'}:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('Invalid shared GPU policy integer: ' + key)
    if (value['peer_project'] != '/home/zxk/oraccle-0905/E8_mengzi_t5_base'
            or not 1 <= value['max_peer_jobs'] <= 2
            or not 1024 <= value['allocator_limit_mib'] <= 8192
            or value['context_margin_mib'] < 1024 or value['free_margin_mib'] < 2048
            or value['peer_reservation_mib'] < 14500 * value['max_peer_jobs']
            or value['minimum_ram_mib'] < 8192 or value['minimum_disk_mib'] < 10240):
        raise ValueError('Shared GPU policy lacks conservative E8 reservations')
    return dict(value, policy_sha256=expected)


def query(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=15).stdout


def process_identity(pid):
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': int(pid), 'start_ticks': int(stat[19]), 'uid': root.stat().st_uid,
            'cwd': os.readlink(root / 'cwd'),
            'command': (root / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}


def verify_peer(identity, policy, state):
    if (identity['uid'] != policy['uid'] or identity['cwd'] != policy['peer_project']
            or state.get('project') != policy['peer_project']
            or state.get('code_sha256') != policy['peer_code_sha256']):
        raise RuntimeError('GPU process does not belong to the authorized E8 deployment')
    for stage in ('train', 'test'):
        for job in state.get('jobs', {}).get(stage, {}).values():
            recorded = job.get('process_identity') or job.get('adopted_process') or {}
            if job.get('pid') != identity['pid']:
                continue
            if (recorded.get('boot_id') == policy['boot_id']
                    and int(recorded.get('start_ticks', -1)) == identity['start_ticks']
                    and recorded.get('uid') == identity['uid']
                    and recorded.get('cwd') == identity['cwd']
                    and recorded.get('command') == identity['command']
                    and job.get('command') == identity['command']):
                return
    raise RuntimeError('E8 GPU PID identity is absent, changed or reused in current state')


def verify_memory(policy, total, free, own, peers):
    own_envelope = policy['allocator_limit_mib'] + policy['context_margin_mib']
    if own > own_envelope or free < policy['free_margin_mib']:
        raise RuntimeError('E9 exceeded its envelope or whole-device safety margin')
    if total < policy['peer_reservation_mib'] + own_envelope + policy['free_margin_mib']:
        raise RuntimeError('Configured E8 reservation plus E9 envelope exceeds total VRAM')
    if peers > policy['peer_reservation_mib']:
        raise RuntimeError('E8 already exceeds the reserved shared VRAM envelope')
    if free + own < own_envelope + policy['free_margin_mib']:
        raise RuntimeError('Insufficient current VRAM headroom for bounded E9 coexistence')


def check_shared_gpu(owned_pid=None):
    policy = load_policy()
    if policy is None:
        raise ValueError('Explicit shared GPU policy is required')
    if (Path('/proc/sys/kernel/random/boot_id').read_text().strip() != policy['boot_id']
            or os.getuid() != policy['uid'] or os.environ.get('CUDA_VISIBLE_DEVICES', '0') != '0'):
        raise RuntimeError('Shared GPU host/user/device identity changed')
    gpu = query(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.total,memory.free',
                 '--format=csv,noheader,nounits']).strip().split(',')
    if len(gpu) != 3 or gpu[0].strip() != policy['gpu_uuid']:
        raise RuntimeError('Shared GPU UUID mismatch')
    total, free = (int(x.strip()) for x in gpu[1:])
    processes = query(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,used_memory',
                       '--format=csv,noheader,nounits'])
    state = json.loads((Path(policy['peer_project']) / 'outputs/ed2a_formal_earlystop/status.json').read_text())
    peers, own, peer_memory = [], 0, 0
    for line in processes.splitlines():
        if not line.strip():
            continue
        pid, used = (int(x.strip()) for x in line.split(','))
        if pid == os.getpid() or pid == owned_pid:
            own += used
            continue
        peer = process_identity(pid)
        verify_peer(peer, policy, state)
        peers.append(peer)
        peer_memory += used
    if len(peers) > policy['max_peer_jobs']:
        raise RuntimeError('Too many authorized peers for the reserved envelope')
    verify_memory(policy, total, free, own, peer_memory)
    ram = next(int(line.split()[1]) // 1024 for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith('MemAvailable:'))
    disk = shutil.disk_usage(Path(__file__).resolve().parents[2]).free // 2**20
    if ram < policy['minimum_ram_mib'] or disk < policy['minimum_disk_mib']:
        raise RuntimeError('Insufficient RAM/disk for E9 coexistence')
    return dict(policy, total_mib=total, free_mib=free, peers=peers, peer_used_mib=peer_memory,
                own_process_mib=own, available_ram_mib=ram, free_disk_mib=disk)


def apply_shared_budget(device):
    global _applied
    policy = load_policy()
    if policy is None:
        return None
    import torch
    device = torch.device(device)
    if device.type != 'cuda' or device.index not in (None, 0):
        raise ValueError('Shared E9 budget only supports CUDA device 0')
    if _applied is not None:
        if _applied['policy_sha256'] != policy['policy_sha256']:
            raise RuntimeError('Cannot change resource policy in an active process')
        return dict(_applied)
    snapshot = check_shared_gpu()
    # This ceiling constrains only this process's PyTorch allocator, not peers.
    # The extra 1 GiB margin accounts for CUDA context/library allocations.
    fraction = policy['allocator_limit_mib'] * 2**20 / torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    _applied = dict(snapshot, allocator_fraction=fraction, allocator_limit_applied=True, pid=os.getpid())
    print('ED3_SHARED_GPU_BUDGET ' + json.dumps(_applied), flush=True)
    return dict(_applied)


def applied_budget():
    return dict(_applied) if _applied is not None else None


def wait_owned_job(proc):
    """Watch only a Popen child created by this caller; never signal peer jobs.

    Caller owns termination/reaping on error. Unknown/reused peers or exhausted
    headroom stop E9 dispatch rather than borrowing E8's reservation.
    """
    if load_policy() is None:
        return proc.wait()
    while True:
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                check_shared_gpu(owned_pid=proc.pid)
