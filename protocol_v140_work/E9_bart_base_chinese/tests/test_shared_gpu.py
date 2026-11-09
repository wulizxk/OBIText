"""CPU-only operational guard tests; never allocate a GPU or inspect live peers."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from ed3 import shared_gpu as shared


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.delenv('ED3_SHARED_GPU_POLICY', raising=False)
    monkeypatch.delenv('ED3_SHARED_GPU_POLICY_SHA256', raising=False)
    return {'schema': 1, 'gpu_uuid': 'GPU-test', 'boot_id': 'boot-test', 'uid': 1002,
            'peer_project': '/home/zxk/oraccle-0905/E8_mengzi_t5_base', 'peer_code_sha256': 'b'*64,
            'allocator_limit_mib': 4096, 'context_margin_mib': 1024, 'peer_reservation_mib': 29000,
            'free_margin_mib': 2048, 'minimum_ram_mib': 8192, 'minimum_disk_mib': 10240, 'max_peer_jobs': 2}


def install(tmp_path, monkeypatch, policy):
    path=tmp_path/'policy.json';raw=json.dumps(policy).encode();path.write_bytes(raw)
    monkeypatch.setenv('ED3_SHARED_GPU_POLICY', str(path))
    monkeypatch.setenv('ED3_SHARED_GPU_POLICY_SHA256', hashlib.sha256(raw).hexdigest())
    return path


def test_default_exclusive_and_checksum_bound_policy(policy, tmp_path, monkeypatch):
    assert shared.load_policy() is None
    path=install(tmp_path,monkeypatch,policy)
    assert shared.load_policy()['allocator_limit_mib']==4096
    path.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):shared.load_policy()


@pytest.mark.parametrize('key,value',[('allocator_limit_mib',10000),('context_margin_mib',0),
    ('peer_reservation_mib',0),('free_margin_mib',0),('minimum_ram_mib',0),('minimum_disk_mib',0),
    ('max_peer_jobs',3),('max_peer_jobs',True),('peer_project','/tmp/other')])
def test_reject_underreserved_or_broad_policy(policy,tmp_path,monkeypatch,key,value):
    policy[key]=value;install(tmp_path,monkeypatch,policy)
    with pytest.raises(ValueError):shared.load_policy()


@pytest.mark.parametrize('total,free,own,peers,valid',[
    (40000,8000,0,22000,True),(40000,5000,3000,26000,True),
    (32607,8000,0,22000,False),(40000,5000,0,22000,False),
    (40000,8000,0,29100,False),(40000,8000,5500,22000,False),(40000,100,5000,22000,False)])
def test_memory_envelope(policy,total,free,own,peers,valid):
    if valid:shared.verify_memory(policy,total,free,own,peers)
    else:
        with pytest.raises(RuntimeError):shared.verify_memory(policy,total,free,own,peers)


@pytest.mark.parametrize('change',[None,'pid','start_ticks','uid','cwd','command','boot_id','code'])
def test_current_peer_identity_not_just_pid(policy,change):
    ident={'pid':123,'start_ticks':456,'uid':1002,'cwd':policy['peer_project'],
           'command':['python','run.py','train','--seed','42']}
    rec=dict(ident,boot_id=policy['boot_id'])
    job={'pid':123,'command':list(ident['command']),'process_identity':rec}
    state={'project':policy['peer_project'],'code_sha256':policy['peer_code_sha256'],
           'jobs':{'train':{'42':job}}}
    if change in ('pid','start_ticks','uid'):ident[change]+=1
    if change in ('cwd','command'):ident[change]='bad'
    if change=='boot_id':rec['boot_id']='other-boot'
    if change=='code':state['code_sha256']='different'
    if change is None:shared.verify_peer(ident,policy,state)
    else:
        with pytest.raises(RuntimeError):shared.verify_peer(ident,policy,state)


def test_wait_only_checks_callers_child_after_timeout(policy,monkeypatch):
    import subprocess
    from types import SimpleNamespace
    monkeypatch.setattr(shared,'load_policy',lambda:policy)
    calls=[]
    def wait(timeout=None):
        calls.append(timeout)
        if len(calls)==1:raise subprocess.TimeoutExpired('owned-child',timeout)
        return 0
    checks=[]
    monkeypatch.setattr(shared,'check_shared_gpu',lambda owned_pid=None:checks.append(owned_pid))
    child=SimpleNamespace(pid=123,wait=wait,poll=lambda:None)
    assert shared.wait_owned_job(child)==0
    assert checks==[123] and calls==[10,10]


def test_real_card_cannot_fit_two_e8_reservations_plus_e9(policy):
    # 2*14500 E8 + 4096 E9 + 1024 CUDA margin + 2048 safety > 32607 MiB.
    with pytest.raises(RuntimeError,match='exceeds total VRAM'):
        shared.verify_memory(policy,32607,20000,0,12000)
