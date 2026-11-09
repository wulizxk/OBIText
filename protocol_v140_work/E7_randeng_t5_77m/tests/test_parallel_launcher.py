"""Launch-only scheduling tests. No formal runs, CUDA work or Test reads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/run_parallel_earlystop.py'
spec = importlib.util.spec_from_file_location('parallel_launcher', PATH)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def snapshot(free=31751, processes=None, host=50000):
    return {'free_gpu_mib': free, 'processes': processes or {}, 'available_host_mib': host}


@pytest.mark.parametrize('free,processes,pids,host,expected', [
    (31751, {}, [], 50000, True),
    (29000, {101: 2800}, [101], 47000, True),
    (26100, {101: 2800, 102: 2900}, [101, 102], 43000, True),
    (4000, {}, [], 50000, False),
    (18000, {}, [101, 102], 50000, False),  # Loading children still reserve future VRAM.
    (31000, {999: 1}, [], 50000, False),
    (31000, {}, [], 4000, False),
    (31000, {}, [1, 2, 3], 50000, False),
])
def test_admission_reserves_resources_and_excludes_foreign_jobs(free, processes, pids, host, expected):
    allowed, _ = launcher.admission(snapshot(free, processes, host), pids)
    assert allowed is expected


def test_gpu_process_parser_rejects_unknown_memory():
    assert launcher.parse_gpu_processes('101, 2500\n102, 3400\n') == {101: 2500, 102: 3400}
    assert launcher.parse_gpu_processes('') == {}
    with pytest.raises(ValueError):
        launcher.parse_gpu_processes('101, [N/A]')


def test_startup_waits_for_real_training_progress(tmp_path):
    log = tmp_path / 'train.log'
    job = {'log': str(log)}
    assert not launcher.ready_for_overlap(job)
    log.write_text('loading model\n')
    assert not launcher.ready_for_overlap(job)
    log.write_text('TRAIN epoch=1 samples=6400/66511 loss=1.0\n')
    assert launcher.ready_for_overlap(job)


def test_default_is_plan_without_mutation(monkeypatch, capsys):
    monkeypatch.setattr(launcher, 'read', lambda _: {'code_sha256': 'code'})
    monkeypatch.setattr(launcher, 'sha', lambda _: 'driver')
    monkeypatch.setattr(launcher, 'verify_driver_and_core', lambda *_: None)
    monkeypatch.setattr(launcher, 'resource_snapshot', lambda: pytest.fail('No GPU probe for plan'))
    monkeypatch.setattr(launcher, 'train_phase', lambda *_: pytest.fail('No train for plan'))
    assert launcher.main([]) == 0
    assert 'plan_only' in capsys.readouterr().out


@pytest.mark.parametrize('fail_seed', [None, 42])
def test_parallel_phase_completion_and_failure_no_new_starts(monkeypatch, fail_seed):
    clock = [0]
    launched, validated = [], []
    state = {'jobs': {}}
    monkeypatch.setattr(launcher.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(launcher.time, 'sleep', lambda t: clock.__setitem__(0, clock[0]+t))
    monkeypatch.setattr(launcher.frozen, 'save_status', lambda _: None)
    monkeypatch.setattr(launcher, 'verify_driver_and_core', lambda *_: None)
    monkeypatch.setattr(launcher, 'resource_snapshot', lambda: snapshot())
    monkeypatch.setattr(launcher, 'ready_for_overlap', lambda _: True)
    monkeypatch.setattr(launcher, 'validate_seed', lambda s: (validated.append(s) or {'ok': True}))

    def start(seed, state):
        launched.append((seed, clock[0]))
        finish_at = clock[0] + (60 if fail_seed == seed else 150)
        state['jobs']['train'][str(seed)] = {'pid': seed, 'log': 'fixture'}
        return SimpleNamespace(pid=seed, poll=lambda: None if clock[0] < finish_at else (1 if seed == fail_seed else 0))

    monkeypatch.setattr(launcher, 'launch_seed', start)
    result = launcher.train_phase(state, 'code', 'driver')
    if fail_seed:
        assert result is False
        assert [s for s, _ in launched] == [42, 3407]
        assert validated == [3407]  # Healthy child drained, never aborted/retried.
        assert state['jobs']['train']['42']['status'] == 'failed'
        assert state['pending_seeds'] == [2026]
    else:
        assert result is True
        assert [s for s, _ in launched] == [42, 3407, 2026]
        assert len(validated) == 3
        assert max(t for _, t in launched) < 150  # Actual overlap in the simulation.


def test_training_failure_blocks_test_and_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'sha', lambda _: 'digest')
    monkeypatch.setattr(launcher, 'read', lambda _: {'code_sha256': 'digest', 'config_sha256': 'digest',
                                                  'data_manifest_sha256': 'digest', 'status': 'PASS',
                                                  'launcher_sha256': 'digest', 'tests_sha256': 'digest'})
    monkeypatch.setattr(launcher, 'verify_driver_and_core', lambda *_: None)
    monkeypatch.setattr(launcher.frozen, 'CONTROL', tmp_path / 'control')
    monkeypatch.setattr(launcher.frozen, 'OUTPUTS', tmp_path / 'outputs')
    monkeypatch.setattr(launcher.frozen, 'SUMMARY', tmp_path / 'summary.json')
    monkeypatch.setattr(launcher.frozen, 'require_gpu_smoke', lambda *_: None)
    monkeypatch.setattr(launcher.frozen, 'require_training_gpus_free', lambda: None)
    monkeypatch.setattr(launcher.frozen, 'save_status', lambda _: None)
    monkeypatch.setattr(launcher, 'train_phase', lambda *_: False)
    monkeypatch.setattr(launcher.frozen, 'phase', lambda *_: pytest.fail('Test started after failed training'))
    assert launcher.main(['--execute']) == 1
