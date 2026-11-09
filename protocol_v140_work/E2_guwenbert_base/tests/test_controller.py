from pathlib import Path
from types import SimpleNamespace

import pytest

from e2 import formal_controller as controller
from e2.protocol import FORMAL_CONFIG


def configure_temp_paths(monkeypatch, tmp_path):
    outputs = tmp_path / 'outputs'
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'OUTPUTS', outputs)
    monkeypatch.setattr(controller, 'CONTROL', outputs / 'e2a_formal_earlystop')
    monkeypatch.setattr(controller, 'SUMMARY', outputs / 'e2a_test_summary.json')
    monkeypatch.setattr(controller, 'PYTHON', str(tmp_path / '.venv/bin/python'))
    return outputs


def test_plan_is_non_mutating_and_skips_gpu_probe(monkeypatch, tmp_path, capsys):
    outputs = configure_temp_paths(monkeypatch, tmp_path)
    checks = []
    monkeypatch.setattr(controller, 'check_frozen', lambda expected: checks.append(expected))
    monkeypatch.setattr(controller, 'require_training_gpus_free',
                        lambda: pytest.fail('--plan must not query or reserve GPUs'))
    assert controller.main(['--plan'], expected_code_sha='a' * 64) == 0
    assert checks == ['a' * 64]
    assert not outputs.exists()
    printed = capsys.readouterr().out
    assert printed.count("'stage': 'train'") == 3
    assert printed.count("'stage': 'test'") == 3
    assert printed.count("'stage': 'summarize'") == 1


def test_gpu_guard_rejects_any_existing_compute_process(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        gpu = int(command[command.index('-i') + 1])
        calls.append((gpu, kwargs))
        stdout = '8745, E1_sikuroberta/.venv/bin/python\n' if gpu == 0 else ''
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(controller.subprocess, 'run', fake_run)
    with pytest.raises(RuntimeError, match='Refusing to overlap'):
        controller.require_training_gpus_free()
    assert [gpu for gpu, _ in calls] == [0, 1, 2]


def test_failed_training_stage_prevents_test_and_summary(monkeypatch, tmp_path):
    configure_temp_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, 'check_frozen', lambda _: None)
    monkeypatch.setattr(controller, 'require_training_gpus_free', lambda: None)
    monkeypatch.setattr(controller, 'read', lambda _: FORMAL_CONFIG)
    stages = []

    def fail_training(stage, state):
        stages.append(stage)
        return False

    monkeypatch.setattr(controller, 'phase', fail_training)
    assert controller.main([], expected_code_sha='a' * 64) == 1
    assert stages == ['train']


def test_controller_core_is_inside_source_fingerprint_tree():
    core = Path(controller.__file__).resolve()
    assert core.name == 'formal_controller.py'
    assert core in set((controller.ROOT / 'src').rglob('*.py'))
