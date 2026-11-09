"""BF16 protocol and fail-closed numerical guards, independent of Test metrics."""
import copy
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from ed3 import engine, formal_controller as controller, model, runtime
from ed3.io import read, sha, write
from ed3.protocol import FORMAL_CONFIG_SHA256, validate_formal_config
from test_training_loop import MiniCodec


def test_frozen_config_is_bf16_with_matching_digest(cfg):
    assert cfg['precision'] == 'bf16'
    validate_formal_config(cfg)
    assert sha(Path(__file__).resolve().parents[1] / 'configs/ed3a.json') == FORMAL_CONFIG_SHA256


@pytest.mark.parametrize('precision', ['fp16', 'fp32', 'auto'])
def test_formal_protocol_rejects_old_or_implicit_precision(cfg, precision):
    with pytest.raises(ValueError, match='precision'):
        validate_formal_config(dict(cfg, precision=precision))


@pytest.mark.parametrize('device,expected', [('cpu', 'fp32'), ('cuda:0', 'bf16')])
def test_effective_precision_has_no_cuda_fallback(cfg, device, expected):
    assert runtime.effective_precision(device, cfg) == expected
    assert not model.scaler_for(device, cfg).is_enabled()
    assert model.scaler_for(device, cfg).state_dict() == {}


def test_unsupported_precision_rejected(cfg):
    with pytest.raises(ValueError, match='Unsupported precision'):
        model.autocast(torch.device('cpu'), dict(cfg, precision='auto'))


@pytest.mark.parametrize('supported', [False, True])
def test_native_bf16_checks_selected_device_without_emulation(cfg, monkeypatch, supported):
    requested = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: (requested.append(str(device)) or nullcontext()))

    def support(*, including_emulation):
        assert including_emulation is False
        return supported

    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', support)
    if supported:
        assert runtime.require_cuda_precision('cuda:1', cfg) == 'bf16'
    else:
        with pytest.raises(RuntimeError, match='Native CUDA BF16'):
            runtime.require_cuda_precision('cuda:1', cfg)
    assert requested == ['cuda:1']


def test_autocast_selects_bf16_not_float16(cfg, monkeypatch):
    monkeypatch.setattr(model, 'require_cuda_precision', lambda *_: 'bf16')
    observed = []
    monkeypatch.setattr(torch, 'autocast', lambda **kwargs: (observed.append(kwargs) or nullcontext()))
    with model.autocast(torch.device('cuda:0'), cfg):
        pass
    assert observed == [{'device_type': 'cuda', 'dtype': torch.bfloat16, 'enabled': True}]


def test_forward_nan_stops_before_backward_or_optimizer(tiny, cfg, monkeypatch):
    optimizer = engine.optimizer_for(tiny, cfg)
    scaler = model.scaler_for('cpu', cfg)
    saved = copy.deepcopy(tiny.state_dict())
    monkeypatch.setattr(engine, 'per_sample_losses', lambda *_: torch.tensor([float('nan')], requires_grad=True))
    monkeypatch.setattr(scaler, 'scale', lambda *_: pytest.fail('Non-finite loss reached backward'))
    with pytest.raises(FloatingPointError, match='before backward'):
        engine.update(tiny, MiniCodec(None, cfg), [{'gold': 5}], cfg,
                      torch.device('cpu'), optimizer, scaler)
    assert not optimizer.state
    assert all(p.grad is None for p in tiny.parameters())
    for key, value in tiny.state_dict().items():
        assert torch.equal(value, saved[key])


def test_unscaled_bf16_policy_aborts_bad_gradients_without_retry(tiny, cfg):
    optimizer = engine.optimizer_for(tiny, cfg)
    saved = copy.deepcopy(tiny.state_dict())
    calls = []

    def infinite_gradient(grad):
        calls.append(True)
        return grad * float('inf')

    handle = tiny.get_input_embeddings().weight.register_hook(infinite_gradient)
    try:
        with pytest.raises(FloatingPointError, match='Non-finite unscaled gradients'):
            engine.update(tiny, MiniCodec(None, cfg), [{'gold': 5}], cfg,
                          torch.device('cpu'), optimizer, model.scaler_for('cpu', cfg))
    finally:
        handle.remove()
    assert calls == [True] and not optimizer.state
    assert all(p.grad is None for p in tiny.parameters())
    for key, value in tiny.state_dict().items():
        assert torch.equal(value, saved[key])


def smoke_fixture(tmp_path, cfg, monkeypatch):
    from test_e9_review_fixes import launch_extensions
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'verify_weights', lambda *_: 'weights')
    versions = runtime.runtime_versions()
    policy = runtime.configure_execution()
    env = {'versions': versions, 'execution_policy': policy, 'effective_precision': 'bf16',
           'hardware': {'type': 'cuda', 'name': 'NVIDIA GeForce RTX 5090', 'capability': [12, 0]}}
    ident = {'cfg': cfg, 'code_sha256': 'a' * 64, 'data_sha256': controller.DATA_MANIFEST_SHA256,
             'weights_sha256': 'weights', 'runtime': versions, 'execution_policy': policy}
    report = {'kind': 'smoke_only_not_paper_metrics', 'status': 'PASS', 'optimizer_updates': 2,
              'train_samples_per_update': 64, 'losses': [30.0, 29.0], 'gradient_norms': [1.0, 2.0],
              'new_target_embedding_updated': True, 'new_tied_output_row_updated': True,
              'checkpoint_predictions_identical': True, 'sampler_restore_identical': True,
              'grad_scaler_enabled': False, 'parameter_dtype': 'torch.float32', 'scheduler_steps': 2,
              'max_shape_forward_backward_passed': True,
              'max_shape_batch_size': cfg['micro_batch_size'],
              'max_shape_input_length': cfg['max_input_length'], 'max_shape_gap_length': 5,
              'max_shape_gradient_norm': 1.0, 'max_shape_optimizer_updates': 0,
              'test_answers_used': False, 'gap_lengths_checked': [1, 2, 3, 4, 5],
              'boundary_sides_checked': ['initial', 'final'], 'identity': ident, 'evaluation_runtime': env}
    report['cuda_graph_eager_predictions_identical'] = True
    report.update(launch_extensions(cfg))
    for key in ('evaluation_execution_before_reload', 'evaluation_execution_after_reload'):
        report[key].update({'backend': cfg['evaluation_backend'], 'enabled': False,
                       'captures': 0, 'replays': 0, 'capture_bitwise_checks_passed': 0}
                          )
    write(tmp_path / 'configs/ed3a.json', cfg)
    monkeypatch.setattr(controller, 'require_cuda_precision', lambda *_: 'bf16')
    monkeypatch.setattr(controller, 'evaluation_environment', lambda *_: copy.deepcopy(env))
    monkeypatch.setattr(controller, 'current_gpu_environment', lambda *_: copy.deepcopy(env))
    monkeypatch.setattr(controller, 'require_training_gpus_free', lambda: None)
    # This fixture isolates report/runtime checks. Physical evidence replay is
    # independently tested by test_smoke_artifact_replay_*.
    monkeypatch.setattr(controller, 'require_smoke_artifacts', lambda *_: None)
    return report


def test_smoke_gate_accepts_current_bf16_evidence(cfg, tmp_path, monkeypatch):
    report = smoke_fixture(tmp_path, cfg, monkeypatch)
    write(tmp_path / 'verification/gpu_smoke_v150/smoke_report.json', report)
    controller.require_gpu_smoke('a' * 64)


@pytest.mark.parametrize('change', ['old_fp16', 'scaler', 'master_dtype', 'scheduler', 'code',
                                    'runtime', 'losses', 'batch', 'hardware', 'missing_stress',
                                    'stress_batch', 'stress_length', 'stress_gap', 'stress_nan',
                                    'stress_update', 'missing_graph', 'graph_eager_mismatch',
                                    'graph_disabled', 'graph_no_capture', 'graph_no_replay',
                                    'graph_capture_mismatch', 'graph_old_backend'])
def test_smoke_gate_rejects_stale_or_incomplete_evidence(cfg, tmp_path, monkeypatch, change):
    report = copy.deepcopy(smoke_fixture(tmp_path, cfg, monkeypatch))
    if change == 'old_fp16':
        report['evaluation_runtime']['effective_precision'] = 'fp16'
    elif change == 'scaler':
        report['grad_scaler_enabled'] = True
    elif change == 'master_dtype':
        report['parameter_dtype'] = 'torch.bfloat16'
    elif change == 'scheduler':
        report['scheduler_steps'] = 0
    elif change == 'code':
        report['identity']['code_sha256'] = 'old'
    elif change == 'runtime':
        report['identity']['runtime']['torch'] = 'old'
    elif change == 'losses':
        report['losses'] = []
    elif change == 'batch':
        report['train_samples_per_update'] = 2
    elif change == 'hardware':
        report['evaluation_runtime']['hardware']['name'] = 'different GPU'
    elif change == 'missing_stress':
        report.pop('max_shape_forward_backward_passed')
    elif change == 'stress_batch':
        report['max_shape_batch_size'] = 8
    elif change == 'stress_length':
        report['max_shape_input_length'] = 80
    elif change == 'stress_gap':
        report['max_shape_gap_length'] = 1
    elif change == 'stress_nan':
        report['max_shape_gradient_norm'] = 'not-a-finite-number'
    elif change == 'stress_update':
        report['max_shape_optimizer_updates'] = 1
    elif change == 'missing_graph':
        report.pop('evaluation_execution_after_reload')
    elif change == 'graph_eager_mismatch':
        report['cuda_graph_eager_predictions_identical'] = False
    elif change == 'graph_disabled':
        report['evaluation_execution_before_reload']['enabled'] = True
    elif change == 'graph_no_capture':
        report['evaluation_execution_before_reload']['captures'] = 1
    elif change == 'graph_no_replay':
        report['evaluation_execution_after_reload']['replays'] = 1
    elif change == 'graph_capture_mismatch':
        report['evaluation_execution_after_reload']['capture_bitwise_checks_passed'] = 1
    elif change == 'graph_old_backend':
        report['evaluation_execution_before_reload']['backend'] = 'cuda_graph_exact_v1'
    write(tmp_path / 'verification/gpu_smoke_v150/smoke_report.json', report)
    with pytest.raises(ValueError):
        controller.require_gpu_smoke('a' * 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA BF16 real compute regression')
def test_cuda_bf16_update_and_disabled_scaler_restore(tiny, cfg):
    device = torch.device('cuda:0')
    runtime.require_cuda_precision(device, cfg)
    runtime.configure_execution(42)
    tiny.to(device)
    cfg = dict(cfg, micro_batch_size=2)
    scaler = model.scaler_for(device, cfg)
    optimizer = engine.optimizer_for(tiny, cfg)
    loss, norm = engine.update(tiny, MiniCodec(None, cfg), [{'gold': 5}, {'gold': 6}], cfg,
                               device, optimizer, scaler)
    assert loss > 0 and norm > 0 and not scaler.is_enabled()
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in tiny.parameters())
    restored = model.scaler_for(device, cfg)
    restored.load_state_dict(scaler.state_dict())
    assert not restored.is_enabled() and restored.state_dict() == {}
    with model.autocast(device, cfg):
        result = torch.nn.functional.linear(torch.ones(2, 16, device=device),
                                            tiny.get_input_embeddings().weight)
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
