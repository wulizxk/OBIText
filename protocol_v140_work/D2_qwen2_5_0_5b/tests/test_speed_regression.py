"""Speed-only microbatch change preserves sampling/objective/update semantics."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from d2 import engine
from d2.data import Codec, training_batch
from d2.io import sha
from d2.protocol import FORMAL_CONFIG_SHA256, validate_formal_config


class VariableCodec:
    candidate_ids = [5, 6, 7, 8]

    def collate(self, questions, device):
        encoded = []
        for index in questions:
            prompt = [2] + [3] * (index % 11) + [4]
            answer = [5 + index % 3] * (index % 5 + 1) + [8]
            encoded.append({'input_ids': prompt + answer,
                            'labels': [-100] * len(prompt) + answer})
        return training_batch(encoded, {5: 0, 6: 1, 7: 2, 8: 3}, 0, 8, device)


def test_speed_configuration_is_locked(cfg):
    validate_formal_config(cfg)
    assert cfg['micro_batch_size'] == 32
    assert cfg['effective_batch_size'] == 64
    assert cfg['micro_batch_token_budget'] == 2048
    assert cfg['gradient_checkpointing'] is True
    assert cfg['precision'] == 'fp16'
    assert cfg['max_epochs'] == 50 and cfg['early_stopping_patience'] == 5
    assert sha(Path(__file__).resolve().parents[1] / 'configs/d2a.json') == FORMAL_CONFIG_SHA256
    with pytest.raises(ValueError, match='protocol change'):
        validate_formal_config(dict(cfg, micro_batch_size=4))


@pytest.mark.parametrize('width,expected', [(31, 32), (64, 32), (65, 16), (128, 16)])
def test_length_guard_is_deterministic_and_padding_aware(cfg, width, expected):
    codec = SimpleNamespace(encode=lambda question, supervised: {'input_ids': [1] * question})
    values = [10, width, 9, 13]
    before = list(values)
    actual = Codec.training_micro_batch_size(codec, values, cfg)
    assert actual == expected and values == before
    padded = ((width + 7) // 8) * 8
    assert actual * padded <= cfg['micro_batch_token_budget']


@pytest.mark.parametrize('count,micro', [(13, 32), (64, 32), (64, 16)])
def test_micro32_matches_micro4_objective_gradients_and_update(tiny, cfg, count, micro):
    # Unequal prompt/answer lengths exercise both padding and per-sample loss;
    # 13 checks the final partial effective batch, not a hard-coded /64 loss.
    models = [copy.deepcopy(tiny), copy.deepcopy(tiny)]
    records = []
    for model, batch_size in zip(models, (4, micro)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        local_cfg = dict(cfg, micro_batch_size=batch_size, precision='fp32')
        optimizer = engine.optimizer_for(model, local_cfg)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        loss, norm = engine.update(model, VariableCodec(), list(range(count)), local_cfg,
                                   torch.device('cpu'), optimizer, scaler)
        records.append((loss, norm, optimizer))
    assert records[0][0] == pytest.approx(records[1][0], rel=2e-6)
    assert records[0][1] == pytest.approx(records[1][1], rel=2e-6)
    for first, second in zip(*[model.parameters() for model in models]):
        torch.testing.assert_close(first.grad, second.grad, rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(first, second, rtol=2e-4, atol=2e-6)
    for _, _, optimizer in records:
        assert all(state['step'].item() == 1 for state in optimizer.state.values())
