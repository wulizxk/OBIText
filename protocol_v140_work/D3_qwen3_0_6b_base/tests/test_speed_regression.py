"""Speed-only changes: sample weighting, AMP boundary and cache lifetime."""
import copy
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from d3 import decoding, engine
from d3.data import Codec, training_batch
from d3.io import sha
from d3.protocol import FORMAL_CONFIG_SHA256, validate_formal_config


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
    assert cfg['micro_batch_token_budget'] == 1024
    assert cfg['gradient_checkpointing'] is True and cfg['precision'] == 'fp16'
    assert cfg['max_epochs'] == 50 and cfg['early_stopping_patience'] == 5
    assert sha(Path(__file__).resolve().parents[1] / 'configs/d3a.json') == FORMAL_CONFIG_SHA256
    with pytest.raises(ValueError, match='protocol change'):
        validate_formal_config(dict(cfg, micro_batch_size=4))


@pytest.mark.parametrize('width,expected', [(31, 32), (32, 32), (33, 16), (64, 16), (65, 8), (128, 8)])
def test_length_guard_is_deterministic_and_padding_aware(cfg, width, expected):
    calls = []
    def encode(question, supervised):
        calls.append(supervised)
        return {'input_ids': [1] * question}
    codec = SimpleNamespace(encode=encode)
    values = [10, width, 9, 13]
    before = list(values)
    actual = Codec.training_micro_batch_size(codec, values, cfg)
    assert actual == expected and values == before and all(calls)
    assert actual * ((width + 7) // 8 * 8) <= cfg['micro_batch_token_budget']


@pytest.mark.parametrize('count,micro', [(13, 32), (64, 32), (64, 16), (64, 8)])
def test_micro_change_preserves_objective_gradients_and_update(tiny, cfg, count, micro):
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
    assert all(state['step'].item() == 1 for _, _, opt in records for state in opt.state.values())


def test_empty_batch_fails_before_model_or_optimizer(cfg):
    with pytest.raises(ValueError, match='Empty effective'):
        Codec.training_micro_batch_size(None, [], cfg)
    with pytest.raises(ValueError, match='Empty effective'):
        engine.update(None, None, [], cfg, torch.device('cpu'), None, None)


def test_selector_is_reused_across_amp_retry(tiny, cfg, monkeypatch):
    class GuardedCodec(VariableCodec):
        choices = 0
        seen = []
        def training_micro_batch_size(self, questions, config):
            self.choices += 1
            return 8
        def collate(self, questions, device):
            self.seen.append(list(questions))
            return super().collate(questions, device)
    class Scaler:
        value = 8
        steps = 0
        def scale(self, value): return value
        def unscale_(self, _): pass
        def is_enabled(self): return True
        def get_scale(self): return self.value
        def update(self, new_scale=None):
            if new_scale is not None: self.value = new_scale
        def step(self, optimizer):
            self.steps += 1
            optimizer.step()
    codec, scaler = GuardedCodec(), Scaler()
    optimizer = engine.optimizer_for(tiny, cfg)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_',
                        lambda *a, **k: torch.tensor(float('inf') if scaler.value > 4 else 1.0))
    engine.update(tiny, codec, list(range(13)), dict(cfg, precision='fp32'),
                  torch.device('cpu'), optimizer, scaler)
    assert codec.choices == 1 and scaler.steps == 1
    assert codec.seen == [list(range(8)), list(range(8, 13))] * 2


def legacy_predict(model, codec, encoded, cfg, device):
    """Disable only the new outer context, retaining original per-forward AMP."""
    actual_autocast = decoding.autocast
    first = True
    def context(dev, config):
        nonlocal first
        if first:
            first = False
            return nullcontext()
        return actual_autocast(dev, config)
    with patch.object(decoding, 'autocast', context):
        return decoding.predict_encoded_batch(model, codec, encoded, cfg, device)


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_decoding_cache_preserves_all_rankings_scores_and_expires(tiny, cfg, device_name):
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Bounded GPU diagnostic only')
    device = torch.device(device_name)
    model = tiny.to(device).eval()
    codec = SimpleNamespace(target_ids=list(range(5, 25)), candidate_ids=list(range(5, 26)),
                            end_gap_id=25, tokenizer=SimpleNamespace(pad_token_id=0))
    # Mixed lengths, unequal prompt sizes, chunk boundaries and END reranking.
    encoded = [{'input_ids': [2] + [3] * length + [4], 'gap_length': length} for length in range(1, 6)]
    local_cfg = dict(cfg, eval_forward_batch_size=7)
    for iteration in range(2):
        expected = legacy_predict(model, codec, encoded, local_cfg, device)
        actual = decoding.predict_encoded_batch(model, codec, encoded, local_cfg, device)
        assert actual == expected  # Includes complete 20-target singleton rankings and exact scores.
        assert not torch.is_autocast_enabled('cuda')
        assert torch.is_grad_enabled()
        if iteration == 0:
            # Mutation outside decoding cannot observe stale cached weights.
            with torch.no_grad():
                model.model.layers[0].mlp.down_proj.weight.add_(0.07)


def test_outer_eval_context_ends_before_gold_join(tiny, cfg, monkeypatch):
    depth = 0
    @contextmanager
    def context(*_):
        nonlocal depth
        depth += 1
        try:
            yield
        finally:
            depth -= 1
    monkeypatch.setattr(engine, 'autocast', context)
    def predict(*args):
        assert depth == 1 and not torch.is_grad_enabled()
        return [{'paths': [[0]], 'scores': [0.0], 'single_ranking': [0],
                 'score_kind': 'fim_direct_character_log_probability'} for _ in args[2]]
    monkeypatch.setattr(engine, 'predict_encoded_batch', predict)
    public = [{'example_id': str(i), 'source_id': str(i), 'group_id': str(i), 'gap_length': 1} for i in range(3)]
    def gold():
        assert depth == 0 and torch.is_grad_enabled()
        return {q['example_id']: ['王'] for q in public}
    codec = SimpleNamespace(target_ids=[5], units=['王'])
    engine.evaluate(tiny, codec, public, gold, dict(cfg, eval_question_batch_size=2),
                    torch.device('cpu'), [{'input_ids': [2, 3, 4]} for _ in public])
    assert depth == 0
