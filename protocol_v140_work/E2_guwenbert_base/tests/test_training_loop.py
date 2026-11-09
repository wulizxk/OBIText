"""Exercise the real epoch/checkpoint loop with a tiny, synthetic, non-research fixture."""
import copy
import random

import pytest
import torch

from e2 import engine
from e2.data import training_batch
from e2.io import read, write


class MiniCodec:
    target_ids = [5, 6, 7]

    def __init__(self, data, cfg):
        pass

    def collate(self, questions, device):
        rows = [{'input_ids': [2, 3, 4], 'labels': [-100, q['gold'], -100]} for q in questions]
        return training_batch(rows, {5: 0, 6: 1, 7: 2}, 0, 8, device)

    def encode(self, question, supervised=False):
        return {'input_ids': [2, 3, 4], 'positions': [1]}


class MiniSampler:
    def __init__(self, data, seed):
        self.rng = random.Random(seed)

    def sample(self):
        return {'gold': self.rng.choice([5, 6, 7]), 'gap_length': 1}


def test_actual_epoch_resume_equivalence(tiny, tmp_path, cfg, monkeypatch):
    cfg = dict(cfg, max_epochs=3, samples_per_epoch=5, micro_batch_size=2, effective_batch_size=4,
               precision='fp32', warmup_epochs=1)
    identity = {'cfg': cfg, 'seed': 42, 'fixture': 'not-real-data'}
    write(tmp_path / 'benchmark/protocol.json', {'validation_length_counts': {str(i): 1 for i in range(1, 6)}})
    public = [{'gap_length': i} for i in range(1, 6)]
    monkeypatch.setattr(engine, 'Codec', MiniCodec)
    monkeypatch.setattr(engine, 'SpanSampler', MiniSampler)
    monkeypatch.setattr(engine, 'benchmark', lambda *_: (public, {}))
    monkeypatch.setattr(engine, 'load_pretrained', lambda *_: (copy.deepcopy(tiny), {}))
    metrics = {'macro5_em1': .2, 'macro5_hit5': .4}
    monkeypatch.setattr(engine, 'evaluate', lambda *_: (metrics, []))
    continuous, interrupted = tmp_path / 'continuous', tmp_path / 'interrupted'
    continuous.mkdir(); interrupted.mkdir()
    args = (tmp_path, tmp_path, tmp_path)
    engine._train_locked(*args, continuous, cfg, 42, torch.device('cpu'), False, identity)
    native_save = engine.save_checkpoint

    def crash_after_epoch_one(path, state):
        native_save(path, state)
        if path.name == 'last.pt' and state['epoch'] == 1:
            raise InterruptedError('Simulated interruption after a committed epoch')
    monkeypatch.setattr(engine, 'save_checkpoint', crash_after_epoch_one)
    with pytest.raises(InterruptedError):
        engine._train_locked(*args, interrupted, cfg, 42, torch.device('cpu'), False, identity)
    monkeypatch.setattr(engine, 'save_checkpoint', native_save)
    engine._train_locked(*args, interrupted, cfg, 42, torch.device('cpu'), True, identity)
    a, b = (engine.load_checkpoint(p / 'last.pt') for p in (continuous, interrupted))
    assert a['epoch'] == b['epoch'] == 3
    assert a['step'] == b['step'] == 6
    assert a['best']['epoch'] == b['best']['epoch'] == 1  # exact metric ties keep earlier
    assert a['rng']['sampler'] == b['rng']['sampler']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['scheduler'] == b['scheduler']


def test_early_stop_after_five_consecutive_non_improvements(tiny, tmp_path, cfg, monkeypatch):
    cfg = dict(cfg, max_epochs=50, samples_per_epoch=5, micro_batch_size=2,
               effective_batch_size=4, precision='fp32', warmup_epochs=1)
    identity = {'cfg': cfg, 'seed': 42, 'fixture': 'early-stop'}
    write(tmp_path / 'benchmark/protocol.json',
          {'validation_length_counts': {str(i): 1 for i in range(1, 6)}})
    monkeypatch.setattr(engine, 'Codec', MiniCodec)
    monkeypatch.setattr(engine, 'SpanSampler', MiniSampler)
    monkeypatch.setattr(engine, 'benchmark', lambda *_: ([{'gap_length': i} for i in range(1, 6)], {}))
    monkeypatch.setattr(engine, 'load_pretrained', lambda *_: (copy.deepcopy(tiny), {}))
    metrics = {'macro5_em1': .2, 'macro5_hit5': .4}
    monkeypatch.setattr(engine, 'evaluate', lambda *_: (metrics, []))
    output = tmp_path / 'early'
    output.mkdir()
    engine._train_locked(tmp_path, tmp_path, tmp_path, output, cfg, 42,
                         torch.device('cpu'), False, identity)
    checkpoint = engine.load_checkpoint(output / 'last.pt')
    marker = read(output / 'TRAINING_COMPLETE.json')
    assert checkpoint['epoch'] == 6 and checkpoint['step'] == 12
    assert checkpoint['best']['epoch'] == 1
    assert marker['epochs_ran'] == 6
    assert marker['stopping'] == {'reason': 'early_stopping', 'patience': 5,
                                  'epochs_without_improvement': 5}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA FP16 overflow regression')
def test_amp_overflow_retries_same_batch(tiny, cfg, monkeypatch):
    # Inject one infinite gradient; the first attempt must not change parameters,
    # consume an optimizer step, or alter dropout draws for the retry.
    device = torch.device('cuda:0')
    tiny.to(device)
    cfg = dict(cfg, micro_batch_size=2, precision='fp16')
    optimizer = engine.optimizer_for(tiny, cfg)
    scaler = torch.amp.GradScaler('cuda', init_scale=16)
    original_losses = engine.per_sample_losses
    draws = []

    def instrumented(*args):
        draws.append(torch.rand(4, device=device).cpu())
        return original_losses(*args)
    monkeypatch.setattr(engine, 'per_sample_losses', instrumented)
    called = [0]

    def overflow_once(grad):
        called[0] += 1
        return grad * float('inf') if called[0] == 1 else grad
    hook = tiny.get_input_embeddings().weight.register_hook(overflow_once)
    loss, norm = engine.update(tiny, MiniCodec(None, cfg), [{'gold': 5}, {'gold': 6}], cfg,
                               device, optimizer, scaler)
    hook.remove()
    assert called[0] == 2
    assert torch.equal(draws[0], draws[1])
    assert scaler.get_scale() == 8
    assert all(s['step'].item() == 1 for s in optimizer.state.values())
    assert loss > 0 and norm > 0
