"""Batch-shape optimizations must preserve the effective training objective."""
import copy
import math

import pytest
import torch

from ed2 import engine
from ed2.model import scaler_for
from test_training_loop import MiniCodec


@pytest.mark.parametrize('count', [13, 64])
@pytest.mark.parametrize('micro', [8, 16, 32, 64])
def test_batch_shapes_preserve_full_and_tail_sample_normalization(tiny, cfg, count, micro):
    # Dropout-free FP32 fixture: changing partitions should preserve the
    # averaged gradient, including the final 13-sample effective batch.
    questions = [{'gold': 5 + i % 3} for i in range(count)]
    config = dict(cfg, precision='fp32', effective_batch_size=64,
                  gradient_clip=1e9, micro_batch_size=micro)
    reference_config = dict(config, micro_batch_size=64)
    candidate, reference = copy.deepcopy(tiny), copy.deepcopy(tiny)
    results = []
    for model, settings in ((candidate, config), (reference, reference_config)):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        result = engine.update(model, MiniCodec(None, settings), questions, settings,
                               torch.device('cpu'), optimizer, scaler_for('cpu', settings))
        assert all(math.isfinite(value) and value > 0 for value in result)
        results.append(result)
    torch.testing.assert_close(torch.tensor(results[0]), torch.tensor(results[1]), rtol=1e-5, atol=1e-5)
    for key, value in candidate.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=1e-5, atol=1e-6)
