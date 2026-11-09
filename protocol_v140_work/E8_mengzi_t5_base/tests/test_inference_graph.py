"""Exact replay, bounded storage and per-evaluation weight lifecycle checks."""
from types import SimpleNamespace

import pytest
import torch

from ed2 import inference_graph as graph, runtime
from ed2.decoding import predict_encoded_batch
from ed2.model import autocast

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph regression')


def test_cpu_fallback_and_closed_session(tiny, cfg):
    session = graph.EvaluationGraphSession(tiny, cfg, 'cpu')
    with session:
        assert not session.enabled
        assert session.call('x', lambda x: x + 1, (torch.tensor(2),)).item() == 3
    assert not session.cache and session.closed
    with pytest.raises(RuntimeError, match='closed'):
        session.call('x', lambda x: x, (torch.tensor(2),))
    with pytest.raises(RuntimeError, match='closed'):
        session.__enter__()


def test_unknown_backend_rejected(tiny, cfg):
    with pytest.raises(ValueError, match='Unknown evaluation backend'):
        graph.EvaluationGraphSession(tiny, dict(cfg, evaluation_backend='unknown'), 'cpu')


@cuda
def test_replay_matches_changing_inputs_without_output_alias(cfg):
    runtime.configure_execution(42)
    model = torch.nn.Linear(16, 16).cuda().eval()
    held = []
    with graph.EvaluationGraphSession(model, cfg, 'cuda:0') as session:
        for _ in range(8):
            inputs = torch.randn(4, 16, device='cuda')
            with torch.no_grad(), autocast('cuda:0', cfg):
                expected = model(inputs)
                actual = session.call('linear', model, (inputs,))
            assert torch.equal(actual, expected)
            held.append((actual, expected.clone()))
        assert session.stats['captures'] == 1 and session.stats['replays'] == 6
        assert session.stats['capture_bitwise_checks_passed'] == 1
        assert all(torch.equal(a, b) for a, b in held)
    assert not session.cache


@cuda
def test_training_mode_gradients_and_weight_mutation_rejected(cfg):
    model = torch.nn.Linear(4, 4).cuda()
    with pytest.raises(RuntimeError, match='evaluation-only'):
        graph.EvaluationGraphSession(model, cfg, 'cuda:0')
    model.eval()
    with graph.EvaluationGraphSession(model, cfg, 'cuda:0') as session:
        with pytest.raises(RuntimeError, match='disabled gradients'):
            session.call('linear', model, (torch.ones(1, 4, device='cuda'),))
    with pytest.raises(RuntimeError, match='parameters changed'):
        with graph.EvaluationGraphSession(model, cfg, 'cuda:0') as session:
            with torch.no_grad():
                model.weight.add_(1)
    assert session.closed and not session.cache


@cuda
def test_rare_shapes_capacity_and_noncontiguous_fallback(cfg, monkeypatch):
    runtime.configure_execution(42)
    monkeypatch.setattr(graph, 'MAX_ENTRIES', 1)
    model = torch.nn.Linear(4, 4).cuda().eval()
    with graph.EvaluationGraphSession(model, cfg, 'cuda:0') as session:
        for shape in ((2, 4), (3, 4)):
            for _ in range(4):
                x = torch.randn(shape, device='cuda')
                with torch.no_grad(), autocast('cuda:0', cfg):
                    assert torch.equal(session.call('linear', model, (x,)), model(x))
        x = torch.randn(4, 2, device='cuda').T
        assert not x.is_contiguous()
        with torch.no_grad(), autocast('cuda:0', cfg):
            assert torch.equal(session.call('linear', model, (x,)), model(x))
        assert len(session.cache) == 1
        assert session.stats['captures'] == 1 and session.stats['eager_calls'] == 7


@cuda
def test_fresh_session_after_optimizer_step_cannot_reuse_stale_weights(cfg):
    runtime.configure_execution(42)
    model = torch.nn.Linear(4, 4).cuda().eval()
    x = torch.randn(2, 4, device='cuda')
    outputs = []
    for epoch in range(2):
        if epoch:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()
        with graph.EvaluationGraphSession(model, cfg, 'cuda:0') as session:
            with torch.no_grad(), autocast('cuda:0', cfg):
                for _ in range(4):
                    result = session.call('linear', model, (x,))
                assert torch.equal(result, model(x))
            assert session.stats['captures'] == 1
            outputs.append(result)
    assert not torch.equal(*outputs)


@cuda
def test_full_beam_paths_scores_and_rankings_bitwise_equal(tiny, cfg):
    runtime.configure_execution(42)
    tiny.cuda().eval()
    codec = SimpleNamespace(target_ids=list(range(5, 17)), candidate_ids=list(range(5, 18)),
                            end_gap_id=17, decoder_start_id=0, tokenizer=SimpleNamespace(pad_token_id=0))
    device = torch.device('cuda:0')
    # Changing source contexts and gap lengths exercise owner reordering and
    # END_GAP rescoring. Both implementations retain precisely the same shapes.
    with graph.EvaluationGraphSession(tiny, cfg, device) as session:
        for repetition in range(2):
            for gap in range(1, 6):
                encoded = [{'input_ids': [2, 3 + repetition, 4], 'gap_length': gap},
                           {'input_ids': [9, 2], 'gap_length': gap}]
                eager = predict_encoded_batch(tiny, codec, encoded, cfg, device)
                accelerated = predict_encoded_batch(tiny, codec, encoded, cfg, device, executor=session)
                assert eager == accelerated
        assert session.stats['captures'] > 0 and session.stats['replays'] > 0

