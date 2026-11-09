"""Architecture-specific regressions beyond the shared experiment lifecycle."""
from types import SimpleNamespace

import numpy as np
import torch

from ed3.data import training_batch, pad_inputs
from ed3.decoding import predict_encoded_batch
from ed3.model import resize_embeddings, restricted_logits


def test_resize_preserves_native_rows_and_tied_head(tiny):
    inputs = tiny.model.shared.weight.detach().clone()
    outputs = tiny.lm_head.weight.detach().clone()
    report = resize_embeddings(tiny, 138)
    assert report['new_rows_initialized'] == 10
    torch.testing.assert_close(tiny.model.shared.weight[:128], inputs, rtol=0, atol=0)
    torch.testing.assert_close(tiny.lm_head.weight[:128], outputs, rtol=0, atol=0)
    assert tiny.model.shared.weight.data_ptr() == tiny.model.encoder.embed_tokens.weight.data_ptr()
    assert tiny.model.shared.weight.data_ptr() == tiny.model.decoder.embed_tokens.weight.data_ptr()
    assert tiny.model.shared.weight.data_ptr() == tiny.lm_head.weight.data_ptr()
    assert torch.isfinite(tiny.model.shared.weight).all() and torch.isfinite(tiny.lm_head.weight).all()


def test_future_gold_cannot_change_first_character_logits(tiny):
    candidates = torch.tensor([5, 6, 7, 8])
    outputs = []
    for labels in ([5, 6, 8], [7, 5, 8]):
        batch = training_batch([{'input_ids': [2, 3, 4], 'labels': labels}],
                               {5: 0, 6: 1, 7: 2, 8: 3}, 0, 8, 'cpu')
        outputs.append(restricted_logits(tiny, batch['input_ids'], batch['attention_mask'],
                       batch['owners'], batch['positions'], candidates,
                       batch['decoder_input_ids'], batch['decoder_attention_mask']).detach())
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    assert not torch.equal(outputs[0][1], outputs[1][1])


def test_decoder_start_pad_remains_visible_while_right_padding_is_masked():
    batch = training_batch([{'input_ids': [2, 3], 'labels': [5, 8]},
                            {'input_ids': [4], 'labels': [6, 5, 8]}],
                           {5: 0, 6: 1, 8: 2}, 0, 8, 'cpu')
    assert batch['decoder_input_ids'][:, 0].tolist() == [102, 102]
    assert batch['decoder_attention_mask'][:, 0].tolist() == [1, 1]
    assert batch['decoder_attention_mask'].sum(1).tolist() == [2, 3]
    assert batch['attention_mask'].sum(1).tolist() == [2, 1]
    assert batch['positions'].tolist() == [0, 1, 0, 1, 2]


def test_batched_beam_matches_independent_questions_and_encodes_once(tiny, cfg):
    codec = SimpleNamespace(target_ids=list(range(5, 17)), candidate_ids=list(range(5, 18)),
                            end_gap_id=17, decoder_start_id=102, tokenizer=SimpleNamespace(pad_token_id=0))
    questions = [{'input_ids': [2, 3, 4], 'gap_length': 1},
                 {'input_ids': [9, 2], 'gap_length': 2},
                 {'input_ids': [3, 4, 9, 2], 'gap_length': 5}]
    calls = []
    hook = tiny.model.encoder.register_forward_hook(lambda *_: calls.append(1))
    config = dict(cfg, precision='fp32', eval_forward_batch_size=7)
    batched = predict_encoded_batch(tiny, codec, questions, config, torch.device('cpu'))
    assert len(calls) == 1
    hook.remove()
    independent = [predict_encoded_batch(tiny, codec, [q], config, torch.device('cpu'))[0] for q in questions]
    for a, b in zip(batched, independent):
        assert a['paths'] == b['paths']
        assert a['single_ranking'] == b['single_ranking']
        np.testing.assert_allclose(a['scores'], b['scores'], rtol=1e-5, atol=1e-5)
    assert [len(p['paths'][0]) for p in batched] == [1, 2, 5]


def test_single_prediction_matches_full_native_bart_forward(tiny, cfg):
    codec = SimpleNamespace(target_ids=list(range(5, 17)), candidate_ids=list(range(5, 18)),
                            end_gap_id=17, decoder_start_id=102, tokenizer=SimpleNamespace(pad_token_id=0))
    encoded = [{'input_ids': [2, 3, 4], 'gap_length': 1}]
    config = dict(cfg, precision='fp32')
    result = predict_encoded_batch(tiny, codec, encoded, config, torch.device('cpu'))[0]
    ids, mask = pad_inputs([[2, 3, 4]], 0, 8, 'cpu')
    with torch.no_grad():
        logits = tiny(input_ids=ids, attention_mask=mask, decoder_input_ids=torch.tensor([[102]]),
                      decoder_attention_mask=torch.ones(1, 1, dtype=torch.long)).logits[0, 0, codec.target_ids]
    order = torch.argsort(logits, descending=True, stable=True).tolist()
    assert result['single_ranking'] == order
    assert result['paths'] == [[i] for i in order[:10]]


def test_no_test_answers_in_smoke_source():
    from ed3 import smoke
    import inspect
    source = inspect.getsource(smoke.smoke)
    assert 'benchmark_gold' not in source and 'benchmark_public' not in source


def test_formal_plan_only_uses_gpu_zero():
    from ed3 import formal_controller as c
    assert c.SEEDS == [(42, 0), (3407, 0), (2026, 0)]


def test_supervised_encoding_input_equal_to_public_encoding_all_lengths(codec, data_root):
    from ed3.data import SpanSampler
    sampler = SpanSampler(data_root, 42)
    for _ in range(200):
        q = sampler.sample()
        supervised, public = codec.encode(q, True), codec.encode(q)
        assert supervised['input_ids'] == public['input_ids']
        assert supervised['attention_mask'] == public['attention_mask']
        assert len(supervised['labels']) == q['gap_length'] + 1
