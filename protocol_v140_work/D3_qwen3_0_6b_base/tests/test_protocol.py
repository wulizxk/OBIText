import copy
from benchmark_protocol_v140 import EVALUATION_SUITES, EXPECTED_MODE_COUNTS, SUITE_QUESTION_COUNTS
from collections import Counter

import numpy as np
import pytest
import torch

from d3.data import SpanSampler, benchmark, pad_inputs, training_batch
from d3.decoding import autoregressive_beam_search, canonical_topk
from d3.engine import (seed_all, rng_state, restore_rng, save_checkpoint, load_checkpoint,
                       model_from_checkpoint, scheduler_for, validate_config)
from d3.io import read, rows, verify_data
from d3.metrics import aggregate, better, score_prediction
from d3.model import (restricted_logits, per_sample_losses, optimizer_for,
                      resize_and_initialize_embeddings)
from d3.protocol import FORMAL_CONFIG


def test_frozen_checksums(data_root):
    assert len(verify_data(data_root)) == 64


def test_full_validation_counts(data_root):
    questions, answers = benchmark(data_root, 'dev_full')
    assert len(questions) == 15024
    assert len({q['source_id'] for q in questions}) == 4689
    assert Counter(q['gap_length'] for q in questions) == {1: 4689, 2: 3911, 3: 2871, 4: 2017, 5: 1536}
    assert all(len(answers[q['example_id']]) == q['gap_length'] for q in questions)


def test_random_masks_include_boundaries_with_protected_targets(data_root):
    questions, answers = benchmark(data_root, 'test_random')
    count = SUITE_QUESTION_COUNTS['test_random']
    assert len(questions) == len(answers) == count
    assert len({q['source_id'] for q in questions}) == count
    assert Counter('1' if q['gap_length'] == 1 else '2' if q['gap_length'] == 2 else '3-5'
                   for q in questions) == EXPECTED_MODE_COUNTS['test_random']
    source_rows = {row['sequence_id']: row for row in rows(data_root / 'splits/test.jsonl')}
    vocab = read(data_root / 'target_vocab.json')
    targets = set(vocab['tokens'])
    support = vocab['training_eligible_position_frequency']
    sides = Counter()
    for q in questions:
        row = source_rows[q['source_id']]
        start, length = q['start_unit'], q['gap_length']
        gold = answers[q['example_id']]
        assert row['group_id'] == q['group_id']
        assert 1 <= length <= 5 and len(gold) == length
        assert q['prefix_units'] or q['suffix_units']
        assert q['prefix_units'] == row['model_tokens'][:start]
        assert q['suffix_units'] == row['model_tokens'][start + length:]
        assert q['prefix_units'] + gold + q['suffix_units'] == row['model_tokens']
        assert start in row['eligible_starts'][str(length)]
        for unit in row['units'][start:start + length]:
            assert unit['targetable'] and not unit['protection_reasons']
            assert unit['role'] == 'main'
            assert unit['token'] in targets and support[unit['token']] > 0
            assert unit['token'] not in ('[LACUNA_1]', '[LACUNA_N]', '<ANNOTATION_BOUNDARY>')
            assert not unit['token'].startswith('<LIG-')
        side = 'initial' if not q['prefix_units'] else 'final' if not q['suffix_units'] else 'internal'
        sides[side] += 1
    assert sides == EVALUATION_SUITES['test_random']['context_sides']


def test_prefix_suffix_training_and_test_protocol(data_root, codec):
    sampler = SpanSampler(data_root, 42)
    length_one = sampler.pools[1]
    assert any(0 in item['starts'] for item in length_one)
    assert any(len(sampler.rows[item['source_id']]['model_tokens']) - 1 in item['starts'] for item in length_one)
    for suite, side in (('test_prefix', 'prefix'), ('test_suffix', 'suffix')):
        questions, answers = benchmark(data_root, suite)
        count = SUITE_QUESTION_COUNTS[suite]
        assert len(questions) == len(answers) == count
        assert {q['gap_length'] for q in questions} == {1}
        assert Counter(q['boundary_side'] for q in questions) == {side: count}
        assert len({q['source_id'] for q in questions}) == count
        assert not {q['source_id'] for q in questions} & set(sampler.rows)
        for q in questions:
            assert q['start_unit'] == (0 if side == 'prefix' else len(q['prefix_units']))
            assert not q['prefix_units'] if side == 'prefix' else not q['suffix_units']
        q = questions[0]
        encoded = codec.encode(dict(q, gold_units=answers[q['example_id']]), supervised=True)
        assert encoded['input_ids'] and encoded['labels']


def test_inference_fim_prompt_is_gold_free(codec):
    q = {'gap_length': 2, 'prefix_units': ['王', '[LACUNA_1]'],
         'suffix_units': ['吉'], 'gold_units': ['王', '王']}
    first = codec.encode(q)
    q['gold_units'] = ['not a target', 'still not a target']
    assert codec.encode(q) == first
    assert 'labels' not in first
    c = codec.mapping['control_to_model_id']
    expected = ([c['<PRE>']]
                + [codec.mapping['unit_to_model_id'][x] for x in ['王', '[LACUNA_1]']]
                + [c['<SUF>'], codec.mapping['unit_to_model_id']['吉'], c['<LEN_2>'], c['<MID>']])
    assert first['input_ids'] == expected
    assert first['prompt_length'] == len(expected)


def test_supervision_is_answer_plus_end_and_causally_shifted(codec):
    q = {'gap_length': 2, 'prefix_units': ['王'], 'suffix_units': ['吉'],
         'gold_units': ['受', '年']}
    encoded = codec.encode(q, True)
    supervised = [x for x in encoded['labels'] if x != -100]
    assert supervised == [codec.mapping['target_unit_to_model_id']['受'],
                          codec.mapping['target_unit_to_model_id']['年'], codec.end_gap_id]
    batch = training_batch([encoded], codec.native_to_candidate,
                           codec.tokenizer.pad_token_id, 8, 'cpu')
    assert batch['positions'].tolist() == [encoded['prompt_length'] - 1,
                                          encoded['prompt_length'], encoded['prompt_length'] + 1]
    assert batch['targets'].tolist() == [codec.target_index['受'], codec.target_index['年'], len(codec.units)]
    assert batch['counts'].tolist() == [3.0]


def test_protected_context_attention_and_target_boundary(codec):
    ligand = next(t for t in codec.mapping['unit_to_model_id'] if t.startswith('<LIG-'))
    named_unknown = next(t for t in codec.units if t.startswith('[UNK-'))
    q = {'gap_length': 1,
         'prefix_units': ['王', '[LACUNA_1]', '[LACUNA_N]', ligand],
         'suffix_units': ['<ANNOTATION_BOUNDARY>', '吉'], 'gold_units': [named_unknown]}
    encoded = codec.encode(q, True)
    assert set(encoded['attention_mask']) == {1}
    assert sum(x != -100 for x in encoded['labels']) == 2  # answer + END
    assert all(t not in codec.units for t in ('[LACUNA_1]', '[LACUNA_N]', ligand, '<ANNOTATION_BOUNDARY>'))
    assert named_unknown in codec.units


@pytest.mark.parametrize('text', [
    '王[MASK]□吉', '王[MASK][MASK]…吉', '王[MASK]<ANNOTATION_BOUNDARY>吉',
])
def test_user_input(codec, text):
    encoded = codec.encode(codec.parse_text(text))
    assert encoded['gap_length'] in range(1, 6)
    assert encoded['input_ids'][-1] == codec.mapping['control_to_model_id']['<MID>']


@pytest.mark.parametrize('text', [
    '王□吉', '王[MASK]其[MASK]吉', '王<ANNOTATION_BOUNDARY>[MASK]', '[MASK]', '王[MASK] 吉',
])
def test_invalid_input(codec, text):
    with pytest.raises(ValueError):
        codec.encode(codec.parse_text(text))


def test_sampler_deterministic_protection_and_lengths(data_root, codec):
    a, b = SpanSampler(data_root, 42), SpanSampler(data_root, 42)
    counts = Counter()
    for _ in range(1000):
        q = a.sample()
        assert q == b.sample()
        counts[q['gap_length']] += 1
        encoded = codec.encode(q, True)
        assert sum(x != -100 for x in encoded['labels']) == q['gap_length'] + 1
        assert all(t in codec.units for t in q['gold_units'])
    assert all(150 < counts[k] < 250 for k in range(1, 6))


def test_expert_restorations_are_real_training_targets(data_root, codec):
    sampler = SpanSampler(data_root, 42)
    restored = [(item['source_id'], start) for item in sampler.pools[1]
                for start in item['starts'] if sampler.rows[item['source_id']]['units'][start]['restored']]
    assert len(restored) == 37196
    source_id, start = restored[0]
    row = sampler.rows[source_id]
    unit = row['units'][start]
    assert unit['targetable'] and not unit['protection_reasons'] and unit['token'] in codec.units
    question = {'source_id': source_id, 'gap_length': 1, 'start_unit': start,
                'prefix_units': row['model_tokens'][:start],
                'suffix_units': row['model_tokens'][start + 1:], 'gold_units': [unit['token']]}
    assert sum(value != -100 for value in codec.encode(question, True)['labels']) == 2


def test_no_truncation(codec):
    with pytest.raises(ValueError, match='truncation'):
        codec.encode({'prefix_units': ['王'] * 125, 'suffix_units': [], 'gap_length': 1})


def test_beam_is_autoregressive_and_uses_generated_prefix():
    seen = []

    def score_next(rows):
        seen.extend(row[:] for row in rows)
        values = []
        for row in rows:
            if len(row) == 1:
                values.append([np.log(.6), np.log(.4)])
            elif row[-1] == 10:
                values.append([np.log(.51), np.log(.49)])
            else:
                values.append([np.log(.01), np.log(.99)])
        return np.array(values)

    result = autoregressive_beam_search(
        [{'input_ids': [9], 'gap_length': 2}], [10, 11], 12,
        score_next, lambda rows: np.zeros(len(rows)), 2, 2, 2,
    )[0]
    assert result['paths'][0] == [1, 1]  # .4 * .99 > .6 * .51
    assert seen[0] == [9]
    assert {row[-1] for row in seen if len(row) == 2} == {10, 11}


def test_multi_character_end_score_can_rerank_complete_paths():
    calls = []

    def score_next(rows):
        return np.tile(np.log([.6, .4]), (len(rows), 1))

    def score_end(rows):
        calls.extend(row[:] for row in rows)
        return np.array([-10.0 if row[-2:] == [10, 10] else 0.0 for row in rows])

    result = autoregressive_beam_search(
        [{'input_ids': [9], 'gap_length': 2}], [10, 11], 12,
        score_next, score_end, 2, 2, 2,
    )[0]
    assert calls
    assert result['paths'][0] != [0, 0]
    assert result['score_kind'] == 'fim_autoregressive_character_plus_end_log_probability'


def test_single_ranking_is_direct_and_does_not_use_end_score():
    end_called = []
    result = autoregressive_beam_search(
        [{'input_ids': [9], 'gap_length': 1}], list(range(20)), 20,
        lambda rows: -np.arange(20, dtype=float)[None, :],
        lambda rows: end_called.append(True), 10, 5, 4,
    )[0]
    assert not end_called
    assert result['paths'] == [[0], [1], [2], [3]]
    assert result['single_ranking'] == list(range(20))
    assert result['score_kind'] == 'fim_direct_character_log_probability'


def test_beam_ties_use_canonical_not_native_ids():
    result = autoregressive_beam_search(
        [{'input_ids': [9], 'gap_length': 2}], [20, 5, 7], 30,
        lambda rows: np.full((len(rows), 3), -np.log(3)),
        lambda rows: np.zeros(len(rows)), 3, 3, 3,
    )[0]
    assert result['paths'] == [[0, 0], [0, 1], [0, 2]]


def test_partial_topk_exactly_matches_stable_full_sort():
    rng = np.random.default_rng(7)
    for size in (10, 4116):
        for _ in range(20):
            scores = np.round(rng.normal(size=size), 2)
            assert np.array_equal(canonical_topk(scores, 10),
                                  np.argsort(-scores, kind='stable')[:10])


def test_vectorized_padding_preserves_ids_attention_and_multiple():
    ids, attention = pad_inputs([[1, 2, 3], [4], [5, 6]], 0, 8, 'cpu')
    assert ids.tolist() == [[1, 2, 3, 0, 0, 0, 0, 0],
                            [4, 0, 0, 0, 0, 0, 0, 0],
                            [5, 6, 0, 0, 0, 0, 0, 0]]
    assert attention.tolist() == [[1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0, 0, 0, 0]]


def test_full_mrr_is_not_top10_truncated():
    tokens = [f't{i}' for i in range(30)]
    prediction = autoregressive_beam_search(
        [{'input_ids': [9], 'gap_length': 1}], list(range(30)), 30,
        lambda rows: -np.arange(30, dtype=float)[None, :], None, 10, 5, 10,
    )[0]
    scored = score_prediction(
        {'example_id': 'q', 'source_id': 's', 'group_id': 'g', 'gap_length': 1},
        prediction, ['t24'], tokens,
    )
    assert scored['gold_rank_full_vocabulary'] == 25
    assert scored['reciprocal_rank'] == 1 / 25
    assert scored['hit10'] == 0


def test_macro_and_tie_breaking():
    values = []
    for length in range(1, 6):
        for i in range(10 if length == 1 else 1):
            values.append({
                'source_id': f'{length}-{i}', 'gap_length': length,
                'em1': float(length == 1), 'hit5': 1.0,
                'correct_positions': int(length == 1),
                'hit10': 1.0 if length == 1 else None,
                'reciprocal_rank': 1.0 if length == 1 else None,
            })
    metrics = aggregate(values)
    assert metrics['macro5_em1'] == .2
    assert metrics['pooled_not_macro']['em1'] == 10 / 14
    assert not better(metrics, metrics)
    assert better(metrics, None)
    assert better(dict(metrics, macro5_hit5=1.1), metrics)
    assert not better(dict(metrics, macro5_em1=.19, macro5_hit5=2), metrics)


def batch_samples():
    return [
        {'input_ids': [2, 3, 4, 5, 8], 'labels': [-100, -100, -100, 5, 8]},
        {'input_ids': [2, 3, 4, 6, 5, 8], 'labels': [-100, -100, -100, 6, 5, 8]},
        {'input_ids': [2, 3, 7, 5, 6, 8], 'labels': [-100, -100, 7, 5, 6, 8]},
    ]


def test_restricted_head_matches_full_causal_logits(tiny):
    batch = training_batch(batch_samples(), {5: 0, 6: 1, 7: 2, 8: 3}, 0, 8, 'cpu')
    candidates = torch.tensor([5, 6, 7, 8])
    tiny.eval()
    with torch.no_grad():
        all_logits = tiny(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        full = all_logits[batch['owners'], batch['positions']][:, candidates]
        restricted = restricted_logits(
            tiny, batch['input_ids'], batch['attention_mask'], batch['owners'],
            batch['positions'], candidates,
        )
    torch.testing.assert_close(full, restricted, atol=1e-6, rtol=1e-5)


def test_microbatch_gradient_matches_effective_batch(tiny):
    other = copy.deepcopy(tiny)
    rows = batch_samples()
    candidates = torch.tensor([5, 6, 7, 8])
    batch = training_batch(rows, {5: 0, 6: 1, 7: 2, 8: 3}, 0, 8, 'cpu')
    per_sample_losses(tiny, batch, candidates).mean().backward()
    for part in (rows[:2], rows[2:]):
        batch = training_batch(part, {5: 0, 6: 1, 7: 2, 8: 3}, 0, 8, 'cpu')
        (per_sample_losses(other, batch, candidates).sum() / 3).backward()
    for first, second in zip(tiny.parameters(), other.parameters()):
        torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-4)


def test_qwen_rmsnorm_parameters_are_excluded_from_weight_decay(tiny, cfg):
    optimizer = optimizer_for(tiny, cfg)
    decay_for = {id(parameter): group['weight_decay']
                 for group in optimizer.param_groups for parameter in group['params']}
    norms = [parameter for module in tiny.modules() if module.__class__.__name__.endswith('Norm')
             for parameter in module.parameters(recurse=False)]
    assert norms and all(decay_for[id(parameter)] == 0.0 for parameter in norms)
    assert decay_for[id(tiny.model.embed_tokens.weight)] == cfg['weight_decay']
    assert optimizer.defaults['foreach'] is False
    assert optimizer.defaults['fused'] is False


def test_all_tokenizer_added_embeddings_are_reinitialized_and_tied(tiny, cfg):
    from types import SimpleNamespace

    first, second = copy.deepcopy(tiny), copy.deepcopy(tiny)
    codec = SimpleNamespace(
        vocab_size=50,
        mapping={'original_vocab_size': 30, 'original_config_vocab_size': 40},
    )
    local_cfg = dict(
        cfg,
        tokenizer_new_id_start=30,
        tokenizer_added_embedding_initialization=(
            'normal_mean_0_std_initializer_range_all_ids_ge_start'),
    )
    native = first.get_input_embeddings().weight.detach().clone()
    torch.manual_seed(123)
    report = resize_and_initialize_embeddings(first, codec, local_cfg)
    torch.manual_seed(123)
    resize_and_initialize_embeddings(second, codec, local_cfg)
    after = first.get_input_embeddings().weight.detach()
    assert torch.equal(after[:30], native[:30])
    assert not torch.equal(after[30:40], native[30:40])
    assert torch.equal(after[30:], second.get_input_embeddings().weight.detach()[30:])
    assert first.get_input_embeddings().weight.data_ptr() == first.get_output_embeddings().weight.data_ptr()
    assert report == {
        'native_tokenizer_rows_preserved': 30,
        'native_config_reserved_rows_reinitialized': 10,
        'total_tokenizer_added_rows_initialized': 20,
        'initialization': 'normal_mean_0_std_initializer_range_all_ids_ge_start',
    }


def test_checkpoint_optimizer_rng_resume(tiny, tmp_path, cfg):
    class Sampler:
        import random
        rng = random.Random(42)

    sampler = Sampler()
    seed_all(42)
    optimizer = optimizer_for(tiny, cfg)
    scheduler = scheduler_for(optimizer, cfg)
    batch = training_batch(batch_samples(), {5: 0, 6: 1, 7: 2, 8: 3}, 0, 8, 'cpu')
    candidates = torch.tensor([5, 6, 7, 8])

    def step(model, opt, sched):
        opt.zero_grad()
        per_sample_losses(model, batch, candidates).mean().backward()
        opt.step()
        sched.step()

    step(tiny, optimizer, scheduler)
    saved = {'model': tiny.state_dict(), 'model_config': tiny.config.to_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'rng': rng_state(sampler)}
    path = tmp_path / 'last.pt'
    save_checkpoint(path, saved)
    expected_random = (torch.rand(3), np.random.rand(3), sampler.rng.random())
    step(tiny, optimizer, scheduler)
    checkpoint = load_checkpoint(path)
    restored = model_from_checkpoint(checkpoint, torch.device('cpu'))
    opt2 = optimizer_for(restored, cfg)
    sched2 = scheduler_for(opt2, cfg)
    opt2.load_state_dict(checkpoint['optimizer'])
    sched2.load_state_dict(checkpoint['scheduler'])
    restore_rng(checkpoint['rng'], sampler)
    torch.testing.assert_close(torch.rand(3), expected_random[0], rtol=0, atol=0)
    assert np.array_equal(np.random.rand(3), expected_random[1])
    assert sampler.rng.random() == expected_random[2]
    step(restored, opt2, sched2)
    for first, second in zip(tiny.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert scheduler.get_last_lr() == sched2.get_last_lr()


@pytest.mark.parametrize('key,value', [
    ('dapt', True), ('max_epochs', 30), ('validation_suite', 'dev_random'),
    ('beam_width', 5), ('early_stopping', False), ('early_stopping_patience', 4),
    ('architecture', 'encoder'), ('generation_constraint', 'unconstrained'),
])
def test_invalid_formal_config(cfg, key, value):
    with pytest.raises(ValueError):
        validate_config(dict(cfg, **{key: value}))


def test_every_formal_config_field_is_strictly_locked(cfg):
    assert cfg == FORMAL_CONFIG

    def altered(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return value + .125
        if isinstance(value, str):
            return value + '-changed'
        if isinstance(value, list):
            return value + [value[0]]
        raise TypeError(type(value))

    for key, value in FORMAL_CONFIG.items():
        with pytest.raises(ValueError, match=key):
            validate_config(dict(cfg, **{key: altered(value)}))
    with pytest.raises(ValueError, match='missing'):
        validate_config({k: v for k, v in cfg.items() if k != 'learning_rate'})
    with pytest.raises(ValueError, match='extra'):
        validate_config(dict(cfg, unauthorized=True))
    with pytest.raises(ValueError, match='max_epochs'):
        validate_config(dict(cfg, max_epochs=50.0))
    with pytest.raises(ValueError, match='early_stopping'):
        validate_config(dict(cfg, early_stopping=1))
