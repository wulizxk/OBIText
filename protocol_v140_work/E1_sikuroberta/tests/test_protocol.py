import copy
from benchmark_protocol_v140 import EVALUATION_SUITES, EXPECTED_MODE_COUNTS, SUITE_QUESTION_COUNTS
from collections import Counter

import numpy as np
import pytest
import torch

from e1.data import SpanSampler, benchmark, pad_inputs, training_batch
from e1.decoding import beam_search, canonical_topk
from e1.engine import (seed_all, rng_state, restore_rng, save_checkpoint, load_checkpoint,
                       model_from_checkpoint, scheduler_for, validate_config)
from e1.io import read, rows, verify_data
from e1.metrics import aggregate, better, score_prediction
from e1.model import restricted_logits, per_sample_losses, optimizer_for


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


def test_gold_never_enters_input(codec):
    q = {'gap_length': 2, 'prefix_units': ['王', '[LACUNA_1]'], 'suffix_units': ['吉'],
         'gold_units': ['王', '王']}
    first = codec.encode(q)
    q['gold_units'] = ['arbitrary invalid gold', 'not in vocabulary']
    assert codec.encode(q) == first
    assert 'labels' not in first
    assert sum(x == codec.tokenizer.mask_token_id for x in first['input_ids']) == 2


def test_protected_context_attention(codec):
    ligand = next(t for t in codec.mapping['unit_to_model_id'] if t.startswith('<LIG-'))
    q = {'gap_length': 1, 'prefix_units': ['王', '[LACUNA_1]', '[LACUNA_N]', ligand],
         'suffix_units': ['<ANNOTATION_BOUNDARY>', '吉'], 'gold_units': ['王']}
    encoded = codec.encode(q, True)
    assert set(encoded['attention_mask']) == {1}
    assert sum(x != -100 for x in encoded['labels']) == 1
    assert all(t not in codec.units for t in ('[LACUNA_1]', '[LACUNA_N]', ligand, '<ANNOTATION_BOUNDARY>'))
    # A named undeciphered glyph is a valid distinct output, unlike natural damage.
    assert any(t.startswith('[UNK-') for t in codec.units)


@pytest.mark.parametrize('text', ['王[MASK]□吉', '王[MASK][MASK]…吉', '王[MASK]<ANNOTATION_BOUNDARY>吉'])
def test_user_input(codec, text):
    assert codec.encode(codec.parse_text(text))['positions']


@pytest.mark.parametrize('text', ['王□吉', '王[MASK]其[MASK]吉', '王<ANNOTATION_BOUNDARY>[MASK]', '[MASK]', '王[MASK] 吉'])
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
        assert len(encoded['positions']) == q['gap_length']
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
    assert len(codec.encode(question, True)['positions']) == 1


def test_no_truncation(codec):
    with pytest.raises(ValueError, match='truncation'):
        codec.encode({'prefix_units': ['王'] * 128, 'suffix_units': [], 'gap_length': 1})


def test_beam_is_conditional_not_independent():
    q = [{'input_ids': [8, 9, 9, 8], 'positions': [1, 2]}]
    seen = []

    def scores(ids, positions):
        seen.extend((row[:], pos) for row, pos in zip(ids, positions))
        return np.array([[np.log(.6), np.log(.4)] if p == 1 else
                         ([np.log(.51), np.log(.49)] if row[1] == 10 else [np.log(.01), np.log(.99)])
                         for row, p in zip(ids, positions)])
    result = beam_search(q, [10, 11], scores, 2, 2)[0]
    assert result['paths'][0] == [1, 1]  # .4*.99 > .6*.51
    assert seen[0][0] == [8, 9, 9, 8]
    assert {row[1] for row, pos in seen if pos == 2} == {10, 11}


def test_beam_ties_canonical_not_native_ids():
    result = beam_search([{'input_ids': [9, 9], 'positions': [0, 1]}], [20, 5, 7],
                         lambda ids, positions: np.full((len(ids), 3), -np.log(3)), 3, 3)[0]
    assert result['paths'] == [[0, 0], [0, 1], [0, 2]]


def test_partial_topk_exactly_matches_stable_full_sort():
    rng = np.random.default_rng(7)
    for size in (10, 4116):
        for _ in range(20):
            scores = np.round(rng.normal(size=size), 2)  # deliberately creates ties
            expected = np.argsort(-scores, kind='stable')[:10]
            assert np.array_equal(canonical_topk(scores, 10), expected)


def test_vectorized_padding_preserves_ids_attention_and_multiple():
    ids, attention = pad_inputs([[1, 2, 3], [4], [5, 6]], 0, 8, 'cpu')
    assert ids.tolist() == [[1, 2, 3, 0, 0, 0, 0, 0],
                            [4, 0, 0, 0, 0, 0, 0, 0],
                            [5, 6, 0, 0, 0, 0, 0, 0]]
    assert attention.tolist() == [[1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0, 0, 0, 0]]


def test_full_mrr_not_top10_truncation():
    tokens = [f't{i}' for i in range(30)]
    result = beam_search([{'input_ids': [9], 'positions': [0]}], list(range(30)),
                         lambda ids, positions: -np.arange(30, dtype=float)[None, :], 10, 5)[0]
    scored = score_prediction({'example_id': 'q', 'source_id': 's', 'group_id': 'g', 'gap_length': 1}, result, ['t24'], tokens)
    assert scored['gold_rank_full_vocabulary'] == 25
    assert scored['reciprocal_rank'] == 1 / 25
    assert scored['hit10'] == 0


def test_macro_and_tie_breaking():
    values = []
    for length in range(1, 6):
        for i in range(10 if length == 1 else 1):
            values.append({'source_id': f'{length}-{i}', 'gap_length': length, 'em1': float(length == 1),
                           'hit5': 1.0, 'correct_positions': int(length == 1), 'hit10': 1.0 if length == 1 else None,
                           'reciprocal_rank': 1.0 if length == 1 else None})
    m = aggregate(values)
    assert m['macro5_em1'] == .2
    assert m['pooled_not_macro']['em1'] == 10 / 14
    assert not better(m, m)
    assert better(m, None)
    assert better(dict(m, macro5_hit5=1.1), m)
    assert not better(dict(m, macro5_em1=.19, macro5_hit5=2), m)


def batch_samples():
    return [{'input_ids': [2, 3, 4], 'labels': [-100, 5, -100]},
            {'input_ids': [2, 3, 3, 4], 'labels': [-100, 6, 5, -100]},
            {'input_ids': [2, 3, 3, 3, 4], 'labels': [-100, 5, 6, 7, -100]}]


def test_restricted_head_matches_full(tiny):
    b = training_batch(batch_samples(), {5: 0, 6: 1, 7: 2}, 0, 8, 'cpu')
    targets = torch.tensor([5, 6, 7])
    tiny.eval()
    with torch.no_grad():
        all_logits = tiny(input_ids=b['input_ids'], attention_mask=b['attention_mask']).logits
        full = all_logits[b['owners'], b['positions']][:, targets]
        restricted = restricted_logits(tiny, b['input_ids'], b['attention_mask'], b['owners'], b['positions'], targets)
    torch.testing.assert_close(full, restricted, atol=1e-6, rtol=1e-5)


def test_microbatch_gradient_matches_effective_batch(tiny):
    other = copy.deepcopy(tiny)
    rows = batch_samples()
    targets = torch.tensor([5, 6, 7])
    b = training_batch(rows, {5: 0, 6: 1, 7: 2}, 0, 8, 'cpu')
    per_sample_losses(tiny, b, targets).mean().backward()
    for part in (rows[:2], rows[2:]):
        b = training_batch(part, {5: 0, 6: 1, 7: 2}, 0, 8, 'cpu')
        (per_sample_losses(other, b, targets).sum() / 3).backward()
    for p, q in zip(tiny.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


def test_checkpoint_optimizer_rng_resume(tiny, tmp_path, cfg):
    class Sampler:
        import random
        rng = random.Random(42)
    sampler = Sampler()
    seed_all(42)
    optimizer = optimizer_for(tiny, cfg)
    scheduler = scheduler_for(optimizer, cfg)
    b = training_batch(batch_samples(), {5: 0, 6: 1, 7: 2}, 0, 8, 'cpu')
    ids = torch.tensor([5, 6, 7])

    def step(model, opt, sched):
        opt.zero_grad()
        per_sample_losses(model, b, ids).mean().backward()
        opt.step(); sched.step()
    step(tiny, optimizer, scheduler)
    saved = {'model': tiny.state_dict(), 'model_config': tiny.config.to_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'rng': rng_state(sampler)}
    path = tmp_path / 'last.pt'
    save_checkpoint(path, saved)
    expected_random = (torch.rand(3), np.random.rand(3), sampler.rng.random())
    step(tiny, optimizer, scheduler)
    ckpt = load_checkpoint(path)
    restored = model_from_checkpoint(ckpt, torch.device('cpu'))
    opt2 = optimizer_for(restored, cfg)
    sch2 = scheduler_for(opt2, cfg)
    opt2.load_state_dict(ckpt['optimizer']); sch2.load_state_dict(ckpt['scheduler'])
    restore_rng(ckpt['rng'], sampler)
    torch.testing.assert_close(torch.rand(3), expected_random[0], rtol=0, atol=0)
    assert np.array_equal(np.random.rand(3), expected_random[1])
    assert sampler.rng.random() == expected_random[2]
    step(restored, opt2, sch2)
    for p, q in zip(tiny.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert scheduler.get_last_lr() == sch2.get_last_lr()


@pytest.mark.parametrize('key,value', [('dapt', True), ('max_epochs', 30), ('validation_suite', 'dev_random'),
                                      ('beam_width', 5), ('early_stopping', False),
                                      ('early_stopping_patience', 4), ('eval_forward_batch_size', 128)])
def test_invalid_formal_config(cfg, key, value):
    bad = dict(cfg, **{key: value})
    with pytest.raises(ValueError):
        validate_config(bad)
