import copy
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'src'), str(root.parent / 'shared')]
import numpy as np

from baselines.data import DataGate, BOS, EOS
from baselines.evaluation import predict_file, score_file, validate_prediction
from baselines.io import ROOT, read, write, read_gzip, write_gzip, reserve, sha, verify_files
from baselines.metrics import aggregate, score_prediction
from baselines.models import Unigram, Ngram, Retriever, features, fit, local_matches, rank, restore, validate_config
from baselines.runner import artifact, check_isolation, choose_smoke_questions, config_for, do_test
from baselines.protocol import FORMAL_TEST_SUITES
from benchmark_protocol_v140 import EVALUATION_SUITES, benchmark_relative_path


def doc(tokens, name='train0', group='train_group'):
    targets = [t if t < 12 else -1 for t in tokens]
    return {'sequence_id': name, 'group_id': group, 'tokens': tokens, 'targets': targets,
            'eligible': {str(n): [i for i in range(len(tokens) - n + 1)
                                 if n < len(tokens) and all(t >= 0 for t in targets[i:i+n])]
                         for n in range(1, 6)}}


def question(length=1, pre=None, suf=None):
    return {'example_id': 'q1', 'source_id': 'dev_source', 'group_id': 'dev_group',
            'gap_length': length, 'prefix_ids': [0] if pre is None else pre,
            'suffix_ids': [1] if suf is None else suf, 'boundary_side': None}


def small_gate():
    return SimpleNamespace(targets=[str(i) for i in range(12)], target_input_ids=list(range(12)),
                           input_ids={'[LACUNA_N]': 12})


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.documents = [doc([0, 2, 1], f't{i}') for i in range(15)]
        self.documents += [doc(list(range(12)), 'all_targets'), doc([0, 12, 3, 1], 'damage')]

    def model(self, name):
        config = config_for(name)
        if name == 'B6':
            config['max_df_ratio'] = 1.0
        return fit(config, small_gate(), self.documents)

    def test_configs(self):
        for name in ('B1', 'B2', 'B6'):
            validate_config(config_for(name))
        for name, key, value in (('B1', 'unigram_alpha', float('nan')),
                                 ('B2', 'order', 4), ('B2', 'forward_weight', 2),
                                 ('B6', 'exclude_query_group', False), ('B6', 'document_limit', 0)):
            cfg = config_for(name); cfg[key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_unigram_counts_exclude_protected(self):
        model = self.model('B1')
        self.assertEqual(sum(model.counts), sum(t >= 0 for r in self.documents for t in r['targets']))
        self.assertEqual(len(model.counts), 12)
        self.assertTrue(np.all(model.probs > 0))
        self.assertAlmostEqual(model.probs.sum(), 1)

    def test_stable_ties(self):
        self.assertEqual(rank([.2, .3, .3, .2]), [1, 2, 0, 3])
        model = Unigram(config_for('B1'), list(range(12)), [1] * 12)
        self.assertEqual(model.predict(question())['single_ranking'], list(range(12)))
        self.assertEqual(model.predict(question(2))['paths'], [[0, i] for i in range(5)])

    def test_witten_bell_formula_and_normalization(self):
        tables = {'forward': {(0,): {2: 6, 3: 2}}, 'backward': {}}
        model = Ngram(config_for('B2'), list(range(12)), [1] * 12, tables, 12)
        actual = model.distribution([0], 'forward')
        expected = np.full(12, 2 / 10 / 12)
        expected[2] += 6 / 10; expected[3] += 2 / 10
        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(actual.sum(), 1)
        np.testing.assert_allclose(model.distribution([11], 'forward'), model.probs)

    def test_context_boundaries_and_indefinite_damage(self):
        model = self.model('B2')
        self.assertEqual(model.context([], BOS), (BOS,))
        self.assertEqual(model.context([], EOS), (EOS,))
        self.assertEqual(model.context([9, 8, 12, 3], BOS), (12, 3))
        self.assertEqual(model.context([8, 9, 0, 1, 2], BOS), (9, 0, 1, 2))

    def test_both_directions_and_normalization(self):
        model = self.model('B2')
        self.assertEqual(model.predict(question())['paths'][0], [2])
        for side, history in (('forward', []), ('backward', []), ('forward', [0]), ('backward', [1])):
            p = model.distribution(history, side)
            self.assertAlmostEqual(p.sum(), 1)
            self.assertTrue(np.all(p > 0))
        pre = question(pre=[])
        suf = question(suf=[])
        validate_prediction(pre, model.predict(pre), 12)
        validate_prediction(suf, model.predict(suf), 12)

    def test_bidirectional_candidate_scores(self):
        model = self.model('B2')
        for n in range(2, 6):
            q = question(n)
            p = validate_prediction(q, model.predict(q), 12)
            self.assertLessEqual(p['proposal_count'], 20)
            for path, score in zip(p['paths'], p['scores']):
                self.assertAlmostEqual(score, model.joint_score(q['prefix_ids'], q['suffix_ids'], path))

    def test_no_bridge_feature(self):
        pre, suf = [0, 1], [2, 3]
        query = set(features(pre)) | set(features(suf))
        self.assertNotIn((1, 2), query)
        self.assertIn((0, 1), query)

    def test_alignment_and_boundary(self):
        self.assertEqual(local_matches([0], [1], [0, 2, 1], 1, 1, 8), (1, 1, 0))
        self.assertEqual(local_matches([], [1], [2, 1], 0, 1, 8), (0, 1, 1))
        self.assertEqual(local_matches([0], [], [0, 2], 1, 1, 8), (1, 0, 1))

    def test_retriever_evidence_and_group_exclusion(self):
        model = self.model('B6')
        p = model.predict(question())
        self.assertEqual(p['paths'][0], [2])
        self.assertFalse(p['retrieval']['fallback_top1'])
        self.assertTrue(p['retrieval']['evidence'])
        q = question(); q['group_id'] = 'train_group'
        result = validate_prediction(q, model.predict(q), 12)
        self.assertTrue(result['retrieval']['fallback_top1'])
        self.assertEqual(result['single_ranking'], model.ranking)

    def test_retrieved_targets_are_eligible(self):
        model = self.model('B6')
        by_id = {r['sequence_id']: r for r in self.documents}
        for n in range(1, 6):
            prediction = validate_prediction(question(n), model.predict(question(n)), 12)
            for candidate in prediction['retrieval']['evidence']:
                for evidence in candidate['evidence']:
                    row = by_id[evidence['sequence_id']]
                    start = evidence['start_unit']
                    self.assertIn(start, row['eligible'][str(n)])
                    self.assertEqual(candidate['path'], row['targets'][start:start+n])

    def test_no_retrieval_keeps_all_questions_and_full_ranking(self):
        model = self.model('B6')
        for n in range(1, 6):
            q = question(n, [1000], [1001])
            result = validate_prediction(q, model.predict(q), 12)
            self.assertTrue(result['retrieval']['fallback_top1'])
            if n == 1:
                self.assertEqual(sorted(result['single_ranking']), list(range(12)))

    def test_serialization_and_repeatability(self):
        for name in ('B1', 'B2', 'B6'):
            model = self.model(name)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / 'model.gz'
                write_gzip(p, model.payload())
                restored = restore(read_gzip(p))
                for n in range(1, 6):
                    q = question(n)
                    self.assertEqual(model.predict(q), restored.predict(q))
                    self.assertEqual(model.predict(q), model.predict(q))


class DataAndLifecycleTests(unittest.TestCase):
    def fake_gate(self):
        gate = DataGate.__new__(DataGate)
        gate.allow_test = False; gate.allow_test_gold = False; gate.accessed = {}
        gate.manifest = {'files': {}}
        gate.input_ids = {'甲': 0, '[UNK-U111]': 1, '[LACUNA_1]': 2, '<ANNOTATION_BOUNDARY>': 3, '<LIG-U1>': 4}
        gate.target_ids = {'甲': 0, '[UNK-U111]': 1}
        gate.groups = {'train_group': 'train'}
        return gate

    def test_test_access_closed_by_default(self):
        gate = self.fake_gate()
        for name in (benchmark_relative_path('test_expert', 'public'), benchmark_relative_path('test_expert', 'gold'), 'splits/test.jsonl'):
            with self.assertRaises(PermissionError):
                gate.path(name)
        gate.allow_test = True
        with self.assertRaises(PermissionError):
            gate.path(benchmark_relative_path('test_expert', 'gold'))

    def test_atomic_unit_question_and_protected_prefix(self):
        gate = self.fake_gate()
        raw = {'example_id': 'q', 'source_id': 's', 'group_id': 'g', 'gap_length': 1,
               'prefix_units': ['[UNK-U111]'], 'suffix_units': ['[LACUNA_1]'], 'mask_units': ['[MASK]']}
        q = gate.question(raw)
        self.assertEqual(q['prefix_ids'], [1])
        self.assertEqual(q['suffix_ids'], [2])
        for changes in ({'gold_units': ['甲']}, {'model_tokens': ['甲']}, {'prefix_units': ['<ANNOTATION_BOUNDARY>']},
                        {'prefix_units': [], 'suffix_units': []}, {'prefix_units': [], 'suffix_units': ['<ANNOTATION_BOUNDARY>']}, {'mask_units': ['[MASK]', '[MASK]']}):
            with self.assertRaises(ValueError):
                gate.question({**raw, **changes})

    def test_frozen_public_suite_names_and_split_isolation(self):
        gate = self.fake_gate()
        gate.groups.update(test_group='test', dev_group='validation')
        gate.protocol = {'validation_length_counts': {'1': 1}}
        counts = {name: {'1': 1} for name in FORMAL_TEST_SUITES}
        for suite in ('dev_full', *FORMAL_TEST_SUITES):
            with self.subTest(suite=suite):
                row = {'example_id': 'q', 'source_id': 's',
                       'group_id': 'dev_group' if suite == 'dev_full' else 'test_group',
                       'suite': suite if suite == 'dev_full' else EVALUATION_SUITES[suite]['row_suite'],
                       'gap_length': 1, 'start_unit': 1,
                       'prefix_units': ['甲'], 'suffix_units': [], 'mask_units': ['[MASK]']}
                with patch.object(gate, 'path', return_value='unused'), \
                     patch('baselines.data.EXPECTED_LENGTH_COUNTS', counts), \
                     patch('baselines.data.rows', side_effect=lambda _: iter([row])):
                    self.assertEqual(len(gate.public(suite)), 1)
                    for changes in ({'suite': 'wrong_suite'}, {'group_id': 'train_group'}):
                        with patch('baselines.data.rows', return_value=iter([{**row, **changes}])):
                            with self.assertRaises(ValueError):
                                gate.public(suite)
                    with patch('baselines.data.rows', return_value=iter([row, row])):
                        with self.assertRaises(ValueError):
                            gate.public(suite)

    def test_expert_unk_and_boundary_are_train_targets(self):
        gate = self.fake_gate()
        row = {'split': 'train', 'sequence_id': 'a', 'group_id': 'train_group',
               'model_tokens': ['甲', '[LACUNA_1]', '[UNK-U111]'], 'targetable_mask': [True, False, True],
               'visible_context_mask': [True, False, True],
               'units': [{'token': '甲', 'targetable': True, 'role': 'main', 'restored': True},
                         {'token': '[LACUNA_1]', 'targetable': False, 'role': 'main'},
                         {'token': '[UNK-U111]', 'targetable': True, 'role': 'main'}],
               'eligible_starts': {'1': [0, 2], '2': [], '3': [], '4': [], '5': []}}
        with patch.object(gate, 'path', return_value='unused'), patch('baselines.data.rows', return_value=iter([row])):
            parsed = gate.train(limit=1)[0]
        self.assertEqual(parsed['targets'], [0, -1, 1])
        self.assertEqual(parsed['eligible']['1'], [0, 2])
        bad = copy.deepcopy(row); bad['units'][0]['protection_reasons'] = ['not_a_target']
        with patch.object(gate, 'path', return_value='unused'), patch('baselines.data.rows', return_value=iter([bad])):
            with self.assertRaises(ValueError):
                gate.train(limit=1)

    def test_isolation_by_source_or_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'train_identities.json', {'s1': 'g1'})
            check_isolation([question()], root)
            for key, value in (('source_id', 's1'), ('group_id', 'g1')):
                q = question(); q[key] = value
                with self.assertRaises(ValueError):
                    check_isolation([q], root)

    def test_write_once_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'example.json'
            write(p, {'a': 1})
            verify_files(root, {'example.json': sha(p)})
            with self.assertRaises(ValueError):
                verify_files(root, {'example.json': 'wrong'})
            with self.assertRaises(FileExistsError):
                write(p, {'b': 2})
            with self.assertRaises(ValueError):
                reserve(root / 'data' / 'output', root / 'data')
            with self.assertRaises(FileExistsError):
                reserve(root, root / 'data')

    def test_formal_test_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(ValueError, 'confirm-final'):
            do_test(SimpleNamespace(confirm_final=False))

    def simulate_test_barrier(self, fail_prediction=False):
        # Synthetic control-flow diagnostic, NOT a real evaluation/selection.
        with tempfile.TemporaryDirectory() as tmp:
            root, phases = Path(tmp), []
            gate = SimpleNamespace(root=Path(tmp) / 'data', allow_test_gold=False, targets=['0'])
            gate.public = lambda suite: [question()]
            def get_gold(suite, questions):
                self.assertTrue(gate.allow_test_gold)
                self.assertEqual(sum(stage == 'predict' for stage in phases), len(FORMAL_TEST_SUITES))
                phases.append('gold')
                return {'q1': ['0']}
            gate.gold = get_gold
            def prediction(model, questions, path):
                self.assertFalse(gate.allow_test_gold)
                self.assertNotIn('gold', phases)
                phases.append('predict')
                if fail_prediction and len(phases) == 3:
                    raise RuntimeError('synthetic prediction failure')
                write(path, {'synthetic_unit_test_only': True})
                return {'questions': 1, 'sha256': 'synthetic'}
            selection = {'fit_path': str(root / 'fit'), 'dev_path': str(root / 'dev'),
                         'fit_completion_sha256': 'h', 'dev_completion_sha256': 'h',
                         'config_sha256': 'c', 'experiment': 'B1'}
            dev_meta = {'fit_completion_sha256': 'h', 'questions': 15024, 'experiment': 'B1'}
            def get_artifact(path, marker, kind):
                return (root / 'selection', selection) if kind == 'selection' else (root / 'dev', dev_meta)
            args = SimpleNamespace(confirm_final=True, data=gate.root, selection=root / 'selection', output=root / 'output')
            with patch('baselines.runner.DataGate', return_value=gate), \
                 patch('baselines.runner.artifact', side_effect=get_artifact), \
                 patch('baselines.runner.load_fit', return_value=(root / 'fit', {'config_sha256': 'c', 'experiment': 'B1'}, object())), \
                 patch('baselines.runner.check_isolation'), patch('baselines.runner.sha', return_value='h'), \
                 patch('baselines.runner.predict_file', side_effect=prediction), \
                 patch('baselines.runner.score_file', return_value={}), \
                 patch('baselines.runner.validate_formal_suite_metrics'), patch('baselines.runner.common', return_value={}), \
                 patch('builtins.print'):
                if fail_prediction:
                    with self.assertRaisesRegex(RuntimeError, 'synthetic prediction failure'):
                        do_test(args)
                    self.assertNotIn('gold', phases)
                    self.assertFalse(gate.allow_test_gold)
                    self.assertFalse((args.output / 'TEST_COMPLETE.json').exists())
                    self.assertFalse((args.output / 'PREDICTIONS_COMPLETE.json').exists())
                else:
                    do_test(args)
                    self.assertEqual(phases, ['predict'] * len(FORMAL_TEST_SUITES) + ['gold'] * len(FORMAL_TEST_SUITES))

    def test_all_manuscript_predictions_precede_any_gold(self):
        self.simulate_test_barrier()

    def test_failed_prediction_prevents_gold_opening(self):
        self.simulate_test_barrier(fail_prediction=True)

    def test_smoke_report_cannot_be_used_as_formal_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'FIT_COMPLETE.json', {'status': 'PASS', 'diagnostic_only': True, 'kind': 'fit'})
            with self.assertRaises(ValueError):
                artifact(root, 'FIT_COMPLETE.json', 'fit')

    def test_prediction_rejects_nan_and_incomplete_rank(self):
        model = Unigram(config_for('B1'), list(range(12)), [1] * 12)
        for mutation in ('nan', 'rank', 'duplicate', 'length'):
            p = model.predict(question())
            p = copy.deepcopy(p)
            if mutation == 'nan': p['scores'][0] = float('nan')
            if mutation == 'rank': p['single_ranking'] = [0, 1]
            if mutation == 'duplicate': p['paths'][1] = p['paths'][0]
            if mutation == 'length': p['paths'][0] = [0, 1]
            with self.assertRaises(ValueError):
                validate_prediction(question(), p, 12)

    def test_metric_parity_and_gold_does_not_change_predictions(self):
        model = Unigram(config_for('B1'), list(range(12)), [1] * 12)
        targets = [str(i) for i in range(12)]
        q = question(); prediction = model.predict(q)
        a = score_prediction(q, prediction, ['0'], targets)
        b = score_prediction(q, prediction, ['11'], targets)
        self.assertEqual(a['em1'], 1)
        self.assertAlmostEqual(b['reciprocal_rank'], 1/12)
        self.assertEqual(prediction, model.predict(q))
        q2 = question(2)
        value = score_prediction(q2, model.predict(q2), ['0', '1'], targets)
        self.assertEqual(value['hit5'], 1)
        self.assertIsNone(value['reciprocal_rank'])
        result = aggregate([a, value])
        self.assertAlmostEqual(result['pooled_not_macro']['position_accuracy'], 2/3)
        self.assertIsNone(result['macro5_em1'])

    def test_prediction_file_then_scoring(self):
        model = Unigram(config_for('B1'), list(range(12)), [1] * 12)
        q = question()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'pred.jsonl.gz'
            result = predict_file(model, [q], p, 0)
            self.assertEqual(result['questions'], 1)
            with gzip.open(p, 'rt') as stream:
                raw = stream.read()
            self.assertNotIn('gold', raw)
            out = score_file([q], {'q1': ['11']}, [str(i) for i in range(12)], p, root / 'scored.gz')
            self.assertAlmostEqual(out['by_length']['1']['mrr_full_vocab'], 1/12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
