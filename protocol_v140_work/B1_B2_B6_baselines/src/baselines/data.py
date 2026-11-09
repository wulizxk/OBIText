"""Purpose-restricted reads: no Test text/answers during fit, Dev or smoke."""
from collections import Counter
from itertools import islice
from pathlib import Path

from .io import read, rows, sha
from benchmark_protocol_v140 import EVALUATION_SUITES, benchmark_relative_path
from .protocol import validate_data_protocol, FORMAL_TEST_SUITES, EXPECTED_LENGTH_COUNTS

MANIFEST_SHA = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'
BOS, EOS = -1, -2


def has_visible_main_context(tokens):
    for token in tokens:
        if token == '<ANNOTATION_BOUNDARY>':
            break
        if token not in {'[LACUNA_1]', '[LACUNA_N]', '[MASK]'}:
            return True
    return False


class DataGate:
    def __init__(self, root, allow_test=False):
        self.root = Path(root).resolve()
        self.allow_test = allow_test
        self.allow_test_gold = False
        self.accessed = {}
        if sha(self.root / 'DATA_SHA256SUMS.json') != MANIFEST_SHA:
            raise ValueError('Expected the fixed v1.5.0 manifest, not another data revision')
        self.manifest = read(self.root / 'DATA_SHA256SUMS.json')
        for name in ('benchmark/protocol.json', 'input_units.json', 'target_vocab.json', 'group_assignments.json'):
            self.path(name)
        self.protocol = validate_data_protocol(self.root)
        self.inputs = read(self.root / 'input_units.json')['tokens']
        self.targets = read(self.root / 'target_vocab.json')['tokens']
        self.input_ids = {x: i for i, x in enumerate(self.inputs)}
        self.target_ids = {x: i for i, x in enumerate(self.targets)}
        self.target_input_ids = [self.input_ids[x] for x in self.targets]
        if len(self.inputs) != len(self.input_ids) or len(self.targets) != len(self.target_ids):
            raise ValueError('Duplicate vocabulary units')
        self.groups = {x['group_id']: x['split'] for x in read(self.root / 'group_assignments.json')}

    def path(self, name):
        permitted = {'benchmark/protocol.json', 'input_units.json', 'target_vocab.json',
                     'group_assignments.json', 'splits/train.jsonl',
                     'benchmark/dev_full_public.jsonl', 'benchmark/dev_full_gold.jsonl'}
        for suite in FORMAL_TEST_SUITES:
            for part in ('public', 'gold'):
                if name != benchmark_relative_path(suite, part):
                    continue
                if not self.allow_test or (part == 'gold' and not self.allow_test_gold):
                    raise PermissionError(f'File not permitted in this stage: {name}')
                path = self.root / name
                expected = EVALUATION_SUITES[suite]['hashes'][part + '.jsonl']
                if sha(path) != expected:
                    raise ValueError('Data hash mismatch: ' + name)
                self.accessed[name] = expected
                return path
        if name not in permitted or name not in self.manifest['files']:
            raise PermissionError(f'File not permitted in this stage: {name}')
        p = self.root / name
        expected = self.manifest['files'][name]
        if name not in self.accessed:
            if sha(p) != expected:
                raise ValueError('Data hash mismatch: ' + name)
            self.accessed[name] = expected
        return p

    def train(self, limit=None):
        compact, ids = [], set()
        stream = rows(self.path('splits/train.jsonl'))
        for row in (islice(stream, limit) if limit else stream):
            if row['split'] != 'train' or self.groups[row['group_id']] != 'train' or row['sequence_id'] in ids:
                raise ValueError('Non-Train/duplicate row in fit input')
            ids.add(row['sequence_id'])
            tokens, units, mask = row['model_tokens'], row['units'], row['targetable_mask']
            visible = row.get('visible_context_mask')
            if len(tokens) != len(units) or len(tokens) != len(mask) or len(tokens) != len(visible):
                raise ValueError('Atomic-unit alignment mismatch')
            eligible = {}
            for i, (token, unit, flag) in enumerate(zip(tokens, units, mask)):
                if unit['token'] != token or unit['targetable'] != flag or type(flag) is not bool:
                    raise ValueError('Targetability mismatch')
                if flag and (token not in self.target_ids or unit.get('protection_reasons')
                             or unit.get('role') != 'main' or token.startswith('<LIG-')
                             or token in ('[LACUNA_1]', '[LACUNA_N]', '<ANNOTATION_BOUNDARY>')):
                    raise ValueError('Protected unit used as target')
                if flag and '<ANNOTATION_BOUNDARY>' in tokens[:i]:
                    raise ValueError('Target after 兆记 boundary')
            for length in range(1, 6):
                expected = [i for i in range(len(tokens) - length + 1)
                            if all(mask[i:i+length]) and any(visible[:i] + visible[i+length:])]
                actual = row['eligible_starts'][str(length)]
                if actual != expected:
                    raise ValueError('Frozen eligible starts mismatch')
                eligible[str(length)] = actual
            compact.append({'sequence_id': row['sequence_id'], 'group_id': row['group_id'],
                            'tokens': [self.input_ids[t] for t in tokens],
                            'targets': [self.target_ids[t] if flag else -1 for t, flag in zip(tokens, mask)],
                            'eligible': eligible})
        if limit is None and len(compact) != 66511:
            raise ValueError('Expected all 66,511 Train rows')
        return compact

    def question(self, row):
        if 'gold_units' in row or 'model_tokens' in row:
            raise ValueError('Gold/full source must never enter prediction API')
        length = row['gap_length']
        if type(length) is not int or length not in range(1, 6):
            raise ValueError('Gap length must be 1..5')
        pre, suf = row['prefix_units'], row['suffix_units']
        if row.get('start_unit', len(pre)) != len(pre):
            raise ValueError('Gap start must equal the atomic-unit prefix length')
        if not has_visible_main_context(pre + suf):
            raise ValueError('At least one visible main-text context unit is required')
        if '<ANNOTATION_BOUNDARY>' in pre or any(t == '[MASK]' for t in pre + suf):
            raise ValueError('Invalid prediction position/context')
        if row.get('mask_units', ['[MASK]'] * length) != ['[MASK]'] * length:
            raise ValueError('Mask-length mismatch')
        q = {k: row.get(k) for k in ('example_id', 'source_id', 'group_id', 'boundary_side', 'start_unit')}
        q.update(gap_length=length, prefix_units=pre, suffix_units=suf,
                 prefix_ids=[self.input_ids[t] for t in pre], suffix_ids=[self.input_ids[t] for t in suf])
        return q

    def public(self, suite):
        if suite != 'dev_full' and suite not in FORMAL_TEST_SUITES:
            raise ValueError('Unknown validation or manuscript evaluation suite')
        raw = list(rows(self.path(benchmark_relative_path(suite, 'public'))))
        required_split = 'validation' if suite == 'dev_full' else 'test'
        expected_row_suite = suite if suite == 'dev_full' else EVALUATION_SUITES[suite]['row_suite']
        ids = set()
        for r in raw:
            if r['example_id'] in ids or r['suite'] != expected_row_suite or self.groups[r['group_id']] != required_split:
                raise ValueError('Public query identity/split mismatch')
            ids.add(r['example_id'])
        expected = self.protocol['validation_length_counts'] if suite == 'dev_full' else EXPECTED_LENGTH_COUNTS[suite]
        if dict(Counter(str(r['gap_length']) for r in raw)) != expected:
            raise ValueError('Frozen suite count/length quotas mismatch')
        return [self.question(r) for r in raw]

    def gold(self, suite, questions):
        gold = list(rows(self.path(benchmark_relative_path(suite, 'gold'))))
        if len(gold) != len(questions) or len({r['example_id'] for r in gold}) != len(gold):
            raise ValueError('Gold count or IDs mismatch')
        by_id = {r['example_id']: r['gold_units'] for r in gold}
        for q in questions:
            answer = by_id[q['example_id']]
            if len(answer) != q['gap_length'] or any(t not in self.target_ids for t in answer):
                raise ValueError('Gold target units mismatch')
        return by_id
