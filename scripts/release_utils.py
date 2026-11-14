"""Data identity checks shared by portable manuscript evaluation entrypoints."""
import gzip
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'
PUBLIC_DATA = json.loads((REPO / 'provenance/public_data.json').read_text(encoding='utf-8'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def checked_data(data, names):
    data = Path(data).resolve()
    path = data / 'DATA_SHA256SUMS.json'
    actual = sha(path)
    if actual not in (FROZEN_MANIFEST, PUBLIC_DATA['data_manifest_sha256']):
        raise ValueError('Unknown OBIText v1.5.0 data manifest')
    manifest = read(path)
    for name in names:
        if sha(data / name) != manifest['files'].get(name):
            raise ValueError('Frozen data file mismatch: ' + name)
    return manifest


def checked_questions(data, public_path, gold_path=None, manifest_path=None):
    checked_data(data, ['splits/test.jsonl', 'splits/train.jsonl', 'target_vocab.json'])
    if manifest_path:
        manifest_path = Path(manifest_path).resolve()
        manifest = read(manifest_path)
        actual = sha(Path(data) / 'DATA_SHA256SUMS.json')
        if (manifest['data_manifest_sha256'] != actual
                or (actual == PUBLIC_DATA['data_manifest_sha256']
                    and sha(manifest_path) != PUBLIC_DATA['evaluation_manifest_sha256'])):
            raise ValueError('Questions were built against a different data edition')
        for path in [public_path] + ([gold_path] if gold_path else []):
            path = Path(path).resolve()
            key = str(path.relative_to(manifest_path.parent))
            if sha(path) != manifest['files'].get(key):
                raise ValueError('Question manifest/file mismatch: ' + key)
    public = rows(public_path)
    ids = [q['example_id'] for q in public]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty questions or duplicate example IDs')
    train = rows(Path(data) / 'splits/train.jsonl')
    train_ids = {r['sequence_id'] for r in train}
    train_groups = {r['group_id'] for r in train}
    test = {r['sequence_id']: r for r in rows(Path(data) / 'splits/test.jsonl')}
    gold = None
    if gold_path:
        raw = rows(gold_path)
        gold = {r['example_id']: r['gold_units'] for r in raw}
        if len(gold) != len(raw) or set(gold) != set(ids):
            raise ValueError('Public/gold identity mismatch')
    for q in public:
        r = test[q['source_id']]
        s, length = q['start_unit'], q['gap_length']
        if 'gold_units' in q or 'model_tokens' in q:
            raise ValueError('Gold/full source text found in public prediction input')
        if (q['source_id'] in train_ids or q['group_id'] in train_groups
                or q['group_id'] != r['group_id'] or r['split'] != 'test'):
            raise ValueError('Invalid source/group split identity')
        if (type(length) is not int or length not in range(1, 6)
                or type(s) is not int or s not in r['eligible_starts'][str(length)]):
            raise ValueError('Question gap is not a frozen legal test target')
        if (q['prefix_units'] != r['model_tokens'][:s]
                or q['suffix_units'] != r['model_tokens'][s + length:]):
            raise ValueError('Question context differs from its frozen source')
        if gold is not None and gold[q['example_id']] != r['model_tokens'][s:s + length]:
            raise ValueError('Gold differs from its frozen source target')
    return public, gold, test
