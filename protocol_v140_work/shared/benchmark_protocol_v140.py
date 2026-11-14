"""Frozen training inputs and manuscript evaluation suites."""

import hashlib
import json
from pathlib import Path


DATA_VERSION = 'v1.5.0'
INPUT_UNIT_COUNT = 4546
TARGET_UNIT_COUNT = 4116
DEV_FULL_QUESTION_COUNT = 15024
DEV_RANDOM_QUESTION_COUNT = 1500
SOURCE_TASK_SEQUENCE_COUNT = 78875
MASK_POLICY = ('targetable main-text units only; at least one visible main-text context unit '
               'remains; damage, ligatures and annotations protected as targets')
REPO = Path(__file__).resolve().parents[2]
DATA_MANIFEST_SHA256 = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'
EVALUATION_SUITES = json.loads((REPO / 'provenance/evaluation_suites.json').read_text(encoding='utf-8'))
PUBLIC_DATA = json.loads((REPO / 'provenance/public_data.json').read_text(encoding='utf-8'))
FORMAL_TEST_SUITES = tuple(EVALUATION_SUITES)
EXPECTED_LENGTH_COUNTS = {suite: spec['lengths'] for suite, spec in EVALUATION_SUITES.items()}
SUITE_QUESTION_COUNTS = {suite: sum(lengths.values()) for suite, lengths in EXPECTED_LENGTH_COUNTS.items()}
EVALUATION_QUESTION_INSTANCES = sum(SUITE_QUESTION_COUNTS.values())


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def benchmark_relative_path(suite, part):
    if part not in ('public', 'gold'):
        raise ValueError('Unknown question file kind')
    if suite == 'dev_full':
        return f'benchmark/dev_full_{part}.jsonl'
    if suite not in EVALUATION_SUITES:
        raise ValueError('Unknown evaluation suite: ' + suite)
    return f"evaluation/{EVALUATION_SUITES[suite]['folder']}/{part}.jsonl"


def data_manifest(root):
    path = Path(root) / 'DATA_SHA256SUMS.json'
    actual = sha_file(path)
    if actual not in (DATA_MANIFEST_SHA256, PUBLIC_DATA['data_manifest_sha256']):
        raise ValueError('Unknown OBIText v1.5.0 data manifest')
    return _read(path), actual


def evaluation_hashes(root, suite):
    _, actual = data_manifest(root)
    if actual == PUBLIC_DATA['data_manifest_sha256']:
        return PUBLIC_DATA['suites'][suite]['hashes']
    return EVALUATION_SUITES[suite]['hashes']


def benchmark_path(root, suite, part):
    path = Path(root) / benchmark_relative_path(suite, part)
    if suite in EVALUATION_SUITES:
        expected = evaluation_hashes(root, suite)[part + '.jsonl']
        if sha_file(path) != expected:
            raise ValueError('Frozen evaluation file checksum mismatch: ' + str(path))
    return path


def identify_evaluation_suite(public_sha, gold_sha, root=None):
    pair = {'public.jsonl': public_sha, 'gold.jsonl': gold_sha}
    for suite, spec in EVALUATION_SUITES.items():
        expected = evaluation_hashes(root, suite) if root is not None else spec['hashes']
        if expected == pair:
            return suite
    raise ValueError('Questions and answers do not match a manuscript evaluation suite')


def verify_training_data(root):
    """Verify all inputs used by the models, independent of evaluation question files."""
    root = Path(root)
    manifest, _ = data_manifest(root)
    required = {
        'benchmark/protocol.json', 'benchmark/dev_full_public.jsonl', 'benchmark/dev_full_gold.jsonl',
        'input_units.json', 'target_vocab.json', 'group_assignments.json',
        'splits/train.jsonl', 'splits/validation.jsonl', 'splits/test.jsonl',
    }
    if manifest.get('edition') == 'public':
        required.add('tokenizer_sources.json')
    required.update(name for name in manifest['files'] if name.startswith(('tokenizers/', 'pools/train_span')))
    for name in sorted(required):
        if sha_file(root / name) != manifest['files'][name]:
            raise ValueError('Data checksum mismatch: ' + name)
    if manifest.get('edition') == 'public':
        sources = _read(root / 'tokenizer_sources.json')
        if sha_file(root / 'tokenizer_sources.json') != manifest['files']['tokenizer_sources.json']:
            raise ValueError('Public tokenizer source inventory checksum mismatch')
        for asset, spec in sources.items():
            for filename, expected in spec['rebuilt_files'].items():
                path = root / 'tokenizers' / asset / filename
                if not path.is_file():
                    raise FileNotFoundError('Run scripts/prepare_tokenizers.py first: ' + str(path))
                if sha_file(path) != expected:
                    raise ValueError('Tokenizer checksum mismatch: ' + str(path))
    validate_data_protocol(root)
    # Both editions retain identical ordered model inputs and training samples.
    return DATA_MANIFEST_SHA256


def evaluation_config(saved):
    """Translate the evaluation field of an exactly identified original checkpoint configuration."""
    registry = _read(REPO / 'provenance/training_source_fingerprints.json')
    code = saved.get('experiment', '').removesuffix('-A')
    payload = json.dumps(saved, ensure_ascii=False, sort_keys=True).encode()
    if hashlib.sha256(payload).hexdigest() == registry.get(code, {}).get('training_config_digest'):
        return {**saved, 'test_suites': list(FORMAL_TEST_SUITES)}
    return saved


def compatible_source(project, recorded, current, baseline=False):
    if recorded == current:
        return True
    if baseline:
        entries = [_read(REPO / 'provenance/baseline_source_fingerprints.json')]
    else:
        entries = [entry for entry in _read(REPO / 'provenance/training_source_fingerprints.json').values()
                   if entry['project'] == Path(project).name]
    return any(recorded in (entry['training_source_sha256'], entry.get('prior_release_source_sha256'),
                           *entry.get('compatible_source_sha256', []))
               and current == entry.get('current_source_sha256')
               for entry in entries)


EXPECTED_MODE_COUNTS = {
    suite: {
        mode: sum(count for length, count in lengths.items() if (
            mode == ('1' if length == '1' else '2' if length == '2' else '3-5')
        ))
        for mode in {'1' if length == '1' else '2' if length == '2' else '3-5' for length in lengths}
    }
    for suite, lengths in EXPECTED_LENGTH_COUNTS.items()
}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_data_protocol(root):
    """Validate semantic metadata after callers have verified every file hash."""
    root = Path(root)
    manifest = _read(root / 'DATA_SHA256SUMS.json')
    protocol = _read(root / 'benchmark/protocol.json')
    if manifest.get('version') != DATA_VERSION or manifest.get('status') != 'FROZEN_DATA_READY':
        raise ValueError('Experiments require frozen v1.5.0 data')
    if protocol.get('version') != DATA_VERSION or protocol.get('parent_version') != 'v1.4.1':
        raise ValueError('Unexpected v1.5.0 protocol provenance')
    validation_suites = {'dev_full': DEV_FULL_QUESTION_COUNT}
    if manifest.get('edition') != 'public':
        validation_suites['dev_random'] = DEV_RANDOM_QUESTION_COUNT
    if (protocol.get('unit_of_length') != 'one atomic source unit, not a Unicode code point or native subword'
            or protocol.get('validation_suite') != 'dev_full'
            or protocol.get('validation_question_count') != DEV_FULL_QUESTION_COUNT
            or protocol.get('validation_suites') != validation_suites):
        raise ValueError('Wrong atomic-unit validation contract')
    if (protocol.get('training_square_bracket_restorations_maskable') is not True
            or protocol.get('training_boundary_masks_enabled') is not True
            or protocol.get('full_sequence_masking') is not False
            or protocol.get('mask_policy') != MASK_POLICY
            or protocol.get('source_task_sequences') != SOURCE_TASK_SEQUENCE_COUNT):
        raise ValueError('Mask/protection policy differs from v1.5.0')
    if (protocol.get('input_unit_count') != INPUT_UNIT_COUNT
            or protocol.get('prediction_target_count') != TARGET_UNIT_COUNT
            or _read(root / 'input_units.json').get('size') != INPUT_UNIT_COUNT
            or _read(root / 'target_vocab.json').get('size') != TARGET_UNIT_COUNT):
        raise ValueError('Input or prediction vocabulary differs from v1.5.0')
    return protocol


def _question_counts(groups):
    if not isinstance(groups, dict):
        raise ValueError('Metric grouping must be a dictionary')
    return {str(key): value.get('questions') for key, value in groups.items()}


def validate_formal_suite_metrics(suites):
    """Reject incomplete, reordered, or quota-incompatible formal Test reports."""
    if not isinstance(suites, dict) or tuple(suites) != FORMAL_TEST_SUITES:
        raise ValueError('Formal summary requires the current manuscript suites in fixed order')
    for suite in FORMAL_TEST_SUITES:
        validate_formal_suite_metric(suite, suites[suite])
    return suites


def validate_formal_suite_metric(suite, metrics):
    if suite not in FORMAL_TEST_SUITES or not isinstance(metrics, dict):
        raise ValueError('Unknown or malformed formal Test suite metrics')
    count = SUITE_QUESTION_COUNTS[suite]
    if metrics.get('questions') != count:
        raise ValueError(f'{suite} must contain exactly {count:,} questions')
    if _question_counts(metrics.get('by_length')) != EXPECTED_LENGTH_COUNTS[suite]:
        raise ValueError(f'{suite} length counts differ from the frozen v1.5.0 protocol')
    if _question_counts(metrics.get('by_mode')) != EXPECTED_MODE_COUNTS[suite]:
        raise ValueError(f'{suite} mode counts differ from the frozen v1.5.0 protocol')
    expected_sides = ({'prefix': count} if suite == 'test_prefix'
                      else {'suffix': count} if suite == 'test_suffix' else {})
    if _question_counts(metrics.get('by_boundary_side')) != expected_sides:
        raise ValueError(f'{suite} boundary-side counts differ from the frozen v1.5.0 protocol')
    return metrics
