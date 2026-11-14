"""CPU-only formal Test acceptance: frozen IDs, prediction bytes and metric replay.

No model loading/inference. This is an artifact-integrity gate, not a way to
recover full-vocabulary logits or to choose a checkpoint using Test results.
"""
import math
from pathlib import Path
from benchmark_protocol_v140 import (
    FORMAL_TEST_SUITES, SUITE_QUESTION_COUNTS, benchmark_path, compatible_source, data_manifest,
    validate_formal_suite_metric,
)

from .io import digest, read, rows, sha, source_fingerprint
from .metrics import aggregate
from .protocol import DATA_MANIFEST_SHA256
from .run_state import validate_completion_record, validate_training_batch
from .runtime import require_evaluation_report

ROOT = Path(__file__).resolve().parents[2]
SUITES = FORMAL_TEST_SUITES
ARTIFACT_SCHEMA = 2


def validate_report_header(report):
    env = require_evaluation_report(report)
    identity = report['identity']
    cfg = validate_completion_record(identity, report.get('training_completion'))
    if (env.get('hardware', {}).get('type') != 'cuda' or env.get('effective_precision') != cfg['precision']
            or type(report.get('seed')) is not int or report['seed'] not in (42, 3407, 2026)):
        raise ValueError('Formal Test artifacts require the expected seed and CUDA BF16 environment')
    completion = report['training_completion']
    if (report.get('completion_sha256') != digest(completion)
            or type(report.get('epoch')) is not int or report['epoch'] != completion['best']['epoch']
            or report.get('checkpoint_sha256') != completion['best']['checkpoint_sha256']):
        raise ValueError('Evaluation is not bound to its selected best/completion')
    batch = validate_training_batch(identity, report.get('training_batch_completion'))
    if (report.get('training_batch_sha256') != digest(batch)
            or batch['completions'][str(report['seed'])] != completion):
        raise ValueError('Test training-batch evidence differs from the selected completion')
    if (type(report.get('test_artifact_schema_version')) is not int
            or report['test_artifact_schema_version'] != ARTIFACT_SCHEMA
            or tuple(report.get('suites', {})) != SUITES
            or tuple(report.get('prediction_artifacts', {})) != SUITES):
        raise ValueError('Formal Test requires all manuscript evaluation prediction artifacts')
    return env


def load_test_contract(data):
    """Verify relevant frozen bytes directly, not declarations from a metrics file."""
    data = Path(data)
    manifest, _ = data_manifest(data)
    if sha(data / 'target_vocab.json') != manifest['files']['target_vocab.json']:
        raise ValueError('Frozen target vocabulary checksum mismatch')
    targets = read(data / 'target_vocab.json')['tokens']
    if len(targets) != 4116 or len(set(targets)) != 4116:
        raise ValueError('Unexpected frozen character target vocabulary')
    contract = {'data_sha256': DATA_MANIFEST_SHA256, 'target_units': targets, 'suites': {}}
    for suite in SUITES:
        public_path = benchmark_path(data, suite, 'public')
        gold_path = benchmark_path(data, suite, 'gold')
        public = rows(public_path)
        answers = rows(gold_path)
        count = SUITE_QUESTION_COUNTS[suite]
        gold = {row['example_id']: row['gold_units'] for row in answers}
        ids = [q['example_id'] for q in public]
        if len(ids) != count or len(set(ids)) != count or len(answers) != count or set(gold) != set(ids):
            raise ValueError('Unexpected frozen question IDs/counts: ' + suite)
        contract['suites'][suite] = {'public': public, 'gold': gold,
            'public_sha256': sha(public_path),
            'gold_sha256': sha(gold_path)}
    return contract


def prediction_manifest(output, data):
    """Bind predictions after all generation is finished; the validator does acceptance."""
    output, data = Path(output), Path(data)
    return {suite: {'file': suite + '_predictions.jsonl', 'sha256': sha(output / (suite + '_predictions.jsonl')),
                    'questions': len(rows(output / (suite + '_predictions.jsonl'))),
                    'public_sha256': sha(benchmark_path(data, suite, 'public')),
                    'gold_sha256': sha(benchmark_path(data, suite, 'gold'))}
            for suite in SUITES}


def _same(actual, expected, location):
    """Strict schema/types and finite values; bool must not masquerade as a number."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Missing/extra metric fields: ' + location)
        for key, value in expected.items():
            _same(actual[key], value, location + '/' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Invalid list schema: ' + location)
        for index, value in enumerate(expected):
            _same(actual[index], value, location + '/' + str(index))
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError('Invalid integer count/rank: ' + location)
    elif type(expected) is float:
        if (type(actual) not in (float, int) or not math.isfinite(actual)
                or (0 <= expected <= 1 and not 0 <= actual <= 1)
                or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Invalid or inconsistent metric: ' + location)
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError('Invalid metric value/type: ' + location)


def validate_prediction(row, question, gold, target_index, cfg):
    required = {'example_id', 'source_id', 'group_id', 'gap_length', 'boundary_side', 'gold_units',
                'candidates', 'candidate_scores', 'score_kind', 'em1', 'hit5', 'hit10',
                'correct_positions', 'gold_rank_full_vocabulary', 'reciprocal_rank'}
    if not isinstance(row, dict) or not required <= set(row):
        raise ValueError('Prediction row lacks required provenance/metric fields')
    length = question['gap_length']
    for key in ('example_id', 'source_id', 'group_id', 'gap_length'):
        _same(row.get(key), question[key], key)
    _same(row.get('boundary_side'), question.get('boundary_side'), 'boundary_side')
    if row.get('gold_units') != gold or len(gold) != length or any(t not in target_index for t in gold):
        raise ValueError('Prediction gold differs from the frozen answer')
    candidates, scores = row.get('candidates'), row.get('candidate_scores')
    count = cfg['single_mask_return_sequences'] if length == 1 else cfg['return_sequences']
    if (not isinstance(candidates, list) or len(candidates) != count
            or any(not isinstance(p, list) or len(p) != length
                   or any(not isinstance(t, str) or t not in target_index for t in p) for p in candidates)
            or len({tuple(p) for p in candidates}) != count):
        raise ValueError('Wrong candidate count, length, duplicate or non-character target')
    if (not isinstance(scores, list) or len(scores) != count
            or any(type(s) not in (float, int) or not math.isfinite(s) or s > 1e-7 for s in scores)):
        raise ValueError('Invalid finite candidate log-probability scores')
    if math.fsum(math.exp(s) for s in scores) > 1.0 + 1e-6:
        raise ValueError('Returned candidate probability mass exceeds one')
    ordered = sorted(range(count), key=lambda i: (-scores[i], tuple(target_index[t] for t in candidates[i])))
    if ordered != list(range(count)):
        raise ValueError('Candidate order violates score/canonical tie-break')
    single = length == 1
    kind = 'seq2seq_direct_character_log_probability' if single else 'seq2seq_autoregressive_character_plus_end_log_probability'
    if row.get('score_kind') != kind:
        raise ValueError('Prediction score kind differs from decoding protocol')
    candidate_rank = candidates.index(gold) + 1 if gold in candidates else None
    rank = row.get('gold_rank_full_vocabulary')
    if single:
        if (type(rank) is not int or not 1 <= rank <= len(target_index)
                or (candidate_rank is not None and rank != candidate_rank)
                or (candidate_rank is None and rank <= count)):
            raise ValueError('Full-vocabulary rank is inconsistent with the returned single-character candidates')
    elif rank is not None:
        raise ValueError('Multi-character predictions must not claim a single-character full rank')
    expected = {'em1': float(candidates[0] == gold),
        'hit5': float(candidate_rank is not None and candidate_rank <= 5),
        'correct_positions': sum(a == b for a, b in zip(candidates[0], gold)),
        'hit10': float(rank <= 10) if single else None,
        'reciprocal_rank': 1.0 / rank if single else None}
    for key, value in expected.items():
        _same(row.get(key), value, key)


def require_test_artifacts(folder, data=None, project=None):
    """Accept only complete, identity-bound, frozen-ID Test outputs. Never writes."""
    folder = Path(folder)
    project = ROOT if project is None else Path(project)
    data = project.parent / 'shared/data/v1.5.0' if data is None else Path(data)
    report_sha = sha(folder / 'metrics.json')
    report = read(folder / 'metrics.json')
    validate_report_header(report)
    if not compatible_source(project, report['identity']['code_sha256'], source_fingerprint(project)):
        raise ValueError('Test acceptance source differs from the evaluated code')
    contract = load_test_contract(data)
    if report['identity']['data_sha256'] != contract['data_sha256']:
        raise ValueError('Test report data identity differs from the frozen benchmark')
    target_index = {t: i for i, t in enumerate(contract['target_units'])}
    accepted = {}
    for suite in SUITES:
        spec = contract['suites'][suite]
        path = folder / (suite + '_predictions.jsonl')
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or symlinked prediction artifact: ' + str(path))
        predictions = rows(path)
        ids = [r.get('example_id') for r in predictions]
        if ids != [q['example_id'] for q in spec['public']] or len(set(ids)) != SUITE_QUESTION_COUNTS[suite]:
            raise ValueError('Prediction IDs do not uniquely cover the fixed public questions in canonical order: ' + suite)
        expected_manifest = {'file': path.name, 'sha256': sha(path), 'questions': len(predictions),
                             'public_sha256': spec['public_sha256'], 'gold_sha256': spec['gold_sha256']}
        _same(report['prediction_artifacts'][suite], expected_manifest, suite + '/artifact')
        for row, question in zip(predictions, spec['public']):
            validate_prediction(row, question, spec['gold'][question['example_id']], target_index, report['identity']['cfg'])
        recalculated = aggregate(predictions)
        _same(report['suites'][suite], recalculated, suite + '/metrics')
        validate_formal_suite_metric(suite, recalculated)
        accepted[suite] = expected_manifest
    if sha(folder / 'metrics.json') != report_sha:
        raise ValueError('Test report changed during artifact acceptance')
    return {'status': 'accepted', 'seed': report['seed'], 'report_sha256': report_sha, 'report_digest': digest(report),
            'training_batch_sha256': report['training_batch_sha256'], 'prediction_artifacts': accepted}
