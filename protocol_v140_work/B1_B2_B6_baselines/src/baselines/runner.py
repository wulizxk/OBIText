"""Explicit CPU-only fit -> full Dev -> freeze -> manuscript Test lifecycle."""
import argparse
import json
from pathlib import Path
from benchmark_protocol_v140 import EVALUATION_QUESTION_INSTANCES, compatible_source
import resource
import time

from .data import DataGate, MANIFEST_SHA
from .evaluation import predict_file, score_file
from .io import ROOT, digest, now, read, read_gzip, reserve, runtime, sha, source_hash, verify_files, write, write_gzip
from .models import fit, restore, validate_config
from .protocol import FORMAL_TEST_SUITES, validate_formal_suite_metrics


def config_for(name):
    # No implicit search against Dev/Test: configurations live in reviewed files.
    value = validate_config(read(ROOT / 'configs' / (name + '.json')))
    if value['experiment'] != name:
        raise ValueError('Experiment/config mismatch')
    return value


def common(gate):
    return {'created_at_utc': now(), 'data_version': 'v1.5.0', 'data_manifest_sha256': MANIFEST_SHA,
            'source_sha256': source_hash(), 'runtime': runtime(), 'data_files_read': dict(gate.accessed)}


def compatible(meta):
    if (meta.get('data_manifest_sha256') != MANIFEST_SHA or not compatible_source(ROOT, meta.get('source_sha256'), source_hash(), baseline=True)
            or meta.get('runtime') != runtime()):
        raise ValueError('Data/code/runtime changed; results must not be silently mixed')


def artifact(root, marker, kind):
    root = Path(root).resolve()
    meta = read(root / marker)
    compatible(meta)
    if meta.get('status') != 'COMPLETE' or meta.get('kind') != kind or meta.get('diagnostic_only') is not False:
        raise ValueError('Not a valid formal completion artifact')
    required = {
        'fit': {'model.json.gz', 'train_identities.json', 'config.json'},
        'dev': {'metrics.json', 'PREDICTIONS_COMPLETE.json', 'dev_full.predictions.jsonl.gz', 'dev_full.scored.json.gz'},
        'selection': set(),
        'test': {'metrics.json', 'PREDICTIONS_COMPLETE.json'} | {
            s + suffix for s in FORMAL_TEST_SUITES for suffix in ('.predictions.jsonl.gz', '.scored.json.gz')},
    }
    if set(meta.get('files', {})) != required[kind]:
        raise ValueError('Missing or unexpected files in formal completion evidence')
    verify_files(root, meta['files'])
    return root, meta


def load_fit(path, gate):
    root, meta = artifact(path, 'FIT_COMPLETE.json', 'fit')
    if meta['train_rows'] != 66511 or meta['covered_targets'] != 4116:
        raise ValueError('Fit did not use the full Train set / target inventory')
    gate.path('splits/train.jsonl')
    for name, value in meta['data_files_read'].items():
        if name not in gate.manifest['files'] or gate.manifest['files'][name] != value:
            raise ValueError('Fit data identity mismatch')
        if 'test' in name or 'dev_' in name or name.startswith('splits/') and name != 'splits/train.jsonl':
            raise ValueError('Non-Train content consumed by fit')
    payload = read_gzip(root / 'model.json.gz')
    if payload['config'] != config_for(meta['experiment']) or digest(payload['config']) != meta['config_sha256']:
        raise ValueError('Fitted config differs from the currently reviewed config')
    if payload['target_input_ids'] != gate.target_input_ids:
        raise ValueError('Fitted target vocabulary alignment changed')
    if len(payload['counts']) != 4116 or not all(payload['counts']):
        raise ValueError('Full fitted model has incomplete target support')
    return root, meta, restore(payload)


def check_isolation(questions, fit_root):
    identities = read(fit_root / 'train_identities.json')
    ids, groups = set(identities), set(identities.values())
    if any(q['source_id'] in ids or q['group_id'] in groups for q in questions):
        raise ValueError('Train/evaluation source or group overlap')


def do_fit(args):
    gate = DataGate(args.data)
    config = config_for(args.experiment)
    output = reserve(args.output, gate.root)
    started = time.monotonic()
    documents = gate.train()
    model = fit(config, gate, documents)
    if not all(model.counts):
        raise ValueError('Not every frozen target has Train support')
    write_gzip(output / 'model.json.gz', model.payload())
    write(output / 'train_identities.json', {r['sequence_id']: r['group_id'] for r in documents})
    write(output / 'config.json', config)
    value = {**common(gate), 'kind': 'fit', 'status': 'COMPLETE', 'diagnostic_only': False,
             'experiment': args.experiment, 'train_rows': len(documents), 'covered_targets': sum(c > 0 for c in model.counts),
             'train_target_occurrences': sum(model.counts), 'config_sha256': digest(config),
             'seconds': time.monotonic() - started,
             'files': {p: sha(output / p) for p in ('model.json.gz', 'train_identities.json', 'config.json')}}
    write(output / 'FIT_COMPLETE.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def do_audit(args):
    """Read-only corpus validation, without fitting a statistical model or opening gold."""
    gate = DataGate(args.data)
    output = reserve(args.output, gate.root)
    documents = gate.train()
    questions = gate.public('dev_full')
    ids, groups = {r['sequence_id'] for r in documents}, {r['group_id'] for r in documents}
    if any(q['source_id'] in ids or q['group_id'] in groups for q in questions):
        raise ValueError('Train/Dev identity overlap')
    targets = {t for row in documents for t in row['targets'] if t >= 0}
    if targets != set(range(4116)):
        raise ValueError('Train target coverage is incomplete')
    write(output / 'DATA_AUDIT.json', {**common(gate), 'status': 'PASS', 'diagnostic_only': True,
          'train_rows': len(documents), 'dev_questions': len(questions), 'covered_targets': len(targets),
          'train_dev_source_overlap': 0, 'train_dev_group_overlap': 0,
          'train_eligible_gaps': {str(n): sum(len(r['eligible'][str(n)]) for r in documents) for n in range(1, 6)},
          'all_train_unit_protection_and_eligible_starts_checked': True,
          'test_opened': False, 'gold_opened': False, 'model_fitted': False})
    print(json.dumps({'data_audit': str(output / 'DATA_AUDIT.json'), 'status': 'PASS'}))


def do_dev(args):
    gate = DataGate(args.data)
    fit_root, fit_meta, model = load_fit(args.fit, gate)
    questions = gate.public('dev_full')
    check_isolation(questions, fit_root)
    output = reserve(args.output, gate.root)
    pred = output / 'dev_full.predictions.jsonl.gz'
    details = predict_file(model, questions, pred)
    write(output / 'PREDICTIONS_COMPLETE.json', {'suite': 'dev_full', **details, 'completed_at_utc': now()})
    gold = gate.gold('dev_full', questions)
    metrics = score_file(questions, gold, gate.targets, pred, output / 'dev_full.scored.json.gz')
    if metrics['questions'] != 15024 or metrics['macro5_em1'] is None or metrics['macro5_hit5'] is None:
        raise ValueError('Full Dev was not completely evaluated')
    write(output / 'metrics.json', metrics)
    meta = {**common(gate), 'kind': 'dev', 'status': 'COMPLETE', 'diagnostic_only': False,
            'experiment': fit_meta['experiment'], 'fit_path': str(fit_root),
            'fit_completion_sha256': sha(fit_root / 'FIT_COMPLETE.json'),
            'config_sha256': fit_meta['config_sha256'], 'questions': len(questions),
            'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write(output / 'DEV_COMPLETE.json', meta)
    print(json.dumps({'dev_complete': str(output), 'macro5_em1': metrics['macro5_em1'],
                      'macro5_hit5': metrics['macro5_hit5']}, ensure_ascii=False), flush=True)


def do_freeze(args):
    gate = DataGate(args.data)
    fit_root, fit_meta = artifact(args.fit, 'FIT_COMPLETE.json', 'fit')
    dev_root, dev_meta = artifact(args.dev, 'DEV_COMPLETE.json', 'dev')
    if (fit_meta['train_rows'] != 66511 or fit_meta['covered_targets'] != 4116 or dev_meta['questions'] != 15024
            or dev_meta['fit_completion_sha256'] != sha(fit_root / 'FIT_COMPLETE.json')
            or Path(dev_meta['fit_path']).resolve() != fit_root or dev_meta['config_sha256'] != fit_meta['config_sha256']
            or fit_meta['config_sha256'] != digest(config_for(fit_meta['experiment']))):
        raise ValueError('Fit/Dev identity or formal completeness mismatch')
    metrics = read(dev_root / 'metrics.json')
    if metrics['questions'] != 15024 or metrics['macro5_em1'] is None or metrics['macro5_hit5'] is None:
        raise ValueError('Cannot freeze without complete five-length Dev metrics')
    if {k: v['questions'] for k, v in metrics['by_length'].items()} != gate.protocol['validation_length_counts']:
        raise ValueError('Full Dev length counts differ from the frozen protocol')
    output = reserve(args.output, gate.root)
    write(output / 'SELECTION.json', {**common(gate), 'kind': 'selection', 'status': 'COMPLETE',
          'diagnostic_only': False, 'experiment': fit_meta['experiment'], 'fit_path': str(fit_root),
          'dev_path': str(dev_root), 'fit_completion_sha256': sha(fit_root / 'FIT_COMPLETE.json'),
          'dev_completion_sha256': sha(dev_root / 'DEV_COMPLETE.json'), 'config_sha256': fit_meta['config_sha256'],
          'selection_rule': 'preregistered config; full Dev recorded before Test; no Test optimization',
          'dev_macro5_em1': metrics['macro5_em1'], 'dev_macro5_hit5': metrics['macro5_hit5'], 'files': {}})
    print(json.dumps({'frozen_selection': str(output)}, ensure_ascii=False))


def do_test(args):
    if not args.confirm_final:
        raise ValueError('Formal Test requires --confirm-final and a full-Dev-frozen selection')
    gate = DataGate(args.data, allow_test=True)
    selection_root, selection = artifact(args.selection, 'SELECTION.json', 'selection')
    fit_root, fit_meta, model = load_fit(selection['fit_path'], gate)
    dev_root, dev_meta = artifact(selection['dev_path'], 'DEV_COMPLETE.json', 'dev')
    if (sha(fit_root / 'FIT_COMPLETE.json') != selection['fit_completion_sha256']
            or sha(dev_root / 'DEV_COMPLETE.json') != selection['dev_completion_sha256']
            or dev_meta['fit_completion_sha256'] != selection['fit_completion_sha256']
            or dev_meta['questions'] != 15024 or selection['config_sha256'] != fit_meta['config_sha256']
            or selection['experiment'] != fit_meta['experiment'] or dev_meta['experiment'] != fit_meta['experiment']):
        raise ValueError('Frozen selection evidence changed')
    output = reserve(args.output, gate.root)
    suites, progress = {}, {}
    # Deliberate two-phase barrier: all public predictions before any Test gold.
    for suite in FORMAL_TEST_SUITES:
        questions = gate.public(suite)
        check_isolation(questions, fit_root)
        suites[suite] = questions
        progress[suite] = predict_file(model, questions, output / (suite + '.predictions.jsonl.gz'))
    write(output / 'PREDICTIONS_COMPLETE.json', {'completed_at_utc': now(), 'suites': progress,
                                               'test_gold_opened': False})
    gate.allow_test_gold = True
    results = {}
    for suite, questions in suites.items():
        gold = gate.gold(suite, questions)
        results[suite] = score_file(questions, gold, gate.targets, output / (suite + '.predictions.jsonl.gz'),
                                    output / (suite + '.scored.json.gz'))
    validate_formal_suite_metrics(results)
    write(output / 'metrics.json', {'experiment': fit_meta['experiment'], 'data_version': 'v1.5.0', 'suites': results,
          'statistical_replicates': 1, 'seed_mean_std': None,
          'cross_suite_note': 'Report each suite separately because evaluation suites can share source records and questions.'})
    write(output / 'TEST_COMPLETE.json', {**common(gate), 'kind': 'test', 'status': 'COMPLETE', 'diagnostic_only': False,
          'experiment': fit_meta['experiment'], 'selection_path': str(selection_root),
          'selection_sha256': sha(selection_root / 'SELECTION.json'), 'fit_completion_sha256': selection['fit_completion_sha256'],
          'questions': EVALUATION_QUESTION_INSTANCES, 'suites': list(FORMAL_TEST_SUITES),
          'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'test_complete': str(output), 'suites': len(FORMAL_TEST_SUITES), 'questions': EVALUATION_QUESTION_INSTANCES}, ensure_ascii=False))


def choose_smoke_questions(questions, limit):
    result, seen = [], set()
    # Fixed answer-free coverage of lengths and both boundaries.
    predicates = [lambda q: not q['prefix_ids'], lambda q: not q['suffix_ids']]
    predicates += [lambda q, n=n: q['gap_length'] == n for n in range(1, 6)]
    for predicate in predicates:
        q = next((q for q in questions if predicate(q) and q['example_id'] not in seen), None)
        if q:
            result.append(q); seen.add(q['example_id'])
    for q in questions:
        if len(result) >= limit:
            break
        if q['example_id'] not in seen:
            result.append(q); seen.add(q['example_id'])
    return result[:limit]


def do_smoke(args):
    if not 10 <= args.train_limit <= 2000 or not 7 <= args.dev_limit <= 30:
        raise ValueError('Bounded smoke only: Train 10..2000, Dev 7..30; no Test')
    gate = DataGate(args.data)
    output = reserve(args.output, gate.root)
    documents = gate.train(args.train_limit)
    full_questions = gate.public('dev_full')
    questions = choose_smoke_questions(full_questions, args.dev_limit)
    train_ids, train_groups = {r['sequence_id'] for r in documents}, {r['group_id'] for r in documents}
    if any(q['source_id'] in train_ids or q['group_id'] in train_groups for q in questions):
        raise ValueError('Smoke Train/Dev overlap')
    names = ('B1', 'B2', 'B6') if args.experiment == 'all' else (args.experiment,)
    predictions, reports = {}, {}
    for name in names:
        started = time.monotonic()
        model = fit(config_for(name), gate, documents)
        write_gzip(output / (name + '.model.json.gz'), model.payload())
        restored = restore(read_gzip(output / (name + '.model.json.gz')))
        if any(model.predict(q) != restored.predict(q) for q in questions[:2]):
            raise ValueError('Model serialization round-trip changed predictions')
        predictions[name] = predict_file(restored, questions, output / (name + '.predictions.jsonl.gz'), 5)
        reports[name] = {'fit_and_prediction_seconds': time.monotonic() - started,
                         'covered_targets': sum(c > 0 for c in model.counts), 'serialization_roundtrip': 'PASS'}
        del model, restored
    gold = gate.gold('dev_full', full_questions)
    for name in names:
        reports[name]['diagnostic_dev_metrics_not_formal'] = score_file(
            questions, gold, gate.targets, output / (name + '.predictions.jsonl.gz'), output / (name + '.scored.json.gz'))
    report = {**common(gate), 'status': 'PASS', 'diagnostic_only': True, 'train_rows': len(documents),
              'dev_questions': len(questions), 'lengths': sorted({q['gap_length'] for q in questions}),
              'prefix_question_present': any(not q['prefix_ids'] for q in questions),
              'suffix_question_present': any(not q['suffix_ids'] for q in questions),
              'formal_test_opened': any('test_' in k for k in gate.accessed), 'models': reports,
              'max_rss_platform_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'predictions': predictions}
    write(output / 'SMOKE_REPORT.json', report)
    print(json.dumps({'smoke_report': str(output / 'SMOKE_REPORT.json'), 'status': 'PASS'}, ensure_ascii=False))


def do_summary(args):
    result = {}
    for path in args.tests:
        root, meta = artifact(path, 'TEST_COMPLETE.json', 'test')
        if meta['questions'] != EVALUATION_QUESTION_INSTANCES or meta['suites'] != list(FORMAL_TEST_SUITES):
            raise ValueError('Incomplete Test completion marker')
        metrics = read(root / 'metrics.json')
        validate_formal_suite_metrics(metrics['suites'])
        name = meta['experiment']
        if name in result or name != metrics['experiment']:
            raise ValueError('Duplicate/mismatched experiment')
        result[name] = {'metrics': metrics, 'test_path': str(root), 'completion_sha256': sha(root / 'TEST_COMPLETE.json')}
    if set(result) != {'B1', 'B2', 'B6'}:
        raise ValueError('Summary requires one complete result for each of B1/B2/B6')
    output = reserve(args.output, args.data)
    write(output / 'B1_B2_B6_summary.json', {'created_at_utc': now(), 'data_version': 'v1.5.0', 'results': result,
                                           'replicates_per_baseline': 1, 'seed_mean_std': None})
    print(str(output / 'B1_B2_B6_summary.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent / 'shared/data/v1.5.0')
    subs = parser.add_subparsers(dest='command', required=True)
    sub = subs.add_parser('audit', help='Validate all Train rows and public Dev; no fit, no gold, no Test')
    sub.add_argument('--output', required=True, type=Path); sub.set_defaults(action=do_audit)
    sub = subs.add_parser('fit', help='Fit once on all Train records, CPU only')
    sub.add_argument('--experiment', choices=('B1', 'B2', 'B6'), required=True)
    sub.add_argument('--output', required=True, type=Path); sub.set_defaults(action=do_fit)
    sub = subs.add_parser('dev', help='Full fixed Dev evaluation, no Test access')
    sub.add_argument('--fit', required=True, type=Path); sub.add_argument('--output', required=True, type=Path)
    sub.set_defaults(action=do_dev)
    sub = subs.add_parser('freeze', help='Freeze reviewed config after complete Dev')
    sub.add_argument('--fit', required=True, type=Path); sub.add_argument('--dev', required=True, type=Path)
    sub.add_argument('--output', required=True, type=Path); sub.set_defaults(action=do_freeze)
    sub = subs.add_parser('test', help='Explicit gated final manuscript Test')
    sub.add_argument('--selection', required=True, type=Path); sub.add_argument('--output', required=True, type=Path)
    sub.add_argument('--confirm-final', action='store_true'); sub.set_defaults(action=do_test)
    sub = subs.add_parser('smoke', help='Bounded CPU Train-subset/Dev diagnostic; never formal Test')
    sub.add_argument('--experiment', choices=('all', 'B1', 'B2', 'B6'), default='all')
    sub.add_argument('--train-limit', type=int, default=1000); sub.add_argument('--dev-limit', type=int, default=15)
    sub.add_argument('--output', required=True, type=Path); sub.set_defaults(action=do_smoke)
    sub = subs.add_parser('summary', help='Combine three deterministic baselines, not three random seeds')
    sub.add_argument('--tests', nargs=3, required=True, type=Path); sub.add_argument('--output', required=True, type=Path)
    sub.set_defaults(action=do_summary)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
