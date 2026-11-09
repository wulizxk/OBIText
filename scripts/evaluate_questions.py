"""Evaluate explicit frozen test questions with the original trained model code."""
import argparse
import gzip
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from release_utils import REPO, FROZEN_MANIFEST, checked_questions, read, sha, write


def main():
    registry = read(REPO / 'provenance/model_registry.json')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=sorted(registry), required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--fit', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory')
    folder, package, _ = registry[a.model]
    if package and (not a.checkpoint or a.fit or a.selection):
        p.error('Neural models require --checkpoint and no baseline fit/selection')
    if not package and (not a.fit or not a.selection or a.checkpoint):
        p.error('Deterministic models require --fit and --selection')
    root = REPO / 'protocol_v140_work'
    project = root / folder
    sys.path[:0] = [str(project / 'src'), str(root / 'shared')]
    from benchmark_protocol_v140 import identify_evaluation_suite, validate_formal_suite_metric
    public_sha, gold_sha = sha(a.questions), sha(a.gold)
    suite = identify_evaluation_suite(public_sha, gold_sha)
    public, gold, _ = checked_questions(a.data, a.questions, a.gold, a.manifest)
    report = dict(model=a.model, suite=suite, questions=len(public), data_manifest_sha256=FROZEN_MANIFEST,
                  questions_sha256=public_sha, gold_sha256=gold_sha,
                  question_manifest_sha256=sha(a.manifest), evaluator_sha256=sha(__file__))
    started = time.monotonic()
    if package:
        import torch
        engine = importlib.import_module(package + '.engine')
        state = importlib.import_module(package + '.run_state')
        runtime = importlib.import_module(package + '.runtime')
        import fcntl
        with (a.checkpoint.parent / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            ckpt, _ = state.require_completed_run(a.checkpoint)
            experiment = ckpt['identity']['cfg']['experiment'].upper().replace('-A', '')
            if experiment != a.model:
                raise ValueError('Checkpoint experiment does not match --model')
            engine.seed_all(ckpt['identity']['seed'])
            device = torch.device(a.device)
            ckpt, cfg, codec, model = engine.inspect_checkpoint(project, a.data, a.checkpoint, device, ckpt)
            metrics, scored = engine.evaluate(model, codec, public, gold, cfg, device)
            report.update(seed=ckpt['identity']['seed'], epoch=ckpt['epoch'],
                          checkpoint_sha256=sha(a.checkpoint), identity=ckpt['identity'],
                          evaluation_runtime=runtime.evaluation_environment(device, cfg))
    else:
        from baselines.data import DataGate
        from baselines.runner import artifact, load_fit, check_isolation
        from baselines.evaluation import validate_prediction
        from baselines.metrics import score_prediction, aggregate
        _, selection = artifact(a.selection, 'SELECTION.json', 'selection')
        gate = DataGate(a.data, allow_test=True)
        fitroot, meta, model = load_fit(a.fit, gate)
        if (meta['experiment'] != a.model or selection['experiment'] != a.model
                or sha(fitroot / 'FIT_COMPLETE.json') != selection['fit_completion_sha256']):
            raise ValueError('Baseline selection/fit identity mismatch')
        check_isolation(public, fitroot)
        scored = []
        for i, q in enumerate(public):
            prediction = validate_prediction(q, model.predict(gate.question(q)), len(gate.targets))
            scored.append(score_prediction(q, prediction, gold[q['example_id']], gate.targets))
            if (i + 1) % 1000 == 0:
                print(f'EVAL {i + 1}/{len(public)}', flush=True)
        metrics = aggregate(scored)
        report.update(seed=None, source_sha256=meta['source_sha256'],
                      selection_sha256=sha(a.selection / 'SELECTION.json'),
                      fit_completion_sha256=sha(fitroot / 'FIT_COMPLETE.json'))
    validate_formal_suite_metric(suite, metrics)
    if len(scored) != len(public):
        raise ValueError('Incomplete prediction output')
    for q, r in zip(public, scored):
        if q['example_id'] != r['example_id'] or r['gold_units'] != gold[q['example_id']]:
            raise ValueError('Prediction order/reference mismatch')
        r['start_unit'] = q['start_unit']
        rank = r['gold_rank_full_vocabulary']
        if q['gap_length'] == 1:
            if not 1 <= rank <= 4116:
                raise ValueError('Invalid full-vocabulary rank')
            r['hit20'] = int(rank <= 20)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + a.output.name + '-', dir=a.output.parent))
    try:
        with gzip.open(staging / 'scored.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for row in scored:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        report.update(status='complete', metrics=metrics,
                      elapsed_seconds=time.monotonic() - started,
                      scored_sha256=sha(staging / 'scored.jsonl.gz'))
        write(staging / 'metrics.json', report)
        if a.output.exists():
            raise FileExistsError(a.output)
        os.rename(staging, a.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
