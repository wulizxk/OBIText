"""Shared E1--E9 scoring, with prediction integrity and retrieval diagnostics."""
from collections import Counter
import gzip
import json
import math
import os
import time

from . import metrics
from .io import sha, write


def validate_prediction(q, p, vocab_size):
    length = q['gap_length']
    paths, scores = p['paths'], p['scores']
    required = min(10 if length == 1 else 5, vocab_size ** length)
    if len(paths) != required or len(scores) != len(paths) or len({tuple(x) for x in paths}) != len(paths):
        raise ValueError('Wrong number of unique candidate sequences')
    if any(len(path) != length or any(type(i) is not int or not 0 <= i < vocab_size for i in path) for path in paths):
        raise ValueError('Wrong candidate length/target ID')
    if any(not math.isfinite(s) for s in scores) or any(a < b for a, b in zip(scores, scores[1:])):
        raise ValueError('Non-finite or incorrectly ordered scores')
    ranking = p.get('single_ranking')
    if length == 1:
        if (not isinstance(ranking, list) or len(ranking) != vocab_size
                or any(type(i) is not int for i in ranking) or sorted(ranking) != list(range(vocab_size))
                or ranking[:required] != [x[0] for x in paths]):
            raise ValueError('Single MRR requires the complete target-vocabulary permutation')
    elif ranking is not None:
        raise ValueError('Multi-token full-vocabulary MRR is undefined')
    return p


def predict_file(model, questions, path, progress_every=250):
    """Do not accept a gold handle. The complete gzip is committed before scoring."""
    tmp = path.with_name(path.name + '.partial')
    if path.exists():
        raise FileExistsError(path)
    started = time.monotonic()
    with tmp.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=3) as stream:
            for n, q in enumerate(questions, 1):
                prediction = validate_prediction(q, model.predict(q), len(model.counts))
                record = {'example_id': q['example_id'], 'prediction': prediction}
                stream.write((json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode())
                if progress_every and (n % progress_every == 0 or n == len(questions)):
                    print(json.dumps({'phase': 'predict', 'suite': path.name, 'done': n,
                                      'total': len(questions), 'seconds': round(time.monotonic() - started, 3)}), flush=True)
        raw.flush(); os.fsync(raw.fileno())
    os.replace(tmp, path)
    return {'questions': len(questions), 'sha256': sha(path), 'seconds': time.monotonic() - started}


def score_file(questions, gold, targets, prediction_path, output):
    scored, diagnostics = [], Counter()
    with gzip.open(prediction_path, 'rt', encoding='utf-8') as stream:
        for q in questions:
            line = stream.readline()
            if not line:
                raise ValueError('Incomplete predictions')
            record = json.loads(line)
            if record['example_id'] != q['example_id']:
                raise ValueError('Prediction/question order mismatch')
            p = validate_prediction(q, record['prediction'], len(targets))
            value = metrics.score_prediction(q, p, gold[q['example_id']], targets)
            value['score_kind'] = p['score_kind']
            if 'retrieval' in p:
                evidence = p['retrieval']
                pool = [[targets[i] for i in path] for path in evidence['candidate_paths']]
                value['retrieval'] = evidence
                value['gold_in_retrieval_pool'] = gold[q['example_id']] in pool
                diagnostics['questions_with_retrieved_candidates'] += bool(pool)
                diagnostics['gold_in_retrieval_pool'] += value['gold_in_retrieval_pool']
                diagnostics['fallback_top1_questions'] += evidence['fallback_top1']
                diagnostics['fallback_in_returned_questions'] += bool(evidence['fallback_in_returned'])
            scored.append(value)
        if stream.read().strip():
            raise ValueError('Unexpected extra prediction records')
    result = metrics.aggregate(scored)
    if diagnostics:
        result['retrieval_diagnostics'] = {
            'denominator_all_questions': len(scored), **dict(diagnostics),
            'retrieval_coverage': diagnostics['questions_with_retrieved_candidates'] / len(scored),
            'gold_retrieval_pool_recall': diagnostics['gold_in_retrieval_pool'] / len(scored),
            'mrr_definition': 'retrieved single candidates followed by complete unigram-ranked tail',
            'scores_are_probabilities': False}
    # Per-question evidence is compact (full rankings remain in the gzip predictions).
    from .io import write_gzip
    write_gzip(output, scored)
    return result
