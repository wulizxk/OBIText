"""Recompute per-run metrics and optional frequency strata from saved predictions."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from release_utils import read, rows, sha, write


def validate(values):
    ids = set()
    for r in values:
        if r['example_id'] in ids:
            raise ValueError('Duplicate scored example')
        ids.add(r['example_id'])
        n, gold, candidates = r['gap_length'], r['gold_units'], r['candidates']
        if not candidates or len(gold) != n or any(len(c) != n for c in candidates):
            raise ValueError('Reference/candidate length mismatch')
        expected = dict(em1=float(candidates[0] == gold),
                        hit5=float(gold in candidates[:5]),
                        correct_positions=sum(x == y for x, y in zip(candidates[0], gold)))
        for key, value in expected.items():
            if r[key] != value:
                raise ValueError(f'Stored {key} disagrees with candidates: {r["example_id"]}')
        if n == 1:
            rank = r['gold_rank_full_vocabulary']
            if type(rank) is not int or rank < 1:
                raise ValueError('Missing/invalid full-vocabulary rank')
            if r['em1'] != int(rank == 1) or r['hit5'] != int(rank <= 5):
                raise ValueError('Candidate scores disagree with full-vocabulary ranking')
            if 'reciprocal_rank' in r and abs(r['reciprocal_rank'] - 1 / rank) > 1e-12:
                raise ValueError('Reciprocal rank mismatch')


def score(part):
    if not part:
        return dict(questions=0, metrics=None)
    out = dict(questions=len(part), em1=float(mean(r['em1'] for r in part)),
               hit5=float(mean(r['hit5'] for r in part)),
               position_accuracy=sum(r['correct_positions'] for r in part) / sum(r['gap_length'] for r in part))
    if all(r['gap_length'] == 1 for r in part):
        out.update(hit10=float(mean(r['gold_rank_full_vocabulary'] <= 10 for r in part)),
                   hit20=float(mean(r['gold_rank_full_vocabulary'] <= 20 for r in part)),
                   mrr_full_vocab=mean(1 / r['gold_rank_full_vocabulary'] for r in part))
    return out


def summarize(values, frequency=None):
    validate(values)
    by_length = {str(n): score([r for r in values if r['gap_length'] == n])
                 for n in sorted({r['gap_length'] for r in values})}
    result = dict(pooled=score(values), by_length=by_length)
    if set(by_length) == set('12345'):
        result.update(macro5_em1=mean(v['em1'] for v in by_length.values()),
                      macro5_hit5=mean(v['hit5'] for v in by_length.values()))
    if frequency is not None:
        if any(r['gap_length'] != 1 for r in values):
            raise ValueError('Frequency stratification requires single-unit questions')
        strata = {'0': [], '1-20': [], '21-99': [], '>=100': []}
        for r in values:
            unit = r['gold_units'][0]
            if unit not in frequency:
                raise ValueError('Target is absent from the supplied frequency inventory')
            f = frequency[unit]
            if type(f) is not int or f < 0:
                raise ValueError('Invalid training target frequency')
            strata['0' if f == 0 else '1-20' if f <= 20 else '21-99' if f <= 99 else '>=100'].append(r)
        result['frequency'] = {name: dict(score(part), target_types=len({r['gold_units'][0] for r in part}))
                               for name, part in strata.items()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scored', nargs='+', type=Path, required=True,
                   help='One run, or the independent seeds of the SAME model and question set')
    p.add_argument('--target-vocab', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new summary output file')
    frequency = read(a.target_vocab)['training_eligible_position_frequency'] if a.target_vocab else None
    per_run, signature = [], None
    for path in a.scored:
        values = rows(path)
        current = [(r['example_id'], r['source_id'], r['group_id'], r['gap_length'], r['gold_units']) for r in values]
        if signature is not None and current != signature:
            raise ValueError('Cannot average runs with different ordered questions or references')
        signature = current
        per_run.append(dict(scored_sha256=sha(path), summary=summarize(values, frequency)))
    def average(nodes):
        if isinstance(nodes[0], dict):
            if any(set(x) != set(nodes[0]) for x in nodes):
                raise ValueError('Summary structures differ')
            return {key: average([n[key] for n in nodes]) for key in nodes[0]}
        if nodes[0] is None:
            return None
        if all(type(n) is int for n in nodes):
            if len(set(nodes)) != 1:
                raise ValueError('Question/type counts differ across seeds')
            return nodes[0]
        return dict(mean=mean(nodes), sample_sd=stdev(nodes) if len(nodes) > 1 else None)
    report = dict(runs=len(per_run), metrics_are_fractions=True, per_run=per_run,
                  across_runs=average([r['summary'] for r in per_run]))
    if a.target_vocab:
        report['target_vocab_sha256'] = sha(a.target_vocab)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write(a.output, report)
    print(a.output)


if __name__ == '__main__':
    main()
