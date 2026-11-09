from collections import defaultdict
import statistics


def score_prediction(q, prediction, gold_units, target_units):
    paths = [[target_units[i] for i in path] for path in prediction['paths']]
    length = q['gap_length']
    assert len(gold_units) == length and all(len(p) == length for p in paths)
    rank = paths.index(gold_units) + 1 if gold_units in paths else None
    exact_rank = None
    if length == 1:
        target_index = target_units.index(gold_units[0])
        exact_rank = prediction['single_ranking'].index(target_index) + 1
    return {'example_id': q['example_id'], 'source_id': q['source_id'], 'group_id': q['group_id'],
            'boundary_side': q.get('boundary_side'),
            'gap_length': length, 'gold_units': gold_units, 'candidates': paths,
            'candidate_scores': prediction['scores'], 'score_kind': 'left_to_right_pseudo_log_probability',
            'em1': float(paths[0] == gold_units), 'hit5': float(rank is not None and rank <= 5),
            'hit10': float(exact_rank <= 10) if exact_rank is not None else None,
            'correct_positions': sum(a == b for a, b in zip(paths[0], gold_units)),
            'gold_rank_full_vocabulary': exact_rank,
            'reciprocal_rank': 1.0 / exact_rank if exact_rank is not None else None}


def aggregate(values):
    def group(rs):
        single = all(r['gap_length'] == 1 for r in rs)
        return {'questions': len(rs), 'em1': statistics.mean(r['em1'] for r in rs),
                'hit5': statistics.mean(r['hit5'] for r in rs),
                'position_accuracy': sum(r['correct_positions'] for r in rs) / sum(r['gap_length'] for r in rs),
                'hit10': statistics.mean(r['hit10'] for r in rs) if single else None,
                'mrr_full_vocab': statistics.mean(r['reciprocal_rank'] for r in rs) if single else None}
    lengths, sources, modes, boundary_sides = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for value in values:
        lengths[value['gap_length']].append(value)
        sources[value['source_id']].append(value)
        modes['1' if value['gap_length'] == 1 else '2' if value['gap_length'] == 2 else '3-5'].append(value)
        if value.get('boundary_side'):
            boundary_sides[value['boundary_side']].append(value)
    by_length = {str(k): group(v) for k, v in sorted(lengths.items())}
    complete = set(lengths) == {1, 2, 3, 4, 5}
    return {'questions': len(values), 'by_length': by_length,
            'by_mode': {k: group(v) for k, v in modes.items()}, 'pooled_not_macro': group(values),
            'by_boundary_side': {k: group(v) for k, v in sorted(boundary_sides.items())},
            'macro5_em1': statistics.mean(v['em1'] for v in by_length.values()) if complete else None,
            'macro5_hit5': statistics.mean(v['hit5'] for v in by_length.values()) if complete else None,
            'source_macro_em1': statistics.mean(statistics.mean(r['em1'] for r in rs) for rs in sources.values())}


def better(candidate, best):
    if candidate['macro5_em1'] is None:
        raise ValueError('Checkpoint selection requires all five lengths')
    return best is None or (candidate['macro5_em1'], candidate['macro5_hit5']) > (best['macro5_em1'], best['macro5_hit5'])
