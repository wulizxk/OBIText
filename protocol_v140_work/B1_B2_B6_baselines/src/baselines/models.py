"""Deterministic atomic-unit baselines; prediction accepts only visible context."""
from collections import Counter, defaultdict, OrderedDict
import heapq
import math

import numpy as np

from .data import BOS, EOS


def rank(scores, count=None):
    # mergesort preserves frozen target-ID order in exact ties.
    result = np.argsort(-np.asarray(scores), kind='stable')
    return result[:count].tolist() if count is not None else result.tolist()


def validate_config(config):
    if config.get('experiment') not in ('B1', 'B2', 'B6') or config.get('dataset_version') != 'v1.5.0':
        raise ValueError('Unknown experiment/data version')
    if (config.get('beam_width'), config.get('single_return'), config.get('multi_return')) != (10, 10, 5):
        raise ValueError('Frozen decoding contract is beam=10, single=10, multi=5')
    if not 0 < config.get('unigram_alpha', 0) < 10:
        raise ValueError('Positive finite unigram smoothing required')
    if config.get('parameter_selection') != 'preregistered_not_dev_optimized':
        raise ValueError('This release implements the preregistered configuration only')
    if config['experiment'] == 'B2':
        if (config.get('order') != 5 or config.get('smoothing') != 'witten_bell_interpolation'
                or config.get('proposal_directions') != 'forward_and_backward'
                or config.get('maximum_proposal_union') != 20
                or not 0 <= config.get('forward_weight', -1) <= 1):
            raise ValueError('B2 requires bidirectional interpolated 5-gram')
    if config['experiment'] == 'B6':
        for key in ('document_limit', 'span_limit', 'retrieval_candidate_limit', 'alignment_limit'):
            if type(config.get(key)) is not int or not 1 <= config[key] <= 10000:
                raise ValueError('Invalid bounded retrieval parameter: ' + key)
        if (config.get('ngram_max') != 3 or not 0 < config.get('max_df_ratio', 0) <= 1
                or not 0 < config.get('bm25_k1', 0) <= 10 or not 0 <= config.get('bm25_b', -1) <= 1
                or not 0 <= config.get('alignment_weight', -1) <= 100
                or not 0 <= config.get('boundary_weight', -1) <= 100
                or config.get('require_local_alignment_or_boundary') is not True
                or config.get('exclude_query_group') is not True
                or config.get('fallback') != 'unigram_after_retrieved_candidates'):
            raise ValueError('Invalid B6 retrieval/fallback contract')
    return config


class Unigram:
    def __init__(self, config, target_input_ids, counts):
        self.config = validate_config(config)
        self.target_input_ids = list(target_input_ids)
        self.counts = list(counts)
        if len(counts) != len(target_input_ids) or any(type(x) is not int or x < 0 for x in counts):
            raise ValueError('Invalid Train counts')
        if not sum(counts):
            raise ValueError('No supervised Train units')
        probs = np.asarray(counts, dtype=np.float64) + config['unigram_alpha']
        self.probs = probs / probs.sum()
        self.logs = np.log(self.probs)
        self.ranking = rank(self.logs)

    def payload(self):
        return {'config': self.config, 'target_input_ids': self.target_input_ids, 'counts': self.counts}

    def predict(self, question):
        length = question['gap_length']
        if length == 1:
            chosen = self.ranking[:10]
            return {'paths': [[i] for i in chosen], 'scores': [float(self.logs[i]) for i in chosen],
                    'single_ranking': self.ranking, 'score_kind': 'unigram_joint_log_probability'}
        beam = [((), 0.0)]
        for _ in range(length):
            expanded = [(path + (i,), score + float(self.logs[i]))
                        for path, score in beam for i in self.ranking[:10]]
            beam = sorted(expanded, key=lambda x: (-x[1], x[0]))[:10]
        return {'paths': [list(p) for p, _ in beam[:5]], 'scores': [s for _, s in beam[:5]],
                'single_ranking': None, 'score_kind': 'unigram_joint_log_probability'}


class Ngram(Unigram):
    def __init__(self, config, target_input_ids, counts, tables, lacuna_n):
        super().__init__(config, target_input_ids, counts)
        self.lacuna_n = lacuna_n
        self.tables = tables
        self.cache = OrderedDict()

    def context(self, history, marker):
        result = []
        for token in reversed(history):
            result.append(token)
            if len(result) == 4 or token == self.lacuna_n:
                break
        else:
            if len(result) < 4:
                result.append(marker)
        return tuple(reversed(result))

    def distribution(self, history, direction):
        context = self.context(history, BOS if direction == 'forward' else EOS)
        key = (direction, context)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        p = self.probs.copy()
        table = self.tables[direction]
        for order in range(1, len(context) + 1):
            counts = table.get(context[-order:])
            if counts:
                total, types = sum(counts.values()), len(counts)
                p *= types / (total + types)
                ids = np.fromiter(counts.keys(), dtype=np.int64)
                vals = np.fromiter(counts.values(), dtype=np.float64)
                p[ids] += vals / (total + types)
        p.flags.writeable = False
        self.cache[key] = p
        if len(self.cache) > 1024:
            self.cache.popitem(last=False)
        return p

    def directional_beam(self, context, length, direction):
        beam = [((), 0.0)]
        for _ in range(length):
            expanded = []
            for path, score in beam:
                history = context + [self.target_input_ids[i] for i in path]
                logs = np.log(self.distribution(history, direction))
                expanded.extend((path + (i,), score + float(logs[i])) for i in rank(logs, 10))
            beam = sorted(expanded, key=lambda x: (-x[1], x[0]))[:10]
        return [path if direction == 'forward' else path[::-1] for path, _ in beam]

    def joint_score(self, pre, suf, path):
        alpha = self.config['forward_weight']
        tokens = [self.target_input_ids[i] for i in path]
        fwd, bwd = 0.0, 0.0
        for i, target in enumerate(path):
            fwd += math.log(float(self.distribution(pre + tokens[:i], 'forward')[target]))
            bwd += math.log(float(self.distribution(list(reversed(suf)) + list(reversed(tokens[i+1:])),
                                                     'backward')[target]))
        return alpha * fwd + (1 - alpha) * bwd

    def predict(self, question):
        pre, suf, length = question['prefix_ids'], question['suffix_ids'], question['gap_length']
        if length == 1:
            alpha = self.config['forward_weight']
            scores = (alpha * np.log(self.distribution(pre, 'forward'))
                      + (1 - alpha) * np.log(self.distribution(suf[::-1], 'backward')))
            ranking = rank(scores)
            return {'paths': [[i] for i in ranking[:10]], 'scores': [float(scores[i]) for i in ranking[:10]],
                    'single_ranking': ranking, 'score_kind': 'bidirectional_log_linear_score_not_normalized'}
        pool = set(self.directional_beam(pre, length, 'forward'))
        pool.update(self.directional_beam(suf[::-1], length, 'backward'))
        scored = sorted(((p, self.joint_score(pre, suf, p)) for p in pool), key=lambda x: (-x[1], x[0]))
        return {'paths': [list(p) for p, _ in scored[:5]], 'scores': [s for _, s in scored[:5]],
                'single_ranking': None, 'score_kind': 'bidirectional_log_linear_score_not_normalized',
                'proposal_count': len(pool)}

    def payload(self):
        value = super().payload()
        value.update(lacuna_n=self.lacuna_n,
                     tables={d: [[list(h), sorted(c.items())] for h, c in sorted(table.items())]
                             for d, table in self.tables.items()})
        return value


def features(tokens, maximum=3):
    return Counter(tuple(tokens[i:i+n]) for n in range(1, maximum+1)
                   for i in range(len(tokens) - n + 1))


def local_matches(pre, suf, tokens, start, length, limit):
    left = right = 0
    for a, b in zip(reversed(pre[-limit:]), reversed(tokens[max(0, start-limit):start])):
        if a != b:
            break
        left += 1
    for a, b in zip(suf[:limit], tokens[start+length:start+length+limit]):
        if a != b:
            break
        right += 1
    boundary = int(not pre and start == 0) + int(not suf and start + length == len(tokens))
    return left, right, boundary


class Retriever(Unigram):
    def __init__(self, config, target_input_ids, counts, documents):
        super().__init__(config, target_input_ids, counts)
        self.documents = documents
        # Two passes avoid keeping both unfiltered postings and all document features.
        dfs, sizes = Counter(), []
        for row in documents:
            fs = features(row['tokens'])
            dfs.update(fs.keys())
            sizes.append(sum(fs.values()))
        self.sizes = sizes
        self.avg_size = sum(sizes) / len(sizes)
        cap = max(1, int(len(documents) * config['max_df_ratio']))
        self.index = defaultdict(list)
        for i, row in enumerate(documents):
            for feature, count in features(row['tokens']).items():
                if dfs[feature] <= cap:
                    self.index[feature].append((i, count))

    def retrieve(self, question):
        c, pre, suf = self.config, question['prefix_ids'], question['suffix_ids']
        # No n-gram is ever constructed across the missing region.
        query_features = set(features(pre)) | set(features(suf))
        scores = defaultdict(float)
        for feature in sorted(query_features):
            postings = self.index.get(feature, ())
            idf = math.log(1 + (len(self.documents) - len(postings) + .5) / (len(postings) + .5))
            for i, tf in postings:
                row = self.documents[i]
                if row['group_id'] == question.get('group_id') or row['sequence_id'] == question.get('source_id'):
                    continue
                norm = c['bm25_k1'] * (1 - c['bm25_b'] + c['bm25_b'] * self.sizes[i] / self.avg_size)
                scores[i] += idf * tf * (c['bm25_k1'] + 1) / (tf + norm)
        docs = heapq.nsmallest(c['document_limit'], scores, key=lambda i: (-scores[i], i))
        length = question['gap_length']

        def spans():
            for i in docs:
                row = self.documents[i]
                for start in row['eligible'][str(length)]:
                    left, right, boundary = local_matches(pre, suf, row['tokens'], start, length, c['alignment_limit'])
                    if not left + right + boundary:
                        continue
                    score = scores[i] + c['alignment_weight'] * (left + right) + c['boundary_weight'] * boundary
                    yield (score, i, start)

        selected = heapq.nsmallest(c['span_limit'], spans(), key=lambda x: (-x[0], x[1], x[2]))
        grouped = {}
        for score, i, start in selected:
            row = self.documents[i]
            path = tuple(row['targets'][start:start+length])
            if len(path) != length or min(path) < 0:
                raise ValueError('Retriever attempted to extract a protected target')
            item = grouped.setdefault(path, {'maximum_score': score, 'support': 0, 'evidence': []})
            item['support'] += 1
            if len(item['evidence']) < 3:
                item['evidence'].append({'sequence_id': row['sequence_id'], 'group_id': row['group_id'],
                                         'start_unit': start, 'score': score})
        result = []
        for path, item in grouped.items():
            item['score'] = item['maximum_score'] + math.log1p(item['support'])
            result.append((path, item))
        return sorted(result, key=lambda x: (-x[1]['score'], x[0]))[:c['retrieval_candidate_limit']]

    def predict(self, question):
        found = self.retrieve(question)
        length = question['gap_length']
        paths = [p for p, _ in found]
        score_map = {p: item['score'] for p, item in found}
        if length == 1:
            seen = {p[0] for p in paths}
            ranking = [p[0] for p in paths] + [i for i in self.ranking if i not in seen]
            selected = [(i,) for i in ranking[:10]]
        else:
            ranking = None
            selected = list(paths[:5])
            fallback = super().predict(question)
            for path in fallback['paths']:
                p = tuple(path)
                if p not in selected and len(selected) < 5:
                    selected.append(p)
        scores = [score_map[p] if p in score_map else sum(float(self.logs[i]) for i in p) for p in selected]
        return {'paths': [list(p) for p in selected], 'scores': scores, 'single_ranking': ranking,
                'score_kind': 'retrieval_score_then_unigram_fallback_not_probability',
                'retrieval': {'candidate_count': len(paths), 'candidate_paths': [list(p) for p in paths],
                              'fallback_top1': selected[0] not in score_map,
                              'fallback_in_returned': sum(p not in score_map for p in selected),
                              'evidence': [{'path': list(p), **item} for p, item in found]}}

    def payload(self):
        value = super().payload()
        value['documents'] = self.documents
        return value


def fit(config, gate, documents):
    validate_config(config)
    counts = [0] * len(gate.targets)
    for row in documents:
        for target in row['targets']:
            if target >= 0:
                counts[target] += 1
    if config['experiment'] == 'B1':
        return Unigram(config, gate.target_input_ids, counts)
    if config['experiment'] == 'B6':
        return Retriever(config, gate.target_input_ids, counts, documents)
    tables = {'forward': defaultdict(Counter), 'backward': defaultdict(Counter)}
    model = Ngram(config, gate.target_input_ids, counts, tables, gate.input_ids['[LACUNA_N]'])
    for row in documents:
        tokens = row['tokens']
        for i, target in enumerate(row['targets']):
            if target < 0:
                continue
            for direction, history, marker in (('forward', tokens[:i], BOS),
                                                ('backward', tokens[i+1:][::-1], EOS)):
                context = model.context(history, marker)
                for order in range(1, len(context) + 1):
                    tables[direction][context[-order:]][target] += 1
    return model


def restore(payload):
    args = (payload['config'], payload['target_input_ids'], payload['counts'])
    name = payload['config']['experiment']
    if name == 'B1':
        return Unigram(*args)
    if name == 'B6':
        return Retriever(*args, payload['documents'])
    tables = {d: {tuple(h): dict(pairs) for h, pairs in entries} for d, entries in payload['tables'].items()}
    return Ngram(*args, tables, payload['lacuna_n'])
