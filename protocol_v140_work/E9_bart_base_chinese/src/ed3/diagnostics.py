"""TRAIN-only launch diagnostics and lossless-generation comparison helpers.

No helper here opens Dev/Test or starts a job. Callers own resource/identity gates.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
import math
import time

import numpy as np
import torch

from .decoding import predict_encoded_batch
from .metrics import aggregate, score_prediction
from .model import autocast, restricted_logits
from .projection_cache import EvaluationProjectionCache


class GenerationProfile:
    """Opt-in synchronized diagnostic timing; never enabled in formal evaluation."""
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds = defaultdict(float)
        self.calls = Counter()
        self.shapes = {'encoder': [0, 0], 'decoder': [0, 0]}

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        self.sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[name] += time.perf_counter() - started
            self.calls[name] += 1

    def observe(self, kind, shape):
        self.shapes[kind] = [max(a, int(b)) for a, b in zip(self.shapes[kind], shape)]

    def report(self):
        total = self.seconds.get('total', 0.)
        # Includes input construction, CPU beam sorting, dispatch and bookkeeping;
        # not falsely labelled a pure CPU-kernel duration.
        residual = max(0., total - sum(v for k, v in self.seconds.items() if k != 'total'))
        return {'seconds': dict(self.seconds), 'calls': dict(self.calls),
                'max_encoder_shape': self.shapes['encoder'], 'max_decoder_shape': self.shapes['decoder'],
                'python_data_beam_unattributed_seconds': residual,
                'timing_scope': 'separate synchronized profile, not uninstrumented throughput'}


def generate_public(model, codec, public, encoded, cfg, device, profile=None, targets=None, candidates=None,
                    use_projection_cache=False, projection_cache=None):
    """Same bucketing as formal evaluation; public input ONLY; restore original order."""
    if len(public) != len(encoded) or any('labels' in row for row in encoded):
        raise ValueError('Expected matching public-only encodings')
    if targets is None:
        targets = torch.tensor(codec.target_ids, dtype=torch.long, device=device)
    if candidates is None:
        candidates = torch.tensor(codec.candidate_ids, dtype=torch.long, device=device)
    output = [None] * len(public)
    if projection_cache is not None and not use_projection_cache:
        raise ValueError('External projection cache requires explicit cached diagnostic mode')
    context = (EvaluationProjectionCache(model) if use_projection_cache and projection_cache is None
               else nullcontext(projection_cache))
    with context as cache:
        for length in sorted({q['gap_length'] for q in public}):
            group = sorted((i for i, q in enumerate(public) if q['gap_length'] == length),
                           key=lambda i: (len(encoded[i]['input_ids']), i))
            for start in range(0, len(group), cfg['eval_question_batch_size']):
                indices = group[start:start + cfg['eval_question_batch_size']]
                predictions = predict_encoded_batch(model, codec, [encoded[i] for i in indices], cfg, device,
                                                    targets, candidates, diagnostics=profile, projection_cache=cache)
                if len(predictions) != len(indices):
                    raise ValueError('Generation returned the wrong number of predictions')
                for i, prediction in zip(indices, predictions):
                    output[i] = prediction
    return output


def compare_predictions(reference, candidate, public, answers, units):
    """Exact means exact, including scores and EVERY single-character rank.

    Answers are used only after both prediction lists exist. All diagnostic
    callers supply TRAIN answers, never Test. Different BF16 batch shapes may
    legitimately differ: report CHANGED, not a misleading lossless PASS.
    """
    count = len(public)
    if len(reference) != count or len(candidate) != count:
        return {'status': 'FAILED', 'reason': 'question_count_mismatch', 'questions': count,
                'reference_count': len(reference), 'candidate_count': len(candidate)}
    differences = Counter()
    max_error = 0.
    indices = []
    for index, (a, b, q) in enumerate(zip(reference, candidate, public)):
        for row in (a, b):
            expected = 10 if q['gap_length'] == 1 else 5
            if (len(row['paths']) != expected or len(row['scores']) != expected
                    or len({tuple(p) for p in row['paths']}) != expected
                    or any(len(p) != q['gap_length'] or any(type(t) is not int or not 0 <= t < len(units)
                                                          for t in p) for p in row['paths'])
                    or any(type(s) not in (float, int) or not math.isfinite(s) for s in row['scores'])):
                raise ValueError('Invalid diagnostic prediction schema')
            rank = row['single_ranking']
            if (q['gap_length'] == 1 and (not isinstance(rank, list) or len(rank) != len(units)
                                         or any(type(t) is not int for t in rank)
                                         or sorted(rank) != list(range(len(units))))) or (
                    q['gap_length'] != 1 and rank is not None):
                raise ValueError('Invalid full single-character ranking')
        before = sum(differences.values())
        for key in ('paths', 'scores', 'single_ranking', 'score_kind', 'end_gap_id'):
            if a.get(key) != b.get(key):
                differences[key] += 1
        max_error = max(max_error, max(abs(float(x)-float(y)) for x, y in zip(a['scores'], b['scores'])))
        if sum(differences.values()) != before and len(indices) < 20:
            indices.append(index)
    metrics = [aggregate([score_prediction(q, p, answers[q['example_id']], units)
                          for q, p in zip(public, predictions)]) for predictions in (reference, candidate)]
    if metrics[0] != metrics[1]:
        differences['metrics'] += 1
    return {'status': 'IDENTICAL' if not differences else 'CHANGED', 'questions': count,
            'different_question_counts_by_field': dict(differences), 'first_difference_indices': indices,
            'max_candidate_score_abs_error': max_error, 'score_atol': 0., 'score_rtol': 0.,
            'reference_metrics': metrics[0], 'candidate_metrics': metrics[1]}


def require_identical(comparison):
    if comparison.get('status') != 'IDENTICAL':
        raise ValueError('Generation comparison is not lossless: ' + repr(comparison))


def train_fixtures(sampler, codec, cfg, per_length=32, include_stress=True):
    """Balanced real TRAIN questions, both boundaries and optional TRAIN-derived max shape."""
    if type(per_length) is not int or not 1 <= per_length <= 128:
        raise ValueError('Diagnostic fixture must remain bounded')
    selected = []
    for length in range(1, 6):
        for item in sampler.pools[length][:per_length]:
            row, start = sampler.rows[item['source_id']], item['starts'][0]
            selected.append({'source_id': item['source_id'], 'group_id': row['group_id'], 'gap_length': length,
                             'prefix_units': row['model_tokens'][:start], 'suffix_units': row['model_tokens'][start+length:],
                             'gold_units': row['model_tokens'][start:start+length]})
    if len(selected) != per_length * 5:
        raise ValueError('Too few TRAIN fixtures')
    for side in ('initial', 'final'):
        for item in sampler.pools[1]:
            row = sampler.rows[item['source_id']]
            start = 0 if side == 'initial' else len(row['model_tokens'])-1
            if start in item['starts']:
                selected.append({'source_id': item['source_id'], 'group_id': row['group_id'], 'gap_length': 1,
                                 'prefix_units': row['model_tokens'][:start], 'suffix_units': row['model_tokens'][start+1:],
                                 'gold_units': row['model_tokens'][start:start+1], 'boundary_side': side})
                break
        else:
            raise ValueError('TRAIN fixtures lack a boundary side')
    if include_stress:
        token = selected[0]['gold_units'][0]
        base = {'source_id': 'TRAIN_DERIVED_SYNTHETIC', 'group_id': 'DIAGNOSTIC', 'gap_length': 5,
                'prefix_units': [token], 'suffix_units': [], 'gold_units': [token]*5}
        overhead = len(codec.encode(base)['input_ids'])-1
        for i in range(cfg['eval_question_batch_size']):
            # Both one-sided and bilateral max-length inputs in the SAME batch.
            left = cfg['max_input_length']-overhead if i % 2 == 0 else 1
            selected.append(dict(base, prefix_units=[token]*left,
                                 suffix_units=[token]*(cfg['max_input_length']-overhead-left)))
    public, answers = [], {}
    for i, item in enumerate(selected):
        q = dict(item, example_id=f'TRAIN-DIAGNOSTIC-{i}')
        answers[q['example_id']] = q.pop('gold_units')
        public.append(q)
    return public, answers


def native_forward_reference(model, codec, questions, cfg, device):
    """Independent native BART.forward vs restricted implementation; no updates."""
    model.eval()
    batch = codec.collate(questions, device)
    candidates = torch.tensor(codec.candidate_ids, dtype=torch.long, device=device)
    native_dtype, restricted_dtype = {}, {}
    handle = model.lm_head.register_forward_hook(lambda _, __, output: native_dtype.update(linear=str(output.dtype)))
    try:
        with torch.no_grad(), autocast(device, cfg):
            result = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           decoder_input_ids=batch['decoder_input_ids'], decoder_attention_mask=batch['decoder_attention_mask'],
                           use_cache=False, return_dict=True).logits
            expected = result[batch['owners'], batch['positions']][:, candidates]
            actual = restricted_logits(model, batch['input_ids'], batch['attention_mask'], batch['owners'], batch['positions'],
                                       candidates, batch['decoder_input_ids'], batch['decoder_attention_mask'], restricted_dtype)
    finally:
        handle.remove()
    is_cuda = torch.device(device).type == 'cuda'
    atol, rtol = (.05, .01) if is_cuda else (1e-5, 1e-4)
    if not bool(torch.isfinite(expected).all() & torch.isfinite(actual).all()):
        raise FloatingPointError('Non-finite native/restricted diagnostic logits')
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    expected_linear = 'torch.bfloat16' if is_cuda else 'torch.float32'
    if (native_dtype.get('linear') != expected_linear or restricted_dtype.get('linear') != expected_linear
            or restricted_dtype.get('bias') != 'torch.float32' or restricted_dtype.get('result') != 'torch.float32'
            or result.dtype != torch.float32):
        raise ValueError('Native/restricted BART projection dtype path changed')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Diagnostic model masters are not FP32')
    relative_bound = atol + rtol * expected.abs()
    return {'status': 'PASS', 'oracle': 'BartForConditionalGeneration.forward', 'device': str(device),
            'questions': len(questions), 'gap_lengths': sorted({q['gap_length'] for q in questions}),
            'atol': atol, 'rtol': rtol, 'max_abs_error': float((actual-expected).abs().max()),
            'max_error_to_allowed_ratio': float(((actual-expected).abs()/relative_bound).max()),
            'native_linear_dtype': native_dtype['linear'], 'native_logits_dtype': str(result.dtype),
            'restricted_dtypes': restricted_dtype, 'parameter_dtype': 'torch.float32',
            'test_answers_used': False, 'finite': True,
            'note': 'Numerical tolerance is NOT a claim of bitwise ranking equivalence between different GEMM shapes.'}


def validate_launch_diagnostics(report, cfg):
    """Only real CUDA evidence covering native numerics AND formal shapes may launch."""
    if report.get('projection_cache_reference') != 'uncached_public_reference_vs_formal_cached_evaluation':
        raise ValueError('Missing uncached-versus-cached full-generation evidence')
    for key in ('evaluation_execution_before_reload', 'evaluation_execution_after_reload'):
        cache = report.get(key, {}).get('output_projection_cache', {})
        if (cache.get('enabled') is not True or cache.get('closed') is not True
                or cache.get('misses') != 2 or type(cache.get('hits')) is not int or cache['hits'] <= 0
                or cache.get('peak_bytes') != (4116+4117)*(768+1)*4):
            raise ValueError('Missing bounded FP32 evaluation projection-cache evidence: ' + key)
    native = report.get('native_forward_reference', {})
    if (native.get('status') != 'PASS' or native.get('oracle') != 'BartForConditionalGeneration.forward'
            or native.get('device') != 'cuda:0' or native.get('gap_lengths') != [1, 2, 3, 4, 5]
            or native.get('finite') is not True or native.get('test_answers_used') is not False
            or native.get('questions') != 5
            or native.get('atol') != .05 or native.get('rtol') != .01
            or type(native.get('max_error_to_allowed_ratio')) not in (int, float)
            or not 0 <= native['max_error_to_allowed_ratio'] <= 1
            or type(native.get('max_abs_error')) not in (int, float) or not math.isfinite(native['max_abs_error'])
            or native['max_abs_error'] < 0
            or native.get('native_linear_dtype') != 'torch.bfloat16' or native.get('native_logits_dtype') != 'torch.float32'
            or native.get('restricted_dtypes') != {'linear': 'torch.bfloat16', 'bias': 'torch.float32', 'result': 'torch.float32'}
            or native.get('parameter_dtype') != 'torch.float32'):
        raise ValueError('Missing/incompatible independent native CUDA BF16 evidence')
    shape = report.get('generation_shape_evidence', {})
    if (shape.get('max_encoder_shape') != [cfg['eval_question_batch_size'], cfg['max_input_length']]
            or shape.get('max_decoder_shape') != [cfg['eval_forward_batch_size'], 8]
            or shape.get('gap_lengths') != [1, 2, 3, 4, 5]
            or shape.get('boundary_sides') != ['initial', 'final']
            or shape.get('test_answers_used') is not False or shape.get('maximum_gap_length') != 5
            or shape.get('question_count') != 5 * 32 + 2 + cfg['eval_question_batch_size']):
        raise ValueError('Smoke did not cover the formal Q32/F64/max-input generation shapes')
    for name in ('full_prediction_eager_comparison', 'full_prediction_checkpoint_comparison'):
        value = report.get(name, {})
        if (value.get('status') != 'IDENTICAL' or value.get('questions') != shape['question_count']
                or value.get('different_question_counts_by_field') != {}
                or value.get('max_candidate_score_abs_error') != 0
                or value.get('score_atol') != 0 or value.get('score_rtol') != 0):
            raise ValueError('Missing full candidate/score/ranking/metric equality: ' + name)
