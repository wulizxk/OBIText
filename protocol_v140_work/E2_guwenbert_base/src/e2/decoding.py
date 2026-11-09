import numpy as np
import torch

from .data import pad_inputs
from .model import autocast, restricted_logits


def canonical_topk(scores, k):
    """Return the exact stable descending top-k without sorting the full vocabulary."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not 0 < k <= len(scores):
        raise ValueError('Invalid top-k request')
    if k == len(scores):
        return np.argsort(-scores, kind='stable')
    # Partition finds the cutoff in O(V). Include lower canonical indices first
    # when multiple candidates tie at that cutoff, then sort only the selected k.
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    greater = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:k - len(greater)]
    selected = np.concatenate((greater, tied))
    return selected[np.lexsort((selected, -scores[selected]))]


def beam_search(encoded, target_native_ids, score_next, beam_width=10, returns=5):
    """Gold-free, batched, deterministic left-to-right pseudo-log-probability beam.

    score_next receives only input IDs and the next model position. Ties are
    resolved by canonical target order, never model-specific token IDs.
    Single-mask full rankings are returned privately for exact, untruncated MRR.
    """
    beams = [[(0.0, (), q['input_ids'][:])] for q in encoded]
    single_ranks = {}
    for step in range(max(len(q['positions']) for q in encoded)):
        tasks = []
        for qi, (q, paths) in enumerate(zip(encoded, beams)):
            if step < len(q['positions']):
                for score, path, ids in paths:
                    tasks.append((qi, score, path, ids, q['positions'][step]))
        probabilities = score_next([t[3] for t in tasks], [t[4] for t in tasks])
        if probabilities.shape != (len(tasks), len(target_native_ids)) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid candidate score matrix')
        candidates = {}
        for task, logp in zip(tasks, probabilities):
            qi, base, path, ids, position = task
            if len(encoded[qi]['positions']) == 1:
                # Single-mask MRR requires the exact full-vocabulary rank.
                order = np.argsort(-logp, kind='stable')
                single_ranks[qi] = order.tolist()
            else:
                order = canonical_topk(logp, beam_width)
            for index in order[:beam_width]:
                index = int(index)
                filled = ids[:]
                filled[position] = target_native_ids[index]
                candidates.setdefault(qi, []).append((base + float(logp[index]), path + (index,), filled))
        for qi, candidates_q in candidates.items():
            beams[qi] = sorted(candidates_q, key=lambda item: (-item[0], item[1]))[:beam_width]
    output = []
    for qi, paths in enumerate(beams):
        keep = beam_width if qi in single_ranks else returns
        output.append({'paths': [list(p[1]) for p in paths[:keep]],
                       'scores': [p[0] for p in paths[:keep]], 'single_ranking': single_ranks.get(qi)})
    return output


def predict_encoded_batch(model, codec, encoded, cfg, device, targets=None):
    """Decode already-tokenized questions; useful for repeatedly evaluated Dev."""
    if targets is None:
        targets = torch.tensor(codec.target_ids, dtype=torch.long, device=device)

    def score_next(inputs, positions):
        arrays = []
        step = cfg['eval_forward_batch_size']
        for start in range(0, len(inputs), step):
            items = inputs[start:start + step]
            ids, attention = pad_inputs(items, codec.tokenizer.pad_token_id, cfg['pad_multiple'], device)
            owners = torch.arange(len(items), device=device)
            selected = torch.tensor(positions[start:start + step], dtype=torch.long, device=device)
            with torch.no_grad(), autocast(device, cfg):
                logits = restricted_logits(model, ids, attention, owners, selected, targets)
            arrays.append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(arrays)

    return beam_search(encoded, codec.target_ids, score_next, cfg['beam_width'], cfg['return_sequences'])


def predict_batch(model, codec, questions, cfg, device):
    return predict_encoded_batch(model, codec, [codec.encode(q) for q in questions], cfg, device)
