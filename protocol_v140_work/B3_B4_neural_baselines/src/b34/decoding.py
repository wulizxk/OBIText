import numpy as np
import torch

from .data import pad_inputs
from .model import autocast, restricted_logits


def canonical_topk(scores, k):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not 0 < k <= len(scores):
        raise ValueError("Invalid top-k request")
    if k == len(scores):
        return np.argsort(-scores, kind="stable")
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    greater = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[: k - len(greater)]
    selected = np.concatenate((greater, tied))
    return selected[np.lexsort((selected, -scores[selected]))]


def beam_search(encoded, target_fill_ids, score_next, beam_width=10, returns=5):
    beams = [[(0.0, (), item["input_ids"][:])] for item in encoded]
    single_ranks = {}
    for step in range(max(len(item["positions"]) for item in encoded)):
        tasks = []
        for question_index, (item, paths) in enumerate(zip(encoded, beams)):
            if step < len(item["positions"]):
                for score, path, ids in paths:
                    tasks.append((question_index, score, path, ids, item["positions"][step]))
        scores = score_next([task[3] for task in tasks], [task[4] for task in tasks])
        if scores.shape != (len(tasks), len(target_fill_ids)) or not np.isfinite(scores).all():
            raise ValueError("Invalid candidate score matrix")
        candidates = {}
        for task, logp in zip(tasks, scores):
            question_index, base, path, ids, position = task
            if len(encoded[question_index]["positions"]) == 1:
                order = np.argsort(-logp, kind="stable")
                single_ranks[question_index] = order.tolist()
            else:
                order = canonical_topk(logp, beam_width)
            for target_index in order[:beam_width]:
                target_index = int(target_index)
                filled = ids[:]
                filled[position] = target_fill_ids[target_index]
                candidates.setdefault(question_index, []).append(
                    (base + float(logp[target_index]), path + (target_index,), filled)
                )
        for question_index, values in candidates.items():
            beams[question_index] = sorted(values, key=lambda item: (-item[0], item[1]))[:beam_width]
    result = []
    for question_index, paths in enumerate(beams):
        keep = beam_width if question_index in single_ranks else returns
        result.append(
            {
                "paths": [list(item[1]) for item in paths[:keep]],
                "scores": [item[0] for item in paths[:keep]],
                "single_ranking": single_ranks.get(question_index),
            }
        )
    return result


def predict_encoded_batch(model, codec, encoded, cfg, device, targets=None):
    if targets is None:
        targets = torch.tensor(codec.target_ids, dtype=torch.long, device=device)

    def score_next(inputs, positions):
        arrays = []
        for start in range(0, len(inputs), cfg["eval_forward_batch_size"]):
            values = inputs[start : start + cfg["eval_forward_batch_size"]]
            ids, attention = pad_inputs(values, codec.pad_id, cfg["pad_multiple"], device)
            owners = torch.arange(len(values), device=device)
            selected = torch.tensor(
                positions[start : start + cfg["eval_forward_batch_size"]],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad(), autocast(device, cfg):
                logits = restricted_logits(
                    model, cfg, ids, attention, owners, selected, targets
                )
            arrays.append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(arrays)

    return beam_search(
        encoded, codec.target_ids, score_next, cfg["beam_width"], cfg["return_sequences"]
    )
