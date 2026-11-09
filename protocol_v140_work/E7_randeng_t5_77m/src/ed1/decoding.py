"""Gold-free constrained encoder-decoder generation with reusable encoder states."""
import numpy as np
import torch

from .data import pad_inputs
from .model import autocast, decoder_hidden, project_hidden


def canonical_topk(scores, k):
    """Exact stable descending top-k; ties prefer canonical target order."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not 0 < k <= len(scores):
        raise ValueError('Invalid top-k request')
    if k == len(scores):
        return np.argsort(-scores, kind='stable')
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    greater = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:k - len(greater)]
    selected = np.concatenate((greater, tied))
    return selected[np.lexsort((selected, -scores[selected]))]


def autoregressive_beam_search(encoded, target_native_ids, end_gap_id, score_next,
                               score_end=None, beam_width=10, returns=5,
                               single_returns=10):
    """Generate exactly the supplied gap length from an opaque initial path.

    ``score_next`` returns normalized log-probabilities over only the 4,116
    legal character targets. ``score_end`` optionally returns the log
    probability of <END_GAP> after a complete multi-character candidate.
    Gold answers are never supplied to either callback.
    """
    if not encoded:
        return []
    if len(target_native_ids) < max(beam_width, single_returns):
        raise ValueError('Prediction target space is smaller than requested top-k')
    beams = [[(0.0, (), q['input_ids'][:])] for q in encoded]
    single_rankings = {}
    max_length = max(q['gap_length'] for q in encoded)
    for step in range(max_length):
        tasks = []
        for qi, (question, paths) in enumerate(zip(encoded, beams)):
            if step < question['gap_length']:
                for score, path, ids in paths:
                    tasks.append((qi, score, path, ids))
        if not tasks:
            continue
        log_probabilities = score_next([task[3] for task in tasks])
        if (log_probabilities.shape != (len(tasks), len(target_native_ids))
                or not np.isfinite(log_probabilities).all()):
            raise ValueError('Invalid decoder character-score matrix')
        candidates = {}
        for task, logp in zip(tasks, log_probabilities):
            qi, base, path, ids = task
            if encoded[qi]['gap_length'] == 1:
                order = np.argsort(-logp, kind='stable')
                single_rankings[qi] = order.tolist()
            else:
                order = canonical_topk(logp, beam_width)
            for target_index in order[:beam_width]:
                target_index = int(target_index)
                candidates.setdefault(qi, []).append(
                    (base + float(logp[target_index]), path + (target_index,),
                     ids + [target_native_ids[target_index]])
                )
        for qi, candidates_q in candidates.items():
            beams[qi] = sorted(candidates_q, key=lambda item: (-item[0], item[1]))[:beam_width]

    # The training target ends with <END_GAP>. For multi-character candidates,
    # its likelihood is a legitimate sequence-ranking signal. Single-character
    # ranking deliberately remains the direct full 4,116-character ranking.
    if score_end is not None:
        end_tasks, owners = [], []
        for qi, (question, paths) in enumerate(zip(encoded, beams)):
            if question['gap_length'] > 1:
                for _, _, ids in paths:
                    owners.append(qi)
                    end_tasks.append(ids)
        if end_tasks:
            end_scores = np.asarray(score_end(end_tasks))
            if end_scores.shape != (len(end_tasks),) or not np.isfinite(end_scores).all():
                raise ValueError('Invalid <END_GAP> score vector')
            cursor = 0
            for qi, question in enumerate(encoded):
                if question['gap_length'] == 1:
                    continue
                rescored = []
                for score, path, ids in beams[qi]:
                    if owners[cursor] != qi:
                        raise AssertionError('Internal end-score owner mismatch')
                    rescored.append((score + float(end_scores[cursor]), path, ids))
                    cursor += 1
                beams[qi] = sorted(rescored, key=lambda item: (-item[0], item[1]))[:beam_width]
            if cursor != len(end_scores):
                raise AssertionError('Unused <END_GAP> scores')

    output = []
    for qi, paths in enumerate(beams):
        keep = single_returns if qi in single_rankings else returns
        output.append({
            'paths': [list(item[1]) for item in paths[:keep]],
            'scores': [item[0] for item in paths[:keep]],
            'single_ranking': single_rankings.get(qi),
            'score_kind': ('seq2seq_direct_character_log_probability'
                           if qi in single_rankings
                           else 'seq2seq_autoregressive_character_plus_end_log_probability'),
            'end_gap_id': end_gap_id,
        })
    return output


def predict_encoded_batch(model, codec, encoded, cfg, device, targets=None, candidates=None):
    """Encode each public context once; decode only START + generated answers.

    Beam path element 0 is an owner index, NOT an input token. It selects the
    corresponding encoder states at every branch/reorder and is stripped before
    decoder invocation. The decoder never receives gold or the encoder tokens.
    """
    if not encoded:
        return []
    model.eval()
    if targets is None:
        targets = torch.tensor(codec.target_ids, dtype=torch.long, device=device)
    if candidates is None:
        candidates = torch.tensor(codec.candidate_ids, dtype=torch.long, device=device)

    encoder_ids, encoder_attention = pad_inputs(
        [q['input_ids'] for q in encoded], codec.tokenizer.pad_token_id, cfg['pad_multiple'], device)
    with torch.no_grad(), autocast(device, cfg):
        encoder_states = model.encoder(input_ids=encoder_ids, attention_mask=encoder_attention,
                                       use_cache=False, return_dict=True).last_hidden_state

    def log_probabilities_for(inputs, native_ids):
        arrays = []
        forward_batch = cfg['eval_forward_batch_size']
        for start in range(0, len(inputs), forward_batch):
            items = inputs[start:start + forward_batch]
            source_owners = torch.tensor([item[0] for item in items], dtype=torch.long, device=device)
            decoder_inputs = [[codec.decoder_start_id] + item[1:] for item in items]
            ids, attention = pad_inputs(decoder_inputs, codec.tokenizer.pad_token_id, cfg['pad_multiple'], device)
            owners = torch.arange(len(items), dtype=torch.long, device=device)
            positions = torch.tensor([len(item) - 1 for item in items], dtype=torch.long, device=device)
            with torch.no_grad(), autocast(device, cfg):
                hidden = decoder_hidden(model, ids, attention,
                                        encoder_states.index_select(0, source_owners),
                                        encoder_attention.index_select(0, source_owners))
                logits = project_hidden(model, hidden[owners, positions], native_ids)
            arrays.append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(arrays, axis=0)

    def score_next(inputs):
        return log_probabilities_for(inputs, targets)

    def score_end(inputs):
        return log_probabilities_for(inputs, candidates)[:, -1]

    return autoregressive_beam_search(
        [{'input_ids': [i], 'gap_length': q['gap_length']} for i, q in enumerate(encoded)],
        codec.target_ids, codec.end_gap_id, score_next, score_end,
        cfg['beam_width'], cfg['return_sequences'], cfg['single_mask_return_sequences'],
    )


def predict_batch(model, codec, questions, cfg, device):
    return predict_encoded_batch(
        model, codec, [codec.encode(question) for question in questions], cfg, device,
    )
