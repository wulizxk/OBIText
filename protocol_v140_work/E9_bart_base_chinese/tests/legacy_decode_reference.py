"""Frozen pre-fix END-transfer reference, for bounded equivalence tests only."""
import numpy as np
import torch
from ed3.data import pad_inputs
from ed3.decoding import autoregressive_beam_search
from ed3.model import autocast, decoder_hidden, project_hidden

def old_predict_encoded_batch(model, codec, encoded, cfg, device, targets=None, candidates=None, executor=None):
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
    def encode(ids, attention):
        return model.get_encoder()(input_ids=ids, attention_mask=attention,
                                   return_dict=True).last_hidden_state
    with torch.no_grad(), autocast(device, cfg):
        encoder_states = (executor.call('encoder', encode, (encoder_ids, encoder_attention))
                          if executor is not None else encode(encoder_ids, encoder_attention))

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
                def decode(a, b, c, d):
                    return decoder_hidden(model, a, b, c, d)
                arguments = (ids, attention, encoder_states.index_select(0, source_owners),
                             encoder_attention.index_select(0, source_owners))
                hidden = (executor.call('decoder', decode, arguments)
                          if executor is not None else decode(*arguments))
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
