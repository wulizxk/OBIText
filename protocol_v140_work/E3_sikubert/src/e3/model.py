from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from .io import read, sha


def autocast(device, cfg):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' and cfg['precision'] == 'fp16' else nullcontext()


def verify_weights(folder, cfg):
    folder = Path(folder)
    provenance = read(folder / 'provenance.json')
    if provenance['repo'] != cfg['model_repo'] or provenance['revision'] != cfg['model_revision']:
        raise ValueError('Wrong pretrained checkpoint revision')
    for name, expected in provenance['hashes'].items():
        if sha(folder / name) != expected:
            raise ValueError('Pretrained file checksum mismatch: ' + name)
    if sha(folder / 'config.json') != cfg['native_config_sha256']:
        raise ValueError('Native architecture config differs from the frozen revision')
    if sha(folder / 'vocab.txt') != cfg['native_vocab_sha256']:
        raise ValueError('Native checkpoint vocabulary differs from the frozen revision')
    return sha(folder / 'provenance.json')


def load_pretrained(folder, codec, cfg):
    from transformers import BertForMaskedLM
    verify_weights(folder, cfg)
    model, info = BertForMaskedLM.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                weights_only=True, output_loading_info=True, attn_implementation='eager')
    harmless_aliases = {'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias'}
    # BertForMaskedLM has no active pooler. Some standard BERT checkpoints keep
    # these two unused base-model tensors; no other unexpected key is accepted.
    harmless_unexpected = {'bert.pooler.dense.weight', 'bert.pooler.dense.bias'}
    missing = set(info['missing_keys']) - harmless_aliases
    unexpected = set(info['unexpected_keys']) - harmless_unexpected
    if missing or unexpected or info['mismatched_keys'] or info['error_msgs']:
        raise RuntimeError('Incomplete pretrained MLM weights: ' + repr(info))
    native = model.config
    if (native.model_type, native.vocab_size, native.num_hidden_layers, native.hidden_size,
            native.num_attention_heads, native.intermediate_size) != ('bert', 29791, 12, 768, 12, 3072):
        raise ValueError('Unexpected SikuBERT backbone')
    old = model.get_input_embeddings().weight.detach().clone()
    old_bias = model.get_output_embeddings().bias.detach().clone()
    model.resize_token_embeddings(codec.vocab_size, mean_resizing=False)
    model.tie_weights()
    assert torch.equal(old, model.get_input_embeddings().weight[:len(old)])
    assert torch.equal(old_bias, model.get_output_embeddings().bias[:len(old_bias)])
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, info


def restricted_logits(model, input_ids, attention_mask, owners, positions, target_ids):
    hidden = model.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
    selected = hidden[owners, positions]
    transformed = model.cls.predictions.transform(selected)
    decoder = model.get_output_embeddings()
    # This is exactly the pretrained flat MLM head restricted to the shared targets.
    # It avoids materializing [batch, all_positions, 33195] logits.
    return F.linear(transformed, decoder.weight.index_select(0, target_ids),
                    decoder.bias.index_select(0, target_ids))


def per_sample_losses(model, batch, target_ids):
    logits = restricted_logits(model, batch['input_ids'], batch['attention_mask'],
                               batch['owners'], batch['positions'], target_ids)
    token_loss = F.cross_entropy(logits.float(), batch['targets'], reduction='none')
    totals = torch.zeros(len(batch['counts']), device=logits.device).scatter_add(0, batch['owners'], token_loss)
    return totals / batch['counts']


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith('bias') or 'LayerNorm' in name else decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=cfg['learning_rate'], betas=tuple(cfg['adam_betas']), eps=cfg['adam_epsilon'])
