from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from .io import read, sha


SUPPORTED_WEIGHT_FILES = frozenset({'model.safetensors', 'pytorch_model.bin'})


def autocast(device, cfg):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' and cfg['precision'] == 'fp16' else nullcontext()


def verify_weights(folder, cfg):
    folder = Path(folder)
    provenance = read(folder / 'provenance.json')
    if provenance['repo'] != cfg['model_repo'] or provenance['revision'] != cfg['model_revision']:
        raise ValueError('Wrong pretrained checkpoint revision')
    weight_file = provenance.get('weight_file')
    if weight_file not in SUPPORTED_WEIGHT_FILES:
        raise ValueError('Pretrained provenance does not declare one supported weight file')
    hashes = provenance.get('hashes')
    required = {'config.json', 'vocab.txt', weight_file}
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise ValueError('Pretrained provenance must hash config, vocabulary and the declared weight file')
    recognized = {name for name in SUPPORTED_WEIGHT_FILES if (folder / name).is_file()}
    indexes = list(folder.glob('*.safetensors.index.json')) + list(folder.glob('pytorch_model*.index.json'))
    if recognized != {weight_file} or indexes:
        raise ValueError('Pretrained directory contains a missing, competing or sharded weight artifact')
    for name, expected in hashes.items():
        if sha(folder / name) != expected:
            raise ValueError('Pretrained file checksum mismatch: ' + name)
    if sha(folder / 'config.json') != cfg['native_config_sha256']:
        raise ValueError('Native architecture config differs from the frozen revision')
    if sha(folder / 'vocab.txt') != cfg['native_vocab_sha256']:
        raise ValueError('Native checkpoint vocabulary differs from the frozen revision')
    return sha(folder / 'provenance.json')


def load_pretrained(folder, codec, cfg):
    from transformers import GPT2LMHeadModel
    verify_weights(folder, cfg)
    model, info = GPT2LMHeadModel.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False, weights_only=True,
        output_loading_info=True, attn_implementation='eager',
        use_safetensors=read(Path(folder) / 'provenance.json')['weight_file'].endswith('.safetensors'),
    )
    # The language-model head is tied to the input embedding. Transformers may
    # report that alias as absent when a checkpoint serializes only one copy.
    missing = set(info['missing_keys']) - {'lm_head.weight'}
    unexpected = set(info['unexpected_keys'])
    if missing or unexpected or info['mismatched_keys'] or info['error_msgs']:
        raise RuntimeError('Incomplete pretrained causal-LM weights: ' + repr(info))
    native = model.config
    if (native.model_type, native.vocab_size, native.n_layer, native.n_embd,
            native.n_head, native.n_positions) != ('gpt2', 25370, 12, 768, 12, 1024):
        raise ValueError('Unexpected GPT2-Chinese-Ancient backbone')
    old = model.get_input_embeddings().weight.detach().clone()
    model.resize_token_embeddings(codec.vocab_size, mean_resizing=False)
    model.tie_weights()
    assert torch.equal(old, model.get_input_embeddings().weight[:len(old)])
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, info


def restricted_logits(model, input_ids, attention_mask, owners, positions, candidate_ids):
    hidden = model.transformer(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
    ).last_hidden_state
    selected = hidden[owners, positions]
    decoder = model.get_output_embeddings()
    # Character prediction is restricted to the shared targets; the single
    # additional class is <END_GAP>, which is supervised after the known span.
    return F.linear(selected, decoder.weight.index_select(0, candidate_ids), None)


def per_sample_losses(model, batch, candidate_ids):
    logits = restricted_logits(model, batch['input_ids'], batch['attention_mask'],
                               batch['owners'], batch['positions'], candidate_ids)
    token_loss = F.cross_entropy(logits.float(), batch['targets'], reduction='none')
    totals = torch.zeros(len(batch['counts']), device=logits.device).scatter_add(0, batch['owners'], token_loss)
    return totals / batch['counts']


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    normalization_parameters = {
        id(parameter)
        for module in model.modules() if isinstance(module, torch.nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith('bias') or id(parameter) in normalization_parameters else decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=cfg['learning_rate'], betas=tuple(cfg['adam_betas']), eps=cfg['adam_epsilon'])
