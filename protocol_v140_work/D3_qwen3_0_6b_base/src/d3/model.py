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
    required = {'config.json', weight_file}
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise ValueError('Pretrained provenance must hash config and the declared weight file')
    # The deployment contract intentionally permits exactly three root files.
    # Hugging Face's local-dir cache may live below ``.cache/``; it is not a
    # loadable root-level model artifact and therefore does not compete here.
    root_files = {path.name for path in folder.iterdir() if path.is_file()}
    expected_root_files = {'provenance.json', 'config.json', weight_file}
    if root_files != expected_root_files:
        raise ValueError('Pretrained directory contains a missing, competing or extra root artifact: '
                         f'{sorted(root_files)!r}')
    for name, expected in hashes.items():
        if sha(folder / name) != expected:
            raise ValueError('Pretrained file checksum mismatch: ' + name)
    if sha(folder / 'config.json') != cfg['native_config_sha256']:
        raise ValueError('Native architecture config differs from the frozen revision')
    if sha(folder / weight_file) != cfg['native_weight_sha256']:
        raise ValueError('Native checkpoint weights differ from the frozen revision')
    return sha(folder / 'provenance.json')


def resize_and_initialize_embeddings(model, codec, cfg):
    """Preserve genuinely native tokens and initialize every tokenizer-added ID.

    Qwen3's config has 267 more embedding rows than its tokenizer exposes.
    The frozen tokenizer assigns new OBI context units to those otherwise
    unreachable IDs.  They must not inherit arbitrary pretrained negative-class
    weights and be mistaken for pretrained semantic representations.
    """
    embeddings = model.get_input_embeddings()
    native_config_rows = embeddings.weight.shape[0]
    native_tokenizer_rows = codec.mapping['original_vocab_size']
    if native_tokenizer_rows != cfg['tokenizer_new_id_start']:
        raise ValueError('Tokenizer-added embedding boundary differs from the frozen protocol')
    if native_config_rows != codec.mapping['original_config_vocab_size']:
        raise ValueError('Native model/tokenizer config vocabulary sizes disagree')
    if cfg['tokenizer_added_embedding_initialization'] != (
            'normal_mean_0_std_initializer_range_all_ids_ge_start'):
        raise ValueError('Unsupported tokenizer-added embedding initialization')
    preserved = embeddings.weight[:native_tokenizer_rows].detach().clone()
    model.resize_token_embeddings(codec.vocab_size, mean_resizing=False)
    embeddings = model.get_input_embeddings()
    with torch.no_grad():
        embeddings.weight[native_tokenizer_rows:].normal_(
            mean=0.0, std=model.config.initializer_range,
        )
    model.tie_weights()
    if not torch.equal(preserved, embeddings.weight[:native_tokenizer_rows]):
        raise RuntimeError('Resizing changed a native tokenizer embedding')
    if embeddings.weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError('Qwen3 input/output embeddings are not tied after resize')
    return {
        'native_tokenizer_rows_preserved': native_tokenizer_rows,
        'native_config_reserved_rows_reinitialized': native_config_rows - native_tokenizer_rows,
        'total_tokenizer_added_rows_initialized': codec.vocab_size - native_tokenizer_rows,
        'initialization': cfg['tokenizer_added_embedding_initialization'],
    }


def load_pretrained(folder, codec, cfg):
    from transformers import Qwen3ForCausalLM
    verify_weights(folder, cfg)
    model, info = Qwen3ForCausalLM.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False, weights_only=True,
        output_loading_info=True, attn_implementation='eager',
        dtype=torch.float32,
        use_safetensors=read(Path(folder) / 'provenance.json')['weight_file'].endswith('.safetensors'),
    )
    # The language-model head is tied to the input embedding. Transformers may
    # report that alias as absent when a checkpoint serializes only one copy.
    missing = set(info['missing_keys']) - {'lm_head.weight'}
    unexpected = set(info['unexpected_keys'])
    if missing or unexpected or info['mismatched_keys'] or info['error_msgs']:
        raise RuntimeError('Incomplete pretrained causal-LM weights: ' + repr(info))
    native = model.config
    actual = (native.model_type, native.vocab_size, native.num_hidden_layers,
              native.hidden_size, native.num_attention_heads,
              native.num_key_value_heads, native.intermediate_size,
              native.head_dim, native.max_position_embeddings,
              native.tie_word_embeddings)
    expected = ('qwen3', 151936, 28, 1024, 16, 8, 3072, 128, 32768, True)
    if actual != expected:
        raise ValueError(f'Unexpected Qwen3-0.6B-Base backbone: {actual!r}')
    info['tokenizer_embedding_initialization'] = resize_and_initialize_embeddings(model, codec, cfg)
    model.config.use_cache = False
    if cfg['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for p in model.parameters():
        p.requires_grad_(True)
    return model, info


def restricted_logits(model, input_ids, attention_mask, owners, positions, candidate_ids):
    hidden = model.model(
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
        for module in model.modules() if module.__class__.__name__.endswith('Norm')
        for parameter in module.parameters(recurse=False)
    }
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith('bias') or id(parameter) in normalization_parameters else decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=cfg['learning_rate'], betas=tuple(cfg['adam_betas']),
                            eps=cfg['adam_epsilon'], foreach=cfg['adamw_foreach'], fused=False)
