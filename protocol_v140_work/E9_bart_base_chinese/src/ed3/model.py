from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .io import read, sha
from .runtime import effective_precision, require_cuda_precision


SUPPORTED_WEIGHT_FILES = frozenset({'model.safetensors'})

# Semantic fields, not merely tensor dimensions: changing heads or dropout does
# not change state_dict shapes. These must be checked BEFORE restoring a model.
FORMAL_MODEL_CONTRACT = {
    'model_type': 'bart', 'vocab_size': 54295, 'd_model': 768,
    'encoder_layers': 6, 'decoder_layers': 6,
    'encoder_attention_heads': 12, 'decoder_attention_heads': 12,
    'encoder_ffn_dim': 3072, 'decoder_ffn_dim': 3072,
    'activation_function': 'gelu', 'dropout': .1, 'attention_dropout': .1,
    'activation_dropout': .1, 'encoder_layerdrop': 0., 'decoder_layerdrop': 0.,
    'max_position_embeddings': 1024, 'scale_embedding': False, 'init_std': .02,
    'decoder_start_token_id': 102, 'pad_token_id': 0, 'bos_token_id': 101,
    'eos_token_id': 102, 'tie_word_embeddings': True, 'tie_encoder_decoder': False,
    'is_encoder_decoder': True, 'is_decoder': False, 'add_cross_attention': False,
    'use_cache': False,
}


def validate_formal_model_config(config):
    values = config if isinstance(config, dict) else config.to_dict()
    changed = {key: {'expected': expected, 'actual': values.get(key)}
               for key, expected in FORMAL_MODEL_CONTRACT.items()
               if key not in values or type(values[key]) is not type(expected) or values[key] != expected}
    for key in ('_attn_implementation', '_attn_implementation_internal'):
        if key in values and values[key] not in (None, 'eager'):
            changed[key] = {'expected': 'eager', 'actual': values[key]}
    if changed:
        raise ValueError('Frozen ED3 BART model_config mismatch: ' + repr(changed))
    return values


def validate_checkpoint_tensors(state):
    """Reject silently cast masters and conflicting aliases before strict loading."""
    shared = state.get('model.shared.weight')
    if not isinstance(shared, torch.Tensor) or tuple(shared.shape) != (54295, 768):
        raise ValueError('Checkpoint shared embedding must be 54295 x 768')
    bias = state.get('final_logits_bias')
    if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (1, 54295):
        raise ValueError('Checkpoint logits bias must be 1 x 54295')
    for name, value in state.items():
        # BART state contains only floating masters/biases. Integer, bool and
        # complex tensors would otherwise be silently cast by load_state_dict.
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError('Checkpoint requires FP32 master tensors: ' + name)
    for name in ('model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight', 'lm_head.weight'):
        value = state.get(name)
        if value is None or not torch.equal(shared, value):
            raise ValueError('Conflicting/missing shared checkpoint alias: ' + name)


def validate_formal_model(model):
    validate_formal_model_config(model.config)
    shared = model.get_input_embeddings().weight
    if tuple(shared.shape) != (54295, 768) or tuple(model.final_logits_bias.shape) != (1, 54295):
        raise ValueError('Model embedding/bias shape differs from the frozen extension')
    if not all(shared is value for value in (model.get_encoder().embed_tokens.weight,
               model.get_decoder().embed_tokens.weight, model.get_output_embeddings().weight)):
        raise ValueError('Model shared embedding/decoder/encoder/head ties were lost')
    if any(p.dtype != torch.float32 for p in model.parameters()) or model.final_logits_bias.dtype != torch.float32:
        raise ValueError('Formal BART masters and logits bias must remain FP32')


def autocast(device, cfg):
    precision = effective_precision(device, cfg)
    if torch.device(device).type == 'cuda':
        require_cuda_precision(device, cfg)
        return torch.autocast(device_type='cuda',
                              dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
                              enabled=precision != 'fp32')
    return nullcontext()


def scaler_for(device, cfg):
    # BF16 has FP32's exponent range; keep FP32 master parameters without loss scaling.
    return torch.amp.GradScaler('cuda', enabled=effective_precision(device, cfg) == 'fp16',
                               init_scale=1024.0)


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
    root_files = {path.name for path in folder.iterdir() if path.is_file()}
    if root_files != required | {'provenance.json'}:
        raise ValueError('Pretrained directory contains a missing, competing or extra root artifact')
    for name, expected in hashes.items():
        if sha(folder / name) != expected:
            raise ValueError('Pretrained file checksum mismatch: ' + name)
    if sha(folder / 'config.json') != cfg['native_config_sha256']:
        raise ValueError('Native architecture config differs from the frozen revision')
    if sha(folder / 'vocab.txt') != cfg['native_vocab_sha256']:
        raise ValueError('Native checkpoint vocabulary differs from the frozen revision')
    if sha(folder / weight_file) != cfg['native_weight_sha256']:
        raise ValueError('Native checkpoint weights differ from the frozen revision')
    return sha(folder / 'provenance.json')


def load_pretrained(folder, codec, cfg):
    from transformers import BartForConditionalGeneration
    verify_weights(folder, cfg)
    model, info = BartForConditionalGeneration.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False, weights_only=True,
        output_loading_info=True, attn_implementation='eager', dtype=torch.float32,
        use_safetensors=read(Path(folder) / 'provenance.json')['weight_file'].endswith('.safetensors'),
    )
    # Safetensors may omit aliases, but a real shared table must be loaded.
    missing = set(info['missing_keys']) - {'model.encoder.embed_tokens.weight',
        'model.decoder.embed_tokens.weight', 'lm_head.weight'}
    unexpected = set(info['unexpected_keys'])
    if missing or unexpected or info['mismatched_keys'] or info['error_msgs']:
        raise RuntimeError('Incomplete pretrained BART weights: ' + repr(info))
    validate_native_backbone(model.config)
    info['embedding_resize'] = resize_embeddings(model, codec.vocab_size)
    model.config.use_cache = False
    validate_formal_model(model)
    for p in model.parameters():
        p.requires_grad_(True)
    return model, info


def validate_native_backbone(native):
    actual = (native.model_type, native.vocab_size, native.encoder_layers, native.decoder_layers,
              native.d_model, native.encoder_attention_heads, native.decoder_attention_heads,
              native.encoder_ffn_dim, native.decoder_ffn_dim, native.activation_function,
              native.tie_word_embeddings, native.decoder_start_token_id, native.pad_token_id,
              native.bos_token_id, native.eos_token_id, native.max_position_embeddings,
              native.dropout, native.attention_dropout, native.activation_dropout,
              native.encoder_layerdrop, native.decoder_layerdrop, native.scale_embedding, native.init_std)
    expected = ('bart', 51271, 6, 6, 768, 12, 12, 3072, 3072, 'gelu', True,
                102, 0, 101, 102, 1024, .1, .1, .1, 0., 0., False, .02)
    if actual != expected:
        raise ValueError(f'Unexpected Chinese BART-base backbone: {actual!r}')
    return actual


def resize_embeddings(model, vocab_size):
    if vocab_size < model.get_input_embeddings().num_embeddings:
        raise ValueError('Shrinking the native vocabulary is forbidden')
    if not model.config.tie_word_embeddings:
        raise ValueError('BART output head must remain tied')
    old_input = model.get_input_embeddings().weight.detach().clone()
    old_output = model.get_output_embeddings().weight.detach().clone()
    old_bias = model.final_logits_bias.detach().clone()
    factor = float(model.config.init_std)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('BART init_std must be finite and positive')
    model.resize_token_embeddings(vocab_size, mean_resizing=False)
    model.tie_weights()
    # Initialize only new shared rows with native BART N(0, init_std).
    with torch.no_grad():
        if vocab_size > len(old_input):
            model.get_input_embeddings().weight[len(old_input):].normal_(mean=0.0, std=factor)
    if not torch.equal(old_input, model.get_input_embeddings().weight[:len(old_input)]):
        raise RuntimeError('Resize changed native shared input embeddings')
    if not torch.equal(old_output, model.get_output_embeddings().weight[:len(old_output)]):
        raise RuntimeError('Resize changed native output head')
    shared = model.get_input_embeddings().weight.data_ptr()
    if shared != model.get_encoder().embed_tokens.weight.data_ptr() or shared != model.get_decoder().embed_tokens.weight.data_ptr():
        raise RuntimeError('Encoder/decoder input embeddings must share parameters')
    if shared != model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError('BART output head must remain tied')
    if (not torch.equal(old_bias, model.final_logits_bias[:, :old_bias.shape[1]])
            or torch.count_nonzero(model.final_logits_bias[:, old_bias.shape[1]:])):
        raise RuntimeError('Resize must preserve native logits bias and zero new bias columns')
    return {'native_rows_preserved': len(old_input), 'new_rows_initialized': vocab_size - len(old_input),
            'input_output_tied': True, 'initialization': 'explicit_native_BART_normal_new_rows_only',
            'new_row_normal_mean': 0.0, 'new_row_normal_std': factor}


def project_hidden(model, hidden, candidate_ids, dtype_report=None, projection_cache=None):
    # Match native forward: BF16 linear, then FP32 final_logits_bias addition.
    # F.linear(..., bias=...) would fuse a different rounding path under AMP.
    if projection_cache is None:
        weight = model.get_output_embeddings().weight.index_select(0, candidate_ids)
        bias = model.final_logits_bias[0].index_select(0, candidate_ids)
    else:
        weight, bias = projection_cache.select(model, candidate_ids)
    logits = F.linear(hidden, weight)
    result = logits + bias
    if dtype_report is not None:
        dtype_report.update(linear=str(logits.dtype), bias=str(bias.dtype), result=str(result.dtype))
    return result


def decoder_hidden(model, decoder_input_ids, decoder_attention_mask, encoder_hidden, encoder_attention_mask):
    return model.get_decoder()(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
                         encoder_hidden_states=encoder_hidden, encoder_attention_mask=encoder_attention_mask,
                         use_cache=False, return_dict=True).last_hidden_state


def restricted_logits(model, input_ids, attention_mask, owners, positions, candidate_ids,
                      decoder_input_ids, decoder_attention_mask, dtype_report=None):
    encoded = model.get_encoder()(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
    ).last_hidden_state
    hidden = decoder_hidden(model, decoder_input_ids, decoder_attention_mask, encoded, attention_mask)
    selected = hidden[owners, positions]
    # Character prediction is restricted to the shared targets; the single
    # additional class is <END_GAP>, which is supervised after the known span.
    return project_hidden(model, selected, candidate_ids, dtype_report)


def per_sample_losses(model, batch, candidate_ids):
    logits = restricted_logits(model, batch['input_ids'], batch['attention_mask'],
                               batch['owners'], batch['positions'], candidate_ids,
                               batch['decoder_input_ids'], batch['decoder_attention_mask'])
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
                            lr=cfg['learning_rate'], betas=tuple(cfg['adam_betas']), eps=cfg['adam_epsilon'],
                            foreach=False, fused=False)
