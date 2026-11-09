from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .io import read, sha
from .runtime import effective_precision, require_cuda_precision


SUPPORTED_WEIGHT_FILES = frozenset({'model.safetensors', 'pytorch_model.bin'})


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
    required = {'config.json', 'spiece.model', weight_file}
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
    if sha(folder / 'spiece.model') != cfg['native_vocab_sha256']:
        raise ValueError('Native checkpoint vocabulary differs from the frozen revision')
    if sha(folder / weight_file) != cfg['native_weight_sha256']:
        raise ValueError('Native checkpoint weights differ from the frozen revision')
    return sha(folder / 'provenance.json')


def load_pretrained(folder, codec, cfg):
    from transformers import T5ForConditionalGeneration
    verify_weights(folder, cfg)
    model, info = T5ForConditionalGeneration.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False, weights_only=True,
        output_loading_info=True, attn_implementation='eager', dtype=torch.float32,
        use_safetensors=read(Path(folder) / 'provenance.json')['weight_file'].endswith('.safetensors'),
    )
    # Encoder/decoder input embeddings alias shared.weight. The LM head is
    # UNTIED for this checkpoint and must actually be present in the weights.
    missing = set(info['missing_keys']) - {'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'}
    unexpected = set(info['unexpected_keys'])
    if missing or unexpected or info['mismatched_keys'] or info['error_msgs']:
        raise RuntimeError('Incomplete pretrained T5 weights: ' + repr(info))
    validate_native_backbone(model.config)
    info['embedding_resize'] = resize_embeddings(model, codec.vocab_size)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    return model, info


def validate_native_backbone(native):
    actual = (native.model_type, native.vocab_size, native.num_layers, native.num_decoder_layers,
              native.d_model, native.num_heads, native.d_kv, native.d_ff, native.feed_forward_proj,
              native.tie_word_embeddings, native.decoder_start_token_id, native.pad_token_id, native.eos_token_id)
    expected = ('t5', 32128, 12, 12, 768, 12, 64, 2048, 'gated-gelu', False, 0, 0, 1)
    if actual != expected:
        raise ValueError(f'Unexpected Mengzi-T5-base backbone: {actual!r}')
    return actual


def resize_embeddings(model, vocab_size):
    if vocab_size < model.get_input_embeddings().num_embeddings:
        raise ValueError('Shrinking the native vocabulary is forbidden')
    if model.config.tie_word_embeddings:
        raise ValueError('Mengzi output head must remain untied')
    old_input = model.get_input_embeddings().weight.detach().clone()
    old_output = model.get_output_embeddings().weight.detach().clone()
    factor = float(model.config.initializer_factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('T5 initializer_factor must be finite and positive')
    model.resize_token_embeddings(vocab_size, mean_resizing=False)
    model.tie_weights()  # Preserves native tie_word_embeddings=False.
    # Transformers 4.57.x calls T5._init_weights on a bare nn.Linear when
    # resizing an untied head. T5 has no bare-Linear initializer: otherwise
    # new output rows retain PyTorch's tiny Kaiming-uniform initialization.
    # Initialize ONLY added rows, separately for the two native T5 tables.
    with torch.no_grad():
        if vocab_size > len(old_input):
            model.get_input_embeddings().weight[len(old_input):].normal_(mean=0.0, std=factor)
            model.get_output_embeddings().weight[len(old_output):].normal_(mean=0.0, std=factor)
    if not torch.equal(old_input, model.get_input_embeddings().weight[:len(old_input)]):
        raise RuntimeError('Resize changed native shared input embeddings')
    if not torch.equal(old_output, model.get_output_embeddings().weight[:len(old_output)]):
        raise RuntimeError('Resize changed native output head')
    shared = model.get_input_embeddings().weight.data_ptr()
    if shared != model.encoder.embed_tokens.weight.data_ptr() or shared != model.decoder.embed_tokens.weight.data_ptr():
        raise RuntimeError('Encoder/decoder input embeddings must share parameters')
    if shared == model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError('Mengzi output head must remain untied')
    return {'native_rows_preserved': len(old_input), 'new_rows_initialized': vocab_size - len(old_input),
            'input_output_tied': False, 'initialization': 'explicit_native_T5_normal_new_rows_only',
            'new_row_normal_mean': 0.0, 'new_row_normal_std': factor}


def project_hidden(model, hidden, candidate_ids):
    if model.config.tie_word_embeddings:
        hidden = hidden * (model.config.d_model ** -0.5)
    return F.linear(hidden, model.get_output_embeddings().weight.index_select(0, candidate_ids))


def decoder_hidden(model, decoder_input_ids, decoder_attention_mask, encoder_hidden, encoder_attention_mask):
    return model.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
                         encoder_hidden_states=encoder_hidden, encoder_attention_mask=encoder_attention_mask,
                         use_cache=False, return_dict=True).last_hidden_state


def restricted_logits(model, input_ids, attention_mask, owners, positions, candidate_ids,
                      decoder_input_ids, decoder_attention_mask):
    encoded = model.encoder(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
    ).last_hidden_state
    hidden = decoder_hidden(model, decoder_input_ids, decoder_attention_mask, encoded, attention_mask)
    selected = hidden[owners, positions]
    # Character prediction is restricted to the shared targets; the single
    # additional class is <END_GAP>, which is supervised after the known span.
    return project_hidden(model, selected, candidate_ids)


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
