"""Test-only loader. Never imported by production CLI, resume or inspection."""
def load_tiny_checkpoint(ckpt, device):
    from transformers import BartConfig, BartForConditionalGeneration
    config = BartConfig.from_dict(ckpt['model_config'])
    config._attn_implementation = 'eager'
    config.use_cache = False
    model = BartForConditionalGeneration(config)
    model.load_state_dict(ckpt['model'], strict=True)
    model.tie_weights()
    return model.to(device)
