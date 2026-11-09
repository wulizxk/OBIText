"""Single source of truth for the frozen ED2-A encoder-decoder protocol."""


FORMAL_CONFIG_SHA256 = 'b86c52f234b31b21d1c83384d0977997d489bad8eb99c38a0b2bc6b98a2828fd'
DATA_MANIFEST_SHA256 = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'


FORMAL_CONFIG = {'experiment': 'ED2-A', 'dataset_version': 'v1.5.0', 'training_square_bracket_restorations_maskable': True, 'training_boundary_masks_enabled': True, 'protected_target_types': ['damage', 'ligature', 'annotation', 'non_main'], 'model_repo': 'Langboat/mengzi-t5-base', 'model_revision': '6616a21ba6f42867e2c14ef4c84ef44e22af3248', 'native_config_sha256': 'dca513b27692b67beec47cff0296e34207d44a1272c8279d0854af30416874d3', 'native_weight_sha256': 'de2ffe9a9ddcf05b154b56c77fc44765a195375192cf7f64a95c440a91830576', 'native_vocab_sha256': 'e0dc796e4a3d83ccbeb33b35ce905c5821b427f19343e1bdad7d0b47a3317cac', 'tokenizer_asset': 'ED2', 'architecture': 'encoder_decoder_gap', 'dapt': False, 'max_epochs': 50, 'samples_per_epoch': 66511, 'learning_rate': 5e-05, 'effective_batch_size': 64, 'micro_batch_size': 64, 'weight_decay': 0.01, 'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-08, 'gradient_clip': 1.0, 'warmup_epochs': 1, 'schedule': 'linear', 'precision': 'bf16', 'max_input_length': 128, 'pad_multiple': 8, 'eval_question_batch_size': 32, 'eval_forward_batch_size': 64, 'evaluation_backend': 'cuda_graph_exact_v1', 'beam_width': 10, 'return_sequences': 5, 'single_mask_return_sequences': 10, 'generation_constraint': 'known_length_then_end_gap', 'fill_order': 'seq2seq_autoregressive_left_to_right', 'loss_reduction': 'mean_of_per_sample_mean_answer_and_end_cross_entropy', 'prediction_space': 'shared_4116_targets_plus_end_gap_control', 'checkpoint_selection': ['macro5_em1', 'macro5_hit5', 'earlier_epoch'], 'validation_suite': 'dev_full', 'test_suites': ['test_random', 'test_expert', 'test_prefix', 'test_suffix', 'test_all_position'], 'early_stopping': True, 'early_stopping_patience': 5, 'seeds': [42, 3407, 2026]}


def _same_value(actual, expected):
    """Compare JSON protocol values without bool/int or int/float coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_value(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_same_value(actual[k], v) for k, v in expected.items())
    return actual == expected


def validate_formal_config(cfg):
    """Reject every deviation from the authorized ED2-A training/evaluation protocol."""
    if not isinstance(cfg, dict):
        raise ValueError('ED2-A configuration must be a JSON object')
    missing = sorted(set(FORMAL_CONFIG) - set(cfg))
    extra = sorted(set(cfg) - set(FORMAL_CONFIG))
    changed = sorted(k for k in set(cfg) & set(FORMAL_CONFIG)
                     if not _same_value(cfg[k], FORMAL_CONFIG[k]))
    if missing or extra or changed:
        details = {'missing': missing, 'extra': extra,
                   'changed': {k: {'expected': FORMAL_CONFIG[k], 'actual': cfg[k]} for k in changed}}
        raise ValueError('Unsupported ED2-A formal protocol change: ' + repr(details))
    return cfg
