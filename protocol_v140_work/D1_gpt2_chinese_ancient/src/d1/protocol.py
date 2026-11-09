"""Single source of truth for the frozen D1-A decoder-only protocol."""


FORMAL_CONFIG_SHA256 = '7fb07108e1f2ed455aa6b3a40ac314c59ba8d485116e1a28a095d312590047ed'
DATA_MANIFEST_SHA256 = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'


FORMAL_CONFIG = {'experiment': 'D1-A', 'dataset_version': 'v1.5.0', 'training_square_bracket_restorations_maskable': True, 'training_boundary_masks_enabled': True, 'protected_target_types': ['damage', 'ligature', 'annotation', 'non_main'], 'model_repo': 'uer/gpt2-chinese-ancient', 'model_revision': '3b264872995b09b5d9873e458f3d03a221c00669', 'native_config_sha256': '84eec2caac9f7f2846a72ff42f4b1ab8092fe2f93c9b42595f26b5875bc076c3', 'native_vocab_sha256': '5e6aec77a6c440fa5a805e68d9ecf10e61280877eb112e19b58e14b6a3417853', 'tokenizer_asset': 'D1', 'architecture': 'decoder_fim', 'dapt': False, 'max_epochs': 50, 'samples_per_epoch': 66511, 'learning_rate': 5e-05, 'effective_batch_size': 64, 'micro_batch_size': 32, 'weight_decay': 0.01, 'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-08, 'gradient_clip': 1.0, 'warmup_epochs': 1, 'schedule': 'linear', 'precision': 'fp16', 'max_input_length': 128, 'pad_multiple': 8, 'eval_question_batch_size': 128, 'eval_forward_batch_size': 256, 'beam_width': 10, 'return_sequences': 5, 'single_mask_return_sequences': 10, 'generation_constraint': 'known_length_then_end_gap', 'fill_order': 'fim_autoregressive_left_to_right', 'loss_reduction': 'mean_of_per_sample_mean_answer_and_end_cross_entropy', 'prediction_space': 'shared_4116_targets_plus_end_gap_control', 'checkpoint_selection': ['macro5_em1', 'macro5_hit5', 'earlier_epoch'], 'validation_suite': 'dev_full', 'test_suites': ['test_random', 'test_expert', 'test_prefix', 'test_suffix', 'test_all_position'], 'early_stopping': True, 'early_stopping_patience': 5, 'seeds': [42, 3407, 2026]}


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
    """Reject every deviation from the authorized D1-A training/evaluation protocol."""
    if not isinstance(cfg, dict):
        raise ValueError('D1-A configuration must be a JSON object')
    missing = sorted(set(FORMAL_CONFIG) - set(cfg))
    extra = sorted(set(cfg) - set(FORMAL_CONFIG))
    changed = sorted(k for k in set(cfg) & set(FORMAL_CONFIG)
                     if not _same_value(cfg[k], FORMAL_CONFIG[k]))
    if missing or extra or changed:
        details = {'missing': missing, 'extra': extra,
                   'changed': {k: {'expected': FORMAL_CONFIG[k], 'actual': cfg[k]} for k in changed}}
        raise ValueError('Unsupported D1-A formal protocol change: ' + repr(details))
    return cfg
