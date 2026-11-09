"""Single source of truth for the frozen E2-A formal protocol."""


FORMAL_CONFIG_SHA256 = '128896cf0505f9756615d7c357c6c2b5fd324dfa5d588a2ea517a6223b06d6f4'
DATA_MANIFEST_SHA256 = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'


FORMAL_CONFIG = {'experiment': 'E2-A', 'dataset_version': 'v1.5.0', 'training_square_bracket_restorations_maskable': True, 'training_boundary_masks_enabled': True, 'protected_target_types': ['damage', 'ligature', 'annotation', 'non_main'], 'model_repo': 'ethanyt/guwenbert-base', 'model_revision': 'eff0d4a5196d7bf7b8be746c5c6437e89d8b9061', 'native_config_sha256': 'c77dd1012bc22fea2d6d422359499c2ef095ae412300ef5d726e96b0a622904c', 'dapt': False, 'max_epochs': 50, 'samples_per_epoch': 66511, 'learning_rate': 5e-05, 'effective_batch_size': 64, 'micro_batch_size': 32, 'weight_decay': 0.01, 'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-08, 'gradient_clip': 1.0, 'warmup_epochs': 1, 'schedule': 'linear', 'precision': 'fp16', 'max_input_length': 128, 'pad_multiple': 8, 'eval_question_batch_size': 128, 'eval_forward_batch_size': 256, 'beam_width': 10, 'return_sequences': 5, 'single_mask_return_sequences': 10, 'fill_order': 'left_to_right', 'loss_reduction': 'mean_of_per_sample_mean_mask_cross_entropy', 'prediction_space': 'shared_4116_targets', 'checkpoint_selection': ['macro5_em1', 'macro5_hit5', 'earlier_epoch'], 'validation_suite': 'dev_full', 'test_suites': ['test_random', 'test_expert', 'test_prefix', 'test_suffix', 'test_all_position'], 'early_stopping': True, 'early_stopping_patience': 5, 'seeds': [42, 3407, 2026]}


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
    """Reject every deviation from the authorized E2-A training/evaluation protocol."""
    if not isinstance(cfg, dict):
        raise ValueError('E2-A configuration must be a JSON object')
    missing = sorted(set(FORMAL_CONFIG) - set(cfg))
    extra = sorted(set(cfg) - set(FORMAL_CONFIG))
    changed = sorted(k for k in set(cfg) & set(FORMAL_CONFIG)
                     if not _same_value(cfg[k], FORMAL_CONFIG[k]))
    if missing or extra or changed:
        details = {'missing': missing, 'extra': extra,
                   'changed': {k: {'expected': FORMAL_CONFIG[k], 'actual': cfg[k]} for k in changed}}
        raise ValueError('Unsupported E2-A formal protocol change: ' + repr(details))
    return cfg
