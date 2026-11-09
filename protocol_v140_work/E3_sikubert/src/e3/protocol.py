"""Single source of truth for the frozen E3-A formal protocol."""


FORMAL_CONFIG_SHA256 = '1f782e1ae4806ac40f08af97576864480ebd2415a13dc2975a5af97442982651'
DATA_MANIFEST_SHA256 = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'


FORMAL_CONFIG = {'experiment': 'E3-A', 'dataset_version': 'v1.5.0', 'training_square_bracket_restorations_maskable': True, 'training_boundary_masks_enabled': True, 'protected_target_types': ['damage', 'ligature', 'annotation', 'non_main'], 'model_repo': 'SIKU-BERT/sikubert', 'model_revision': 'fc656de2d6bde33919102dd3abe31c843f42226a', 'native_config_sha256': '8c75c486410cda6a3d37dafd435f20cca55fbbc10834314dadc7750f4aaa55c2', 'native_vocab_sha256': '51d3161cc29db6b5d2e841da562b9f8b59c050e44bcab2319c3eafa71c62522e', 'tokenizer_asset': 'E1', 'dapt': False, 'max_epochs': 50, 'samples_per_epoch': 66511, 'learning_rate': 5e-05, 'effective_batch_size': 64, 'micro_batch_size': 32, 'weight_decay': 0.01, 'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-08, 'gradient_clip': 1.0, 'warmup_epochs': 1, 'schedule': 'linear', 'precision': 'fp16', 'max_input_length': 128, 'pad_multiple': 8, 'eval_question_batch_size': 128, 'eval_forward_batch_size': 256, 'beam_width': 10, 'return_sequences': 5, 'single_mask_return_sequences': 10, 'fill_order': 'left_to_right', 'loss_reduction': 'mean_of_per_sample_mean_mask_cross_entropy', 'prediction_space': 'shared_4116_targets', 'checkpoint_selection': ['macro5_em1', 'macro5_hit5', 'earlier_epoch'], 'validation_suite': 'dev_full', 'test_suites': ['test_random', 'test_expert', 'test_prefix', 'test_suffix', 'test_all_position'], 'early_stopping': True, 'early_stopping_patience': 5, 'seeds': [42, 3407, 2026]}


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
    """Reject every deviation from the authorized E3-A training/evaluation protocol."""
    if not isinstance(cfg, dict):
        raise ValueError('E3-A configuration must be a JSON object')
    missing = sorted(set(FORMAL_CONFIG) - set(cfg))
    extra = sorted(set(cfg) - set(FORMAL_CONFIG))
    changed = sorted(k for k in set(cfg) & set(FORMAL_CONFIG)
                     if not _same_value(cfg[k], FORMAL_CONFIG[k]))
    if missing or extra or changed:
        details = {'missing': missing, 'extra': extra,
                   'changed': {k: {'expected': FORMAL_CONFIG[k], 'actual': cfg[k]} for k in changed}}
        raise ValueError('Unsupported E3-A formal protocol change: ' + repr(details))
    return cfg
