"""Single source of truth for the frozen ED1-A encoder-decoder protocol."""


FORMAL_CONFIG_SHA256 = '9ee0573fe9bd85bd61ae17e5dc4f683086776da2d8b91368faa1847838c4506e'
DATA_MANIFEST_SHA256 = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'


FORMAL_CONFIG = {'experiment': 'ED1-A', 'dataset_version': 'v1.5.0', 'training_square_bracket_restorations_maskable': True, 'training_boundary_masks_enabled': True, 'protected_target_types': ['damage', 'ligature', 'annotation', 'non_main'], 'model_repo': 'IDEA-CCNL/Randeng-T5-77M', 'model_revision': '31ab711e7a339442d7c203437272f5bbfd20d7f3', 'native_config_sha256': 'f96348ccc5ce5a6adf107257e161076e21e01bb8f0100d76a7c1d0a163c1791a', 'native_weight_sha256': '92ce8af84829ce18372bd54b24c499fc0a3691aadda2bea59117c48c06c9d607', 'native_vocab_sha256': 'c65feffa65ff0378759778193852083d23349cb1b40c906e9463a12f8076ff32', 'tokenizer_asset': 'ED1', 'architecture': 'encoder_decoder_gap', 'dapt': False, 'max_epochs': 50, 'samples_per_epoch': 66511, 'learning_rate': 5e-05, 'effective_batch_size': 64, 'micro_batch_size': 32, 'weight_decay': 0.01, 'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-08, 'gradient_clip': 1.0, 'warmup_epochs': 1, 'schedule': 'linear', 'precision': 'bf16', 'max_input_length': 128, 'pad_multiple': 8, 'eval_question_batch_size': 128, 'eval_forward_batch_size': 256, 'beam_width': 10, 'return_sequences': 5, 'single_mask_return_sequences': 10, 'generation_constraint': 'known_length_then_end_gap', 'fill_order': 'seq2seq_autoregressive_left_to_right', 'loss_reduction': 'mean_of_per_sample_mean_answer_and_end_cross_entropy', 'prediction_space': 'shared_4116_targets_plus_end_gap_control', 'checkpoint_selection': ['macro5_em1', 'macro5_hit5', 'earlier_epoch'], 'validation_suite': 'dev_full', 'test_suites': ['test_random', 'test_expert', 'test_prefix', 'test_suffix', 'test_all_position'], 'early_stopping': True, 'early_stopping_patience': 5, 'seeds': [42, 3407, 2026]}


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
    """Reject every deviation from the authorized ED1-A training/evaluation protocol."""
    if not isinstance(cfg, dict):
        raise ValueError('ED1-A configuration must be a JSON object')
    missing = sorted(set(FORMAL_CONFIG) - set(cfg))
    extra = sorted(set(cfg) - set(FORMAL_CONFIG))
    changed = sorted(k for k in set(cfg) & set(FORMAL_CONFIG)
                     if not _same_value(cfg[k], FORMAL_CONFIG[k]))
    if missing or extra or changed:
        details = {'missing': missing, 'extra': extra,
                   'changed': {k: {'expected': FORMAL_CONFIG[k], 'actual': cfg[k]} for k in changed}}
        raise ValueError('Unsupported ED1-A formal protocol change: ' + repr(details))
    return cfg
