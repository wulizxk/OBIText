"""Single source of truth for the frozen D2-A decoder-only protocol."""


FORMAL_CONFIG_SHA256 = '553655fd3c0bcef2142901c592ef63a04ea60fc6642f129b31a38d3924064828'
DATA_MANIFEST_SHA256 = 'ec2e8fe2a9f46a7bb642f0736146f8aeefcdb62a509b27eef7cade7102087500'


FORMAL_CONFIG = {'experiment': 'D2-A', 'dataset_version': 'v1.5.0', 'training_square_bracket_restorations_maskable': True, 'training_boundary_masks_enabled': True, 'protected_target_types': ['damage', 'ligature', 'annotation', 'non_main'], 'model_repo': 'Qwen/Qwen2.5-0.5B', 'model_revision': '060db6499f32faf8b98477b0a26969ef7d8b9987', 'native_config_sha256': '479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b', 'native_weight_sha256': '88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342', 'native_vocab_sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 'frozen_merges_sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5', 'tokenizer_asset': 'D2', 'architecture': 'decoder_fim', 'dapt': False, 'max_epochs': 50, 'samples_per_epoch': 66511, 'learning_rate': 2e-05, 'learning_rate_basis': 'preregistered_model_specific_value_not_dev_optimized', 'effective_batch_size': 64, 'micro_batch_size': 32, 'micro_batch_token_budget': 2048, 'weight_decay': 0.01, 'adamw_foreach': False, 'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-08, 'gradient_clip': 1.0, 'warmup_epochs': 1, 'schedule': 'linear', 'precision': 'fp16', 'max_input_length': 128, 'pad_multiple': 8, 'eval_question_batch_size': 32, 'eval_forward_batch_size': 64, 'gradient_checkpointing': True, 'tokenizer_new_id_start': 151665, 'tokenizer_added_embedding_initialization': 'normal_mean_0_std_initializer_range_all_ids_ge_start', 'beam_width': 10, 'return_sequences': 5, 'single_mask_return_sequences': 10, 'generation_constraint': 'known_length_then_end_gap', 'fill_order': 'fim_autoregressive_left_to_right', 'loss_reduction': 'mean_of_per_sample_mean_answer_and_end_cross_entropy', 'prediction_space': 'shared_4116_targets_plus_end_gap_control', 'checkpoint_selection': ['macro5_em1', 'macro5_hit5', 'earlier_epoch'], 'validation_suite': 'dev_full', 'test_suites': ['test_random', 'test_expert', 'test_prefix', 'test_suffix', 'test_all_position'], 'early_stopping': True, 'early_stopping_patience': 5, 'seeds': [42, 3407, 2026]}


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
    """Reject every deviation from the authorized D2-A training/evaluation protocol."""
    if not isinstance(cfg, dict):
        raise ValueError('D2-A configuration must be a JSON object')
    missing = sorted(set(FORMAL_CONFIG) - set(cfg))
    extra = sorted(set(cfg) - set(FORMAL_CONFIG))
    changed = sorted(k for k in set(cfg) & set(FORMAL_CONFIG)
                     if not _same_value(cfg[k], FORMAL_CONFIG[k]))
    if missing or extra or changed:
        details = {'missing': missing, 'extra': extra,
                   'changed': {k: {'expected': FORMAL_CONFIG[k], 'actual': cfg[k]} for k in changed}}
        raise ValueError('Unsupported D2-A formal protocol change: ' + repr(details))
    return cfg
