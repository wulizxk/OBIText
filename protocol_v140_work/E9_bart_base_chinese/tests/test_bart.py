"""E9-specific regressions; numerical fixtures are tiny CPU BART models."""
import pytest
import torch
from transformers import BartConfig, BartForConditionalGeneration

from ed3 import engine, formal_controller, run_state, cli
from ed3.model import validate_native_backbone, resize_embeddings
from ed3.io import write, digest
from test_audit_fixes import evaluation_report


def native_config():
    return BartConfig(vocab_size=51271, encoder_layers=6, decoder_layers=6,
                    d_model=768, encoder_attention_heads=12, decoder_attention_heads=12,
                    encoder_ffn_dim=3072, decoder_ffn_dim=3072, activation_function='gelu',
                    tie_word_embeddings=True, attention_dropout=.1, activation_dropout=.1,
                    decoder_start_token_id=102, pad_token_id=0, bos_token_id=101, eos_token_id=102)


def test_native_bart_architecture_contract():
    assert validate_native_backbone(native_config())[:4] == ('bart', 51271, 6, 6)
    # Metadata allocation only: no real weights, CPU forward or GPU allocation.
    with torch.device('meta'):
        model = BartForConditionalGeneration(native_config())
    assert len(model.model.encoder.layers) == len(model.model.decoder.layers) == 6
    assert model.model.shared.weight is model.get_encoder().embed_tokens.weight
    assert model.model.shared.weight is model.get_decoder().embed_tokens.weight
    assert model.model.shared.weight is model.lm_head.weight
    assert 140_000_000 < sum(p.numel() for p in model.parameters()) < 141_000_000


@pytest.mark.parametrize('key,value', [('encoder_layers', 8), ('d_model', 512),
    ('encoder_attention_heads', 6), ('decoder_ffn_dim', 1024), ('tie_word_embeddings', False),
    ('decoder_start_token_id', 0), ('attention_dropout', 0.), ('model_type', 't5')])
def test_e7_or_wrong_native_architecture_rejected(key, value):
    cfg = native_config()
    setattr(cfg, key, value)
    with pytest.raises(ValueError, match='backbone'):
        validate_native_backbone(cfg)


def test_tokenizer_ids_are_bart_not_t5(codec, cfg):
    assert cfg['model_repo'] == 'OpenMOSS-Team/bart-base-chinese'
    assert codec.mapping['original_vocab_size'] == 51271
    assert codec.vocab_size == 54295
    assert codec.mapping['obi_input_types'] == 4546 and len(codec.units) == 4116
    assert codec.vocab_size - codec.mapping['original_vocab_size'] == 3024
    assert codec.decoder_start_id == 102


def test_resize_rejects_shrink_and_untied_head(tiny):
    with pytest.raises(ValueError, match='Shrinking'):
        resize_embeddings(tiny, 30)
    tiny.config.tie_word_embeddings = False
    with pytest.raises(ValueError, match='remain tied'):
        resize_embeddings(tiny, 138)


def test_e7_checkpoint_rejected_before_model_creation(tiny):
    config = tiny.config.to_dict()
    config['model_type'] = 'mt5'
    with pytest.raises(ValueError, match='T5/E7/E8'):
        engine.model_from_checkpoint({'model_config': config}, torch.device('cpu'))


def illegal_history(cfg):
    history = []
    for epoch in range(1, 51):
        metrics = {'macro5_em1': .3 if epoch >= 47 else .1, 'macro5_hit5': .5}
        history.append({'epoch': epoch, 'step': run_state.steps_for(cfg, epoch),
                        'validation': metrics, 'new_best': epoch in (1, 47)})
    return history, {'epoch': 47, 'metrics': history[46]['validation']}


@pytest.mark.parametrize('entry', ['resume', 'completion', 'trace'])
def test_prior_patience_hit_cannot_be_hidden_by_later_improvement(cfg, entry):
    history, best = illegal_history(cfg)
    with pytest.raises(ValueError, match='required early stop at epoch 6'):
        if entry == 'resume':
            run_state.validate_history(cfg, 50, run_state.steps_for(cfg, 50), history, best)
        elif entry == 'completion':
            run_state.validate_stopping(cfg, 50, history, best, 'max_epochs')
        else:
            run_state.validate_completion_trace(cfg, {'epochs_ran': 50, 'best': best,
                'stopping': {'reason': 'max_epochs'}, 'selection_history': history})


def test_summary_replays_early_stop_history(cfg, tmp_path):
    directories = []
    for seed in cfg['seeds']:
        report = evaluation_report(cfg, seed)
        marker = report['training_completion']
        # Still 22 epochs, best 17: hide a first patience hit at epoch 6.
        for row in marker['selection_history'][:16]:
            row['validation'] = {'macro5_em1': .01, 'macro5_hit5': .02}
            row['new_best'] = row['epoch'] == 1
        report['completion_sha256'] = digest(marker)
        directory = tmp_path / str(seed)
        write(directory / 'metrics.json', report)
        directories.append(directory)
    with pytest.raises(ValueError, match='required early stop at epoch 6'):
        cli.summarize(directories, tmp_path / 'summary.json')
    assert not (tmp_path / 'summary.json').exists()


def test_launcher_paths_never_target_another_experiment():
    for stage in ('train', 'test', 'summarize'):
        command = formal_controller.command(stage, None if stage == 'summarize' else 42)
        assert 'E9_bart_base_chinese' in ' '.join(command)
        assert 'ed3a_' in ' '.join(command)
        # sys.executable may belong to a read-only shared diagnostic environment;
        # experiment arguments/outputs must nevertheless remain exclusively E9.
        assert all('E7_randeng' not in value and 'ed1a_' not in value for value in command[1:])


def test_evaluation_logs_explicit_completion(tiny, cfg, monkeypatch, capsys):
    from types import SimpleNamespace
    codec = SimpleNamespace(target_ids=[5], candidate_ids=[5, 6], units=['王'],
                            encode=lambda q: {'input_ids': [2], 'gap_length': 1})
    public = [{'example_id': 'cpu-fixture', 'source_id': 's', 'group_id': 'g', 'gap_length': 1}]
    monkeypatch.setattr(engine, 'predict_encoded_batch', lambda *args, **kwargs: [
        {'paths': [[0]], 'scores': [0.0], 'single_ranking': [0], 'score_kind': 'fixture'}])
    metrics, _ = engine.evaluate(tiny, codec, public, {'cpu-fixture': ['王']}, cfg, torch.device('cpu'))
    assert metrics['questions'] == 1
    assert 'EVAL 1/1' in capsys.readouterr().out
