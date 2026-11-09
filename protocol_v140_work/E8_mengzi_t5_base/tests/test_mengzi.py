"""E8-specific regressions; all numerical fixtures are tiny CPU T5 models."""
import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from ed2 import engine, formal_controller, run_state, cli
from ed2.model import validate_native_backbone, resize_embeddings
from ed2.io import write, digest
from test_audit_fixes import evaluation_report


def native_config():
    return T5Config(vocab_size=32128, num_layers=12, num_decoder_layers=12,
                    d_model=768, num_heads=12, d_kv=64, d_ff=2048,
                    feed_forward_proj='gated-gelu', tie_word_embeddings=False,
                    decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)


def test_native_t5_architecture_contract():
    assert validate_native_backbone(native_config())[:4] == ('t5', 32128, 12, 12)
    # Metadata allocation only: no real weights, CPU forward or GPU allocation.
    with torch.device('meta'):
        model = T5ForConditionalGeneration(native_config())
    assert len(model.encoder.block) == len(model.decoder.block) == 12
    assert model.shared.weight is model.encoder.embed_tokens.weight
    assert model.shared.weight is model.decoder.embed_tokens.weight
    assert model.shared.weight is not model.lm_head.weight
    assert sum(p.numel() for p in model.parameters()) == 247577856


@pytest.mark.parametrize('key,value', [('num_layers', 8), ('d_model', 512),
    ('num_heads', 6), ('d_ff', 1024), ('tie_word_embeddings', True), ('model_type', 'mt5')])
def test_e7_or_wrong_native_architecture_rejected(key, value):
    cfg = native_config()
    setattr(cfg, key, value)
    with pytest.raises(ValueError, match='backbone'):
        validate_native_backbone(cfg)


def test_tokenizer_ids_are_mengzi_not_randeng(codec, cfg):
    assert cfg['model_repo'] == 'Langboat/mengzi-t5-base'
    assert codec.mapping['original_vocab_size'] == 32128
    assert codec.vocab_size == 39275
    assert codec.mapping['obi_input_types'] == 4546 and len(codec.units) == 4116
    assert codec.vocab_size - codec.mapping['original_vocab_size'] == 7147


def test_resize_rejects_shrink_and_tied_head(tiny):
    with pytest.raises(ValueError, match='Shrinking'):
        resize_embeddings(tiny, 30)
    tiny.config.tie_word_embeddings = True
    with pytest.raises(ValueError, match='untied'):
        resize_embeddings(tiny, 50)


def test_e7_checkpoint_rejected_before_model_creation(tiny):
    config = tiny.config.to_dict()
    config['model_type'] = 'mt5'
    with pytest.raises(ValueError, match='mT5/E7'):
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
        assert 'E8_mengzi_t5_base' in ' '.join(command)
        assert 'ed2a_' in ' '.join(command)
        # sys.executable may belong to a read-only shared diagnostic environment;
        # experiment arguments/outputs must nevertheless remain exclusively E8.
        assert all('E7_randeng' not in value and 'ed1a_' not in value for value in command[1:])


def test_evaluation_logs_explicit_completion(tiny, cfg, monkeypatch, capsys):
    from types import SimpleNamespace
    codec = SimpleNamespace(target_ids=[5], units=['王'],
                            encode=lambda q: {'input_ids': [2], 'gap_length': 1})
    public = [{'example_id': 'cpu-fixture', 'source_id': 's', 'group_id': 'g', 'gap_length': 1}]
    monkeypatch.setattr(engine, 'predict_encoded_batch', lambda *args: [
        {'paths': [[0]], 'scores': [0.0], 'single_ranking': [0], 'score_kind': 'fixture'}])
    metrics, _ = engine.evaluate(tiny, codec, public, {'cpu-fixture': ['王']}, cfg, torch.device('cpu'))
    assert metrics['questions'] == 1
    assert 'EVAL 1/1' in capsys.readouterr().out
