import copy
from pathlib import Path
import sys

import pytest
from benchmark_test_fixtures_v140 import formal_metric_fixture
import torch

from e2 import engine, cli, run_state, runtime
from e2.io import read, write, sha, digest
from test_training_loop import MiniCodec, MiniSampler


def identity_for(cfg, seed=42):
    return {'cfg': cfg, 'seed': seed, 'runtime': runtime.runtime_versions(),
            'execution_policy': runtime.configure_execution(), 'data_sha256': 'synthetic-data',
            'code_sha256': 'synthetic-code', 'weights_sha256': 'synthetic-weights'}


def test_runtime_mismatch_rejected_before_model_load(cfg, monkeypatch):
    identity = identity_for(cfg)
    identity['runtime']['torch'] = '0.0.AUDIT_MISMATCH'
    monkeypatch.setattr(engine, 'load_checkpoint', lambda _: {'identity': identity})
    monkeypatch.setattr(engine, 'model_from_checkpoint', lambda *_: pytest.fail('Model loaded before runtime rejection'))
    with pytest.raises(ValueError, match='runtime mismatch'):
        engine.inspect_checkpoint(Path('.'), Path('.'), Path('fake.pt'), torch.device('cpu'))


def test_inference_applies_training_policy_and_records_environment(cfg, monkeypatch):
    identity = identity_for(cfg)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    runtime.require_runtime(identity)
    assert runtime.execution_policy() == identity['execution_policy']
    env = runtime.evaluation_environment('cpu', cfg)
    assert env['effective_precision'] == 'fp32'
    assert env['versions'] == identity['runtime']
    bad = copy.deepcopy(identity)
    bad['execution_policy']['cublas_workspace_config'] = 'different'
    with pytest.raises(ValueError, match='execution-policy mismatch'):
        runtime.require_runtime(bad)


def evaluation_report(cfg, seed=42):
    ident = identity_for(cfg, seed)
    return {'seed': seed, 'epoch': 17, 'identity': ident,
            'evaluation_runtime': runtime.evaluation_environment('cpu', cfg),
            'suites': formal_metric_fixture()}


@pytest.mark.parametrize('change', ['missing_runtime', 'version', 'hardware', 'precision', 'duplicate_seed', 'config'])
def test_summary_rejects_incompatible_evaluations(cfg, tmp_path, change):
    a, b = evaluation_report(cfg, 42), evaluation_report(cfg, 3407)
    if change == 'missing_runtime':
        b.pop('evaluation_runtime')
    elif change == 'version':
        b['evaluation_runtime']['versions']['torch'] = 'different'
    elif change == 'hardware':
        b['evaluation_runtime']['hardware']['name'] = 'different-hardware'
    elif change == 'precision':
        b['evaluation_runtime']['effective_precision'] = 'fp16'
    elif change == 'duplicate_seed':
        b['seed'] = b['identity']['seed'] = 42
    elif change == 'config':
        b['identity']['cfg'] = dict(cfg, learning_rate=1e-4)
    write(tmp_path / 'a/metrics.json', a); write(tmp_path / 'b/metrics.json', b)
    with pytest.raises(ValueError):
        cli.summarize([tmp_path / 'a', tmp_path / 'b'], tmp_path / 'summary.json')
    assert not (tmp_path / 'summary.json').exists()


def test_summary_accepts_three_matched_seeds(cfg, tmp_path):
    directories = []
    for seed in cfg['seeds']:
        d = tmp_path / str(seed)
        write(d / 'metrics.json', evaluation_report(cfg, seed))
        directories.append(d)
    cli.summarize(directories, tmp_path / 'summary.json')
    result = read(tmp_path / 'summary.json')
    assert result['formal_three_seed_complete'] is True
    assert result['evaluation_runtime']['versions'] == runtime.runtime_versions()


def make_completed_fixture(tmp_path, cfg, tiny):
    """Fabricate metadata for 50 epochs; NO 50-epoch model training occurs."""
    output = tmp_path / 'run'
    output.mkdir()
    identity = identity_for(cfg)
    metrics = {'macro5_em1': .3, 'macro5_hit5': .5}
    checkpoint = {'kind': 'best', 'schema_version': run_state.SCHEMA_VERSION,
                  'identity': identity, 'epoch': 47,
                  'metrics': metrics, 'model_config': tiny.config.to_dict(), 'model': tiny.state_dict()}
    best = run_state.stage_best(output, checkpoint, [], engine.save_checkpoint)
    history = []
    for epoch in range(1, 51):
        score = metrics if epoch == 47 else {'macro5_em1': .1, 'macro5_hit5': .2}
        history.append({'epoch': epoch, 'step': run_state.steps_for(cfg, epoch),
                        'validation': score, 'new_best': epoch in (1, 47)})
    state = {'kind': 'training', 'schema_version': run_state.SCHEMA_VERSION,
             'identity': identity, 'epoch': 50,
             'step': run_state.steps_for(cfg, 50), 'history': history, 'best': best}
    engine.save_checkpoint(output / 'last.pt', state)
    write(output / 'run.json', {'kind': 'formal', 'identity': identity})
    run_state.complete_run(output, identity, 50, state['step'], history, best, 'max_epochs')
    return output, identity


def test_completed_run_accepts_best_before_epoch50(tmp_path, cfg, tiny):
    out, _ = make_completed_fixture(tmp_path, cfg, tiny)
    checkpoint, marker = run_state.require_completed_run(out / 'best.pt')
    assert checkpoint['epoch'] == 47
    assert marker['max_epochs'] == 50 and marker['epochs_ran'] == 50 and marker['steps'] == cfg['max_epochs'] * ((cfg['samples_per_epoch'] + cfg['effective_batch_size'] - 1) // cfg['effective_batch_size'])
    assert marker['stopping']['reason'] == 'max_epochs'


def test_completed_run_accepts_exact_patience_early_stop(tmp_path, cfg, tiny):
    output = tmp_path / 'early-complete'
    output.mkdir()
    identity = identity_for(cfg)
    metrics = {'macro5_em1': .3, 'macro5_hit5': .5}
    checkpoint = {'kind': 'best', 'schema_version': run_state.SCHEMA_VERSION,
                  'identity': identity, 'epoch': 1, 'metrics': metrics,
                  'model_config': tiny.config.to_dict(), 'model': tiny.state_dict()}
    best = run_state.stage_best(output, checkpoint, [], engine.save_checkpoint)
    history = []
    for epoch in range(1, 7):
        score = metrics if epoch == 1 else {'macro5_em1': .1, 'macro5_hit5': .2}
        history.append({'epoch': epoch, 'step': run_state.steps_for(cfg, epoch),
                        'validation': score, 'new_best': epoch == 1})
    state = {'kind': 'training', 'schema_version': run_state.SCHEMA_VERSION,
             'identity': identity, 'epoch': 6, 'step': run_state.steps_for(cfg, 6),
             'history': history, 'best': best}
    engine.save_checkpoint(output / 'last.pt', state)
    write(output / 'run.json', {'kind': 'formal', 'identity': identity})
    run_state.complete_run(output, identity, 6, state['step'], history, best, 'early_stopping')
    loaded, marker = run_state.require_completed_run(output / 'best.pt')
    assert loaded['epoch'] == 1 and marker['epochs_ran'] == 6
    assert marker['stopping']['epochs_without_improvement'] == 5


def test_final_gate_rejects_nonformal_hyperparameters(tmp_path, cfg, tiny):
    bad = dict(cfg, learning_rate=1e-4)
    out, _ = make_completed_fixture(tmp_path, bad, tiny)
    with pytest.raises(ValueError, match='learning_rate'):
        run_state.require_completed_run(out / 'best.pt')


@pytest.mark.parametrize('change', ['missing', 'partial', 'steps', 'identity', 'epoch', 'stopping',
                                    'hash', 'history', 'best_metrics', 'checkpoint'])
def test_final_gate_rejects_wrong_completion(tmp_path, cfg, tiny, change):
    out, _ = make_completed_fixture(tmp_path, cfg, tiny)
    marker_path = out / 'TRAINING_COMPLETE.json'
    marker = read(marker_path)
    if change == 'missing':
        marker_path.rename(out / 'marker.backup.json')
    elif change == 'partial':
        marker['epochs_ran'] = 1; write(marker_path, marker)
    elif change == 'steps':
        marker['steps'] -= 1; write(marker_path, marker)
    elif change == 'identity':
        marker['identity_sha256'] = 'different-run'; write(marker_path, marker)
    elif change == 'epoch':
        marker['best']['epoch'] = 18; write(marker_path, marker)
    elif change == 'stopping':
        marker['stopping']['reason'] = 'early_stopping'; write(marker_path, marker)
    elif change == 'hash':
        marker['artifacts']['best.pt'] = 'not-a-checksum'; write(marker_path, marker)
    elif change == 'history':
        write(out / 'history.json', read(out / 'history.json')[:-1])
    elif change == 'best_metrics':
        write(out / 'best_metrics.json', {'epoch': 1})
    elif change == 'checkpoint':
        point = engine.load_checkpoint(out / 'best.pt'); point['identity']['seed'] = 2026
        engine.save_checkpoint(out / 'best.pt', point)
    with pytest.raises((ValueError, FileNotFoundError)):
        run_state.require_completed_run(out / 'best.pt')


def test_cli_partial_marker_rejected_before_test_read(cfg, tmp_path, tiny, monkeypatch):
    out, _ = make_completed_fixture(tmp_path, cfg, tiny)
    marker = read(out / 'TRAINING_COMPLETE.json'); marker['epochs_ran'] = 1
    write(out / 'TRAINING_COMPLETE.json', marker)
    monkeypatch.setattr(engine, 'benchmark', lambda *_: pytest.fail('Test must not be opened'))
    monkeypatch.setattr(engine, 'inspect_checkpoint', lambda *_: pytest.fail('Model must not load'))
    monkeypatch.setattr(sys, 'argv', ['run.py', 'test', '--device', 'cpu', '--confirm-final',
                                    '--checkpoint', str(out / 'best.pt'), '--output', str(tmp_path / 'test_out')])
    with pytest.raises(ValueError, match='optimizer-step'):
        cli.main(Path(__file__).resolve().parents[1])
    assert not (tmp_path / 'test_out').exists()


def test_test_api_requires_confirmation_even_without_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'inspect_checkpoint', lambda *_: pytest.fail('No model access without confirmation'))
    with pytest.raises(ValueError, match='confirm_final'):
        engine.evaluate_checkpoint(Path('.'), Path('.'), tmp_path / 'best.pt', tmp_path / 'eval',
                                   torch.device('cpu'), ['test_random'])


@pytest.mark.parametrize('is_test', [False, True])
def test_standalone_evaluation_records_actual_environment(tmp_path, cfg, tiny, monkeypatch, is_test):
    # Synthetic fixtures only: never read the frozen Test questions or answers.
    out, identity = make_completed_fixture(tmp_path, cfg, tiny)
    monkeypatch.setattr(engine, 'verify_data', lambda _: identity['data_sha256'])
    monkeypatch.setattr(engine, 'source_fingerprint', lambda _: identity['code_sha256'])
    monkeypatch.setattr(engine, 'Codec', MiniCodec)
    opened = []

    def benchmark(_, suite):
        opened.append(suite)
        return [], {}

    monkeypatch.setattr(engine, 'benchmark', benchmark)
    monkeypatch.setattr(engine, 'evaluate', lambda *_: ({'macro5_em1': .1, 'macro5_hit5': .2}, []))
    torch.use_deterministic_algorithms(False)
    suites = cfg['test_suites'] if is_test else ['dev_full']
    if not is_test:
        # Dev must remain usable during training, with no final completion marker.
        (out / 'TRAINING_COMPLETE.json').rename(out / 'marker.backup.json')
    engine.evaluate_checkpoint(tmp_path, tmp_path, out / 'best.pt', tmp_path / 'eval',
                               torch.device('cpu'), suites, confirm_final=is_test)
    report = read(tmp_path / 'eval/metrics.json')
    assert opened == suites
    assert runtime.require_evaluation_report(report) == runtime.evaluation_environment('cpu', cfg)
    expected = digest(read(out / 'TRAINING_COMPLETE.json')) if is_test else None
    assert report['completion_sha256'] == expected
    assert report['checkpoint_sha256'] == sha(out / 'best.pt')


def test_final_test_refuses_run_with_active_writer(tmp_path, cfg, tiny, monkeypatch):
    import fcntl
    out, _ = make_completed_fixture(tmp_path, cfg, tiny)
    monkeypatch.setattr(engine, 'benchmark', lambda *_: pytest.fail('No Test read while training holds lock'))
    monkeypatch.setattr(engine, 'require_completed_run', lambda *_: pytest.fail('Gate must wait for writer lock release'))
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            engine.evaluate_checkpoint(tmp_path, tmp_path, out / 'best.pt', tmp_path / 'eval',
                                       torch.device('cpu'), cfg['test_suites'], confirm_final=True)
    assert not (tmp_path / 'eval').exists()


@pytest.mark.parametrize('failure', ['pending_weight', 'pending_predictions', 'before_last', 'after_last',
                                    'publish_weight', 'publish_predictions', 'history_json', 'before_complete'])
def test_recovery_across_final_commit_windows(tiny, cfg, tmp_path, monkeypatch, failure):
    config = dict(cfg, max_epochs=3, samples_per_epoch=5, effective_batch_size=4,
                  micro_batch_size=2, precision='fp32')
    identity = identity_for(config)
    write(tmp_path / 'benchmark/protocol.json', {'validation_length_counts': {str(i): 1 for i in range(1, 6)}})
    monkeypatch.setattr(engine, 'Codec', MiniCodec); monkeypatch.setattr(engine, 'SpanSampler', MiniSampler)
    monkeypatch.setattr(engine, 'benchmark', lambda *_: ([{'gap_length': i} for i in range(1, 6)], {}))
    monkeypatch.setattr(engine, 'load_pretrained', lambda *_: (copy.deepcopy(tiny), {}))
    evaluation_index = [0]

    def evaluate(*_):
        evaluation_index[0] += 1
        m = {'macro5_em1': evaluation_index[0]/10, 'macro5_hit5': evaluation_index[0]/10}
        return m, [{'audit_epoch': evaluation_index[0]}]
    monkeypatch.setattr(engine, 'evaluate', evaluate)
    uninterrupted, interrupted = tmp_path / 'continuous', tmp_path / 'interrupted'
    uninterrupted.mkdir(); interrupted.mkdir()
    engine._train_locked(tmp_path, tmp_path, tmp_path, uninterrupted, config, 42, torch.device('cpu'), False, identity)
    evaluation_index[0] = 0
    real_save, real_rows = engine.save_checkpoint, run_state.write_rows
    real_publish, real_write, real_complete = run_state._publish_matching, run_state.write, engine.complete_run

    def fail():
        raise InterruptedError('Injected final-epoch fault: ' + failure)

    def save(path, state):
        if state.get('epoch') == 3 and path.name == 'last.pt' and failure == 'before_last':
            fail()
        real_save(path, state)
        if state.get('epoch') == 3 and ((path.name == 'best.pending.pt' and failure == 'pending_weight') or
                                      (path.name == 'last.pt' and failure == 'after_last')):
            fail()

    def rows(path, values):
        real_rows(path, values)
        if evaluation_index[0] == 3 and failure == 'pending_predictions':
            fail()

    def publish(root, final, pending, expected):
        real_publish(root, final, pending, expected)
        if evaluation_index[0] == 3 and ((final == 'best.pt' and failure == 'publish_weight') or
                                       (final == 'dev_best_predictions.jsonl' and failure == 'publish_predictions')):
            fail()

    def write_json(path, obj):
        real_write(path, obj)
        if evaluation_index[0] == 3 and Path(path).name == 'history.json' and failure == 'history_json':
            fail()

    def complete(*args):
        if failure == 'before_complete':
            fail()
        return real_complete(*args)

    with monkeypatch.context() as m:
        m.setattr(engine, 'save_checkpoint', save); m.setattr(run_state, 'write_rows', rows)
        m.setattr(run_state, '_publish_matching', publish); m.setattr(run_state, 'write', write_json)
        m.setattr(engine, 'complete_run', complete)
        with pytest.raises(InterruptedError):
            engine._train_locked(tmp_path, tmp_path, tmp_path, interrupted, config, 42, torch.device('cpu'), False, identity)
    committed = engine.load_checkpoint(interrupted / 'last.pt')
    evaluation_index[0] = committed['epoch']
    engine._train_locked(tmp_path, tmp_path, tmp_path, interrupted, config, 42, torch.device('cpu'), True, identity)
    a, b = [engine.load_checkpoint(d / 'last.pt') for d in [uninterrupted, interrupted]]
    assert a['epoch'] == b['epoch'] == 3 and a['step'] == b['step'] == 6
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    assert a['rng']['sampler'] == b['rng']['sampler'] and a['scheduler'] == b['scheduler']
    history = read(interrupted / 'history.json')
    assert [r['epoch'] for r in history] == [1, 2, 3]
    assert read(interrupted / 'best_metrics.json') == b['best']
    marker = read(interrupted / 'TRAINING_COMPLETE.json')
    assert marker['max_epochs'] == 3 and marker['epochs_ran'] == 3
    assert marker['steps'] == 6 and marker['best']['epoch'] == 3
    assert marker['stopping']['reason'] == 'max_epochs'
    for name, expected in marker['artifacts'].items():
        assert sha(interrupted / name) == expected


def test_formal_test_rejects_partial_manuscript_suite_list(tmp_path, monkeypatch):
    if hasattr(engine, 'require_formal_cuda'):
        monkeypatch.setattr(engine, 'require_formal_cuda', lambda device, *_: torch.device(device))
    with pytest.raises(ValueError, match='current manuscript suites'):
        engine.evaluate_checkpoint(Path('.'), Path('.'), tmp_path / 'best.pt', tmp_path / 'partial_eval',
                                   torch.device('cpu'), ['test_random'], confirm_final=True)
