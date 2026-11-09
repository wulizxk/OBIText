"""Epoch-boundary commit, recovery and final-Test provenance checks."""
import math
import os
import re
from contextlib import contextmanager, ExitStack
from pathlib import Path
from benchmark_protocol_v140 import evaluation_config

import torch

from .io import digest, read, sha, write, write_rows
from .metrics import better
from .protocol import validate_formal_config

SCHEMA_VERSION = 4


def steps_for(cfg, epoch):
    return epoch * math.ceil(cfg['samples_per_epoch'] / cfg['effective_batch_size'])


def validate_history(cfg, epoch, step, history, best):
    if type(epoch) is not int or not 0 <= epoch <= cfg['max_epochs'] or type(step) is not int or step != steps_for(cfg, epoch):
        raise ValueError('Committed epoch/step count is inconsistent with the training budget')
    if len(history) != epoch:
        raise ValueError('Committed history does not cover every completed epoch')
    selected = None
    for index, item in enumerate(history, 1):
        if item['epoch'] != index or item['step'] != steps_for(cfg, index):
            raise ValueError('History epoch/step order is inconsistent')
        metrics = item['validation']
        if any(type(metrics[k]) not in (float, int) or not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1
               for k in ['macro5_em1', 'macro5_hit5']):
            raise ValueError('Invalid checkpoint-selection metrics')
        improved = selected is None or better(metrics, selected['metrics'])
        if item.get('new_best') is not improved:
            raise ValueError('History new_best flags disagree with checkpoint selection')
        if improved:
            selected = {'epoch': index, 'metrics': metrics}
        # A later improvement cannot legalize an earlier patience hit.
        if (cfg.get('early_stopping') and index < epoch
                and index - selected['epoch'] >= cfg['early_stopping_patience']):
            raise ValueError(f'History continued after required early stop at epoch {index}')
    if selected is None:
        if best is not None:
            raise ValueError('Epoch zero cannot have a selected best checkpoint')
    elif best is None or selected != {k: best[k] for k in ['epoch', 'metrics']}:
        raise ValueError('Best checkpoint is not the full-Dev winner recorded in history')


def validate_stopping(cfg, epoch, history, best, reason):
    """Validate either a full budget run or exactly patience failed Dev checks."""
    validate_history(cfg, epoch, steps_for(cfg, epoch), history, best)
    if epoch < 1 or best is None:
        raise ValueError('A completed run must contain at least one validated epoch')
    without_improvement = epoch - best['epoch']
    if reason == 'max_epochs':
        if epoch != cfg['max_epochs'] or (cfg.get('early_stopping') and
                                          without_improvement > cfg.get('early_stopping_patience')):
            raise ValueError('max_epochs stop is inconsistent with the budget or earlier patience trigger')
    elif reason == 'early_stopping':
        if (not cfg.get('early_stopping') or epoch >= cfg['max_epochs']
                or without_improvement != cfg.get('early_stopping_patience')):
            raise ValueError('Early stop does not equal the configured consecutive non-improvement patience')
    else:
        raise ValueError('Unknown training stopping reason')
    return without_improvement


def selection_trace(history):
    """Small replayable evidence bound by the completion digest in Test reports."""
    return [{'epoch': row['epoch'], 'step': row['step'], 'new_best': row['new_best'],
             'validation': {k: row['validation'][k] for k in ('macro5_em1', 'macro5_hit5')}}
            for row in history]


def validate_completion_trace(cfg, marker):
    trace = marker.get('selection_history')
    if not isinstance(trace, list) or len(trace) != marker['epochs_ran']:
        raise ValueError('Completion lacks a full replayable selection history')
    best = marker['best']
    reduced_best = {'epoch': best['epoch'],
                    'metrics': {k: best['metrics'][k] for k in ('macro5_em1', 'macro5_hit5')}}
    return validate_stopping(cfg, marker['epochs_ran'], trace, reduced_best, marker['stopping']['reason'])


def stage_best(output, checkpoint, predictions, save_checkpoint):
    """Do not replace committed best artifacts until last.pt commits their hashes."""
    output = Path(output)
    weight_path = output / 'best.pending.pt'
    prediction_path = output / 'dev_best_predictions.pending.jsonl'
    save_checkpoint(weight_path, checkpoint)
    write_rows(prediction_path, predictions)
    return {'epoch': checkpoint['epoch'], 'metrics': checkpoint['metrics'],
            'checkpoint_sha256': sha(weight_path), 'predictions_sha256': sha(prediction_path)}


def _publish_matching(output, final_name, pending_name, expected):
    final, pending = output / final_name, output / pending_name
    if final.is_file() and sha(final) == expected:
        return
    if pending.is_file() and sha(pending) == expected:
        os.replace(pending, final)
        return
    raise ValueError('Cannot recover the committed artifact: ' + final_name)


def reconcile_records(output, identity, epoch, step, history, best):
    output = Path(output)
    validate_history(identity['cfg'], epoch, step, history, best)
    if best is not None:
        _publish_matching(output, 'best.pt', 'best.pending.pt', best['checkpoint_sha256'])
        _publish_matching(output, 'dev_best_predictions.jsonl', 'dev_best_predictions.pending.jsonl', best['predictions_sha256'])
    # This runs even if resume finds the final epoch already completed.
    write(output / 'history.json', history)
    write(output / 'best_metrics.json', best)


def _check_best(checkpoint, identity, best):
    if (checkpoint.get('kind') != 'best' or checkpoint.get('identity') != identity
            or checkpoint.get('epoch') != best['epoch'] or checkpoint.get('metrics') != best['metrics']):
        raise ValueError('Selected checkpoint identity/epoch/metrics do not match this completed run')


def _check_formal_training_execution(run, identity):
    environment = run.get('training_execution')
    if not isinstance(environment, dict):
        raise ValueError('Formal run lacks its actual training execution environment')
    if environment.get('versions') != identity.get('runtime'):
        raise ValueError('Formal training runtime differs from the run identity')
    if environment.get('execution_policy') != identity.get('execution_policy'):
        raise ValueError('Formal training execution policy differs from the run identity')
    if environment.get('hardware', {}).get('type') != 'cuda' or environment.get('effective_precision') != identity['cfg']['precision']:
        raise ValueError('Formal ED2-A completion requires CUDA BF16 training')


def complete_run(output, identity, epoch, step, history, best, stopping_reason):
    output = Path(output)
    without_improvement = validate_stopping(identity['cfg'], epoch, history, best, stopping_reason)
    reconcile_records(output, identity, epoch, step, history, best)
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    _check_best(checkpoint, identity, best)
    del checkpoint
    run = read(output / 'run.json')
    if run.get('kind') != 'formal' or run.get('identity') != identity:
        raise ValueError('Run metadata differs from the committed training state')
    artifacts = {name: sha(output / name) for name in ['last.pt', 'history.json', 'best_metrics.json', 'run.json']}
    artifacts.update({'best.pt': best['checkpoint_sha256'], 'dev_best_predictions.jsonl': best['predictions_sha256']})
    marker = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'identity_sha256': digest(identity),
              'max_epochs': identity['cfg']['max_epochs'], 'epochs_ran': epoch, 'steps': step,
              'stopping': {'reason': stopping_reason,
                           'patience': identity['cfg']['early_stopping_patience'],
                           'epochs_without_improvement': without_improvement},
              'best': best, 'artifacts': artifacts, 'test_evaluated': False,
              'selection_history': selection_trace(history)}
    write(output / 'TRAINING_COMPLETE.json', marker)
    return marker


def require_completed_run(path):
    """Cheap header checks precede hashes/deserialization and any Test read."""
    path = Path(path)
    root = path.parent
    marker = read(root / 'TRAINING_COMPLETE.json')
    run = read(root / 'run.json')
    identity = run.get('identity', {})
    cfg = identity.get('cfg', {})
    validate_formal_config(cfg)
    if run.get('kind') != 'formal' or marker.get('schema_version') != SCHEMA_VERSION or marker.get('status') != 'complete':
        raise ValueError('Formal Test requires a complete, current-schema formal run')
    _check_formal_training_execution(run, identity)
    if (cfg.get('max_epochs') != 50 or cfg.get('early_stopping') is not True
            or cfg.get('early_stopping_patience') != 5 or marker.get('max_epochs') != 50):
        raise ValueError('Formal Test requires max_epochs=50 and early-stopping patience=5')
    epochs_ran = marker.get('epochs_ran')
    if type(epochs_ran) is not int or not 1 <= epochs_ran <= 50 or marker.get('steps') != steps_for(cfg, epochs_ran):
        raise ValueError('Formal Test requires a valid completed optimizer-step count')
    if marker.get('identity_sha256') != digest(identity):
        raise ValueError('Completion marker belongs to a different run identity')
    best = marker.get('best')
    if not isinstance(best, dict) or type(best.get('epoch')) is not int or not 1 <= best['epoch'] <= epochs_ran:
        raise ValueError('Invalid selected-best epoch; it may be earlier than the stopping epoch')
    expected_names = {'last.pt', 'best.pt', 'history.json', 'best_metrics.json', 'run.json', 'dev_best_predictions.jsonl'}
    artifacts = marker.get('artifacts', {})
    if set(artifacts) != expected_names:
        raise ValueError('Incomplete final artifact manifest')
    if artifacts['best.pt'] != best.get('checkpoint_sha256') or artifacts['dev_best_predictions.jsonl'] != best.get('predictions_sha256'):
        raise ValueError('Completion marker best hashes are inconsistent')
    for name, expected in artifacts.items():
        if sha(root / name) != expected:
            raise ValueError('Completed artifact checksum mismatch: ' + name)
    if sha(path) != best['checkpoint_sha256']:
        raise ValueError('Requested checkpoint is not the selected best of this completed run')
    history = read(root / 'history.json')
    if marker.get('selection_history') != selection_trace(history):
        raise ValueError('Completion selection history differs from committed history')
    validate_completion_trace(cfg, marker)
    stopping = marker.get('stopping')
    if not isinstance(stopping, dict) or stopping.get('patience') != 5:
        raise ValueError('Invalid early-stopping completion record')
    without_improvement = validate_stopping(cfg, epochs_ran, history, best, stopping.get('reason'))
    if stopping.get('epochs_without_improvement') != without_improvement:
        raise ValueError('Completion marker non-improvement count is inconsistent')
    if read(root / 'best_metrics.json') != best:
        raise ValueError('Best metrics JSON disagrees with the completion marker')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    _check_best(checkpoint, identity, best)
    return checkpoint, marker


def validate_completion_record(identity, completion):
    """Validate the replayable metadata embedded in Test reports, without loading weights."""
    cfg = validate_formal_config(evaluation_config(identity['cfg']))
    if (not isinstance(completion, dict) or completion.get('schema_version') != SCHEMA_VERSION
            or completion.get('status') != 'complete'
            or completion.get('identity_sha256') != digest(identity)):
        raise ValueError('Training completion is not bound to this evaluation identity')
    epochs, best = completion.get('epochs_ran'), completion.get('best', {})
    if (type(epochs) is not int or not 1 <= epochs <= cfg['max_epochs']
            or type(best.get('epoch')) is not int or not 1 <= best['epoch'] <= epochs
            or completion.get('max_epochs') != cfg['max_epochs']
            or type(completion.get('steps')) is not int or completion['steps'] != steps_for(cfg, epochs)
            or completion.get('test_evaluated') is not False):
        raise ValueError('Invalid training-completion epoch/step/best record')
    names = {'last.pt', 'best.pt', 'history.json', 'best_metrics.json', 'run.json', 'dev_best_predictions.jsonl'}
    artifacts = completion.get('artifacts', {})
    if (set(artifacts) != names or any(not isinstance(v, str) or re.fullmatch(r'[0-9a-f]{64}', v) is None
                                      for v in artifacts.values())
            or artifacts['best.pt'] != best.get('checkpoint_sha256')
            or artifacts['dev_best_predictions.jsonl'] != best.get('predictions_sha256')):
        raise ValueError('Invalid training-completion artifact manifest')
    validate_completion_trace(cfg, completion)
    stopping = completion.get('stopping', {})
    if (stopping.get('patience') != cfg['early_stopping_patience']
            or type(stopping.get('epochs_without_improvement')) is not int
            or stopping['epochs_without_improvement'] != epochs - best['epoch']):
        raise ValueError('Invalid training-completion patience record')
    return cfg


def validate_training_batch(identity, batch):
    """A Test report must carry the same three legal, identity-matched completions."""
    if (not isinstance(batch, dict) or type(batch.get('schema_version')) is not int
            or batch['schema_version'] != 1 or batch.get('seeds') != [42, 3407, 2026]
            or any(type(seed) is not int for seed in batch['seeds'])):
        raise ValueError('Formal Test requires all three training completions')
    base = {k: v for k, v in identity.items() if k != 'seed'}
    if batch.get('identity_without_seed_sha256') != digest(base):
        raise ValueError('Training batch identity differs from the evaluated run')
    completions = batch.get('completions', {})
    if set(completions) != {'42', '3407', '2026'}:
        raise ValueError('Training batch lacks the three distinct required seeds')
    for seed in batch['seeds']:
        validate_completion_record(dict(base, seed=seed), completions[str(seed)])
    return batch


@contextmanager
def completed_training_batch(path, training_runs=None):
    """Hold all three run locks through Test; never infer completion from best.pt alone.

    Custom train --output directories are supported via explicit training_runs.
    Otherwise the other seeds must be siblings named ed2a_v150_s<seed>.
    """
    import fcntl
    path = Path(path).resolve()
    with ExitStack() as stack:
        def lock(root):
            # Parent must already exist. Creating a lock never creates a run.
            handle = stack.enter_context((root / '.run.lock').open('a'))
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        lock(path.parent)
        selected, own_completion = require_completed_run(path)
        identity = selected['identity']
        seed = identity['seed']
        if type(seed) is not int or seed not in (42, 3407, 2026):
            raise ValueError('Unexpected selected training seed')
        if training_runs is None:
            roots = [path.parent] + [path.parent.parent / f'ed2a_v150_s{s}' for s in (42, 3407, 2026) if s != seed]
        else:
            roots = [Path(p).resolve() for p in training_runs]
        if len(roots) != 3 or len(set(roots)) != 3 or path.parent not in roots:
            raise ValueError('Provide exactly three distinct training directories including the selected run')
        base = {k: v for k, v in identity.items() if k != 'seed'}
        completions = {str(seed): own_completion}
        for root in sorted(r for r in roots if r != path.parent):
            if not (root / 'TRAINING_COMPLETE.json').is_file():
                raise ValueError('All three seeds must finish training before Test; missing completion: ' + str(root))
            lock(root)
            checkpoint, marker = require_completed_run(root / 'best.pt')
            other = checkpoint['identity']
            other_seed = other['seed']
            if (type(other_seed) is not int or other_seed not in (42, 3407, 2026)
                    or str(other_seed) in completions or {k: v for k, v in other.items() if k != 'seed'} != base):
                raise ValueError('Mixed identities or duplicate seeds in training batch')
            completions[str(other_seed)] = marker
            del checkpoint
        batch = {'schema_version': 1, 'seeds': [42, 3407, 2026],
                 'identity_without_seed_sha256': digest(base), 'completions': completions}
        validate_training_batch(identity, batch)
        yield selected, own_completion, batch
