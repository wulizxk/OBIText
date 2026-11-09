"""Epoch-boundary commit, recovery and final-Test provenance checks."""
import math
import os
from pathlib import Path

import torch

from .io import digest, read, sha, write, write_rows
from .metrics import better
from .protocol import validate_formal_config

SCHEMA_VERSION = 3


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
        if any(not isinstance(metrics[k], (float, int)) or not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1
               for k in ['macro5_em1', 'macro5_hit5']):
            raise ValueError('Invalid checkpoint-selection metrics')
        improved = selected is None or better(metrics, selected['metrics'])
        if item.get('new_best') is not improved:
            raise ValueError('History new_best flags disagree with checkpoint selection')
        if improved:
            selected = {'epoch': index, 'metrics': metrics}
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
    if environment.get('hardware', {}).get('type') != 'cuda' or environment.get('effective_precision') != 'fp16':
        raise ValueError('Formal E3-A completion requires CUDA FP16 training')


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
              'best': best, 'artifacts': artifacts, 'test_evaluated': False}
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
