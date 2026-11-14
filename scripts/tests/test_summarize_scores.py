import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'summarize_scores.py'


def write_run(directory, *, model='E1', seed=42, metadata=True):
    directory.mkdir()
    prediction = directory / 'scored.jsonl'
    row = {
        'example_id': 'question1',
        'source_id': 'source1',
        'group_id': 'group1',
        'gap_length': 1,
        'gold_units': ['甲'],
        'candidates': [['甲'], ['乙']],
        'em1': 1,
        'hit5': 1,
        'correct_positions': 1,
        'gold_rank_full_vocabulary': 1,
        'reciprocal_rank': 1.0,
    }
    prediction.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    if metadata:
        report = {
            'model': model,
            'seed': seed,
            'scored_sha256': hashlib.sha256(prediction.read_bytes()).hexdigest(),
        }
        (directory / 'metrics.json').write_text(json.dumps(report), encoding='utf-8')
    return prediction


def run_summary(paths, output):
    return subprocess.run(
        [sys.executable, '-B', str(SCRIPT), '--scored', *map(str, paths),
         '--output', str(output)],
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize('alias', ['same_path', 'symlink', 'hardlink'])
def test_duplicate_prediction_file_is_rejected(tmp_path, alias):
    prediction = write_run(tmp_path / 'run')
    second = prediction
    if alias == 'symlink':
        second = tmp_path / 'alias.jsonl'
        second.symlink_to(prediction)
    elif alias == 'hardlink':
        second = tmp_path / 'alias.jsonl'
        os.link(prediction, second)
    output = tmp_path / 'summary.json'
    result = run_summary([prediction, second], output)
    assert result.returncode != 0
    assert 'same prediction file' in result.stderr
    assert not output.exists()


def test_same_model_and_seed_is_rejected(tmp_path):
    first = write_run(tmp_path / 'first')
    second = write_run(tmp_path / 'second')
    result = run_summary([first, second], tmp_path / 'summary.json')
    assert result.returncode != 0
    assert 'same model and seed' in result.stderr


def test_different_models_are_rejected(tmp_path):
    first = write_run(tmp_path / 'first', model='E1', seed=42)
    second = write_run(tmp_path / 'second', model='B4', seed=3407)
    result = run_summary([first, second], tmp_path / 'summary.json')
    assert result.returncode != 0
    assert 'different models' in result.stderr


def test_identical_predictions_from_distinct_seeds_are_accepted(tmp_path):
    predictions = [write_run(tmp_path / str(seed), seed=seed) for seed in (42, 3407, 2026)]
    output = tmp_path / 'summary.json'
    result = run_summary(predictions, output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['runs'] == 3
    assert report['run_identities_verified'] is True
    assert report['across_runs']['pooled']['em1'] == {'mean': 1.0, 'sample_sd': 0.0}


def test_missing_run_metadata_is_reported(tmp_path):
    prediction = write_run(tmp_path / 'legacy', metadata=False)
    output = tmp_path / 'summary.json'
    result = run_summary([prediction], output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['run_identities_verified'] is False
    assert report['per_run'][0]['identity'] is None


def test_metadata_checksum_must_match_prediction(tmp_path):
    prediction = write_run(tmp_path / 'run')
    metadata = prediction.parent / 'metrics.json'
    report = json.loads(metadata.read_text())
    report['scored_sha256'] = '0' * 64
    metadata.write_text(json.dumps(report))
    result = run_summary([prediction], tmp_path / 'summary.json')
    assert result.returncode != 0
    assert 'checksum differs' in result.stderr


def test_legacy_experiment_identity_is_accepted(tmp_path):
    first = write_run(tmp_path / 'legacy', seed=42)
    metadata = first.parent / 'metrics.json'
    report = json.loads(metadata.read_text())
    report.pop('model')
    report['identity'] = {'cfg': {'experiment': 'E1-A'}}
    metadata.write_text(json.dumps(report))
    second = write_run(tmp_path / 'current', model='E1', seed=3407)
    output = tmp_path / 'summary.json'
    result = run_summary([first, second], output)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['run_identities_verified'] is True


def test_boolean_seed_is_rejected(tmp_path):
    prediction = write_run(tmp_path / 'run', seed=True)
    result = run_summary([prediction], tmp_path / 'summary.json')
    assert result.returncode != 0
    assert 'Invalid model or seed' in result.stderr
