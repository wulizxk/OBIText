import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'shared'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


@pytest.fixture(scope='session')
def cfg():
    import json
    return json.loads((ROOT / 'configs/ed2a.json').read_text())


@pytest.fixture(scope='session')
def data_root():
    path = Path(os.environ.get('ED2_DATA', str(ROOT.parent / 'shared/data/v1.5.0')))
    if not path.is_dir():
        pytest.skip('Set ED2_DATA to the frozen v1.5.0 directory')
    return path


@pytest.fixture(scope='session')
def codec(data_root, cfg):
    from ed2.data import Codec
    return Codec(data_root, cfg)


@pytest.fixture
def tiny():
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
    torch.manual_seed(7)
    config = T5Config(vocab_size=40, d_model=16, num_layers=2, num_decoder_layers=2,
                       num_heads=2, d_kv=8, d_ff=32, dropout_rate=0.0,
                       tie_word_embeddings=False, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    config._attn_implementation = 'eager'
    return T5ForConditionalGeneration(config)


@pytest.fixture(scope='session')
def artifact_template(tmp_path_factory, cfg):
    from test_review_repairs import build_artifact_fixture
    return build_artifact_fixture(tmp_path_factory.mktemp('synthetic_artifact_template'), cfg)


@pytest.fixture
def artifact_bundle(artifact_template, tmp_path, monkeypatch):
    import shutil
    from ed2 import evaluation_artifacts
    output = tmp_path / 'synthetic_outputs'
    shutil.copytree(artifact_template['outputs'], output)
    monkeypatch.setattr(evaluation_artifacts, 'DATA_MANIFEST_SHA256', artifact_template['manifest_sha'])
    import benchmark_protocol_v140 as protocol
    specs = artifact_template['suite_specs']
    lengths = {suite: spec['lengths'] for suite, spec in specs.items()}
    counts = {suite: sum(group.values()) for suite, group in lengths.items()}
    modes = {}
    for suite, group in lengths.items():
        modes[suite] = {}
        for length, count in group.items():
            mode = length if length in ('1', '2') else '3-5'
            modes[suite][mode] = modes[suite].get(mode, 0) + count
    monkeypatch.setattr(protocol, 'DATA_MANIFEST_SHA256', artifact_template['manifest_sha'])
    monkeypatch.setattr(protocol, 'EVALUATION_SUITES', specs)
    monkeypatch.setattr(protocol, 'EXPECTED_LENGTH_COUNTS', lengths)
    monkeypatch.setattr(protocol, 'EXPECTED_MODE_COUNTS', modes)
    monkeypatch.setattr(protocol, 'SUITE_QUESTION_COUNTS', counts)
    monkeypatch.setattr(evaluation_artifacts, 'SUITE_QUESTION_COUNTS', counts)
    return dict(artifact_template, outputs=output,
                directories=[output / f'ed2a_v150_s{s}_test' for s in (42, 3407, 2026)])
