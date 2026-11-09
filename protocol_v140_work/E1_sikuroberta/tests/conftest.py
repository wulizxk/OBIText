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
    return json.loads((ROOT / 'configs/e1a.json').read_text())


@pytest.fixture(scope='session')
def data_root():
    path = Path(os.environ.get('E1_DATA', str(ROOT.parent / 'shared/data/v1.5.0')))
    if not path.is_dir():
        pytest.skip('Set E1_DATA to the frozen v1.5.0 directory')
    return path


@pytest.fixture(scope='session')
def codec(data_root, cfg):
    from e1.data import Codec
    return Codec(data_root, cfg)


@pytest.fixture
def tiny():
    import torch
    from transformers import BertConfig, BertForMaskedLM
    torch.manual_seed(7)
    config = BertConfig(vocab_size=40, hidden_size=16, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
                        hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    config._attn_implementation = 'eager'
    return BertForMaskedLM(config)
