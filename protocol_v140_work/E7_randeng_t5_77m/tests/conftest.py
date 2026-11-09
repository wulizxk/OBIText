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
    return json.loads((ROOT / 'configs/ed1a.json').read_text())


@pytest.fixture(scope='session')
def data_root():
    path = Path(os.environ.get('ED1_DATA', str(ROOT.parent / 'shared/data/v1.5.0')))
    if not path.is_dir():
        pytest.skip('Set ED1_DATA to the frozen v1.5.0 directory')
    return path


@pytest.fixture(scope='session')
def codec(data_root, cfg):
    from ed1.data import Codec
    return Codec(data_root, cfg)


@pytest.fixture
def tiny():
    import torch
    from transformers import MT5Config, MT5ForConditionalGeneration
    torch.manual_seed(7)
    config = MT5Config(vocab_size=40, d_model=16, num_layers=2, num_decoder_layers=2,
                       num_heads=2, d_kv=8, d_ff=32, dropout_rate=0.0,
                       tie_word_embeddings=False, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    config._attn_implementation = 'eager'
    return MT5ForConditionalGeneration(config)
