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
    return json.loads((ROOT / 'configs/d1a.json').read_text())


@pytest.fixture(scope='session')
def data_root():
    path = Path(os.environ.get('D1_DATA', str(ROOT.parent / 'shared/data/v1.5.0')))
    if not path.is_dir():
        pytest.skip('Set D1_DATA to the frozen v1.5.0 directory')
    return path


@pytest.fixture(scope='session')
def codec(data_root, cfg):
    from d1.data import Codec
    return Codec(data_root, cfg)


@pytest.fixture
def tiny():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(7)
    config = GPT2Config(vocab_size=40, n_embd=16, n_layer=2, n_head=2,
                        n_positions=64, n_ctx=64, n_inner=32,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                        pad_token_id=0)
    config._attn_implementation = 'eager'
    return GPT2LMHeadModel(config)
