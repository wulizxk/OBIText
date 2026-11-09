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
    return json.loads((ROOT / 'configs/d3a.json').read_text())


@pytest.fixture(scope='session')
def data_root():
    path = Path(os.environ.get('D3_DATA', str(ROOT.parent / 'shared/data/v1.5.0')))
    if not path.is_dir():
        pytest.skip('Set D3_DATA to the frozen v1.5.0 directory')
    return path


@pytest.fixture(scope='session')
def codec(data_root, cfg):
    from d3.data import Codec
    return Codec(data_root, cfg)


@pytest.fixture
def tiny():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=40, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=4,
                         max_position_embeddings=64,
                         attention_dropout=0.0, pad_token_id=0,
                         bos_token_id=1, eos_token_id=1,
                         tie_word_embeddings=True, use_cache=False)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config)
