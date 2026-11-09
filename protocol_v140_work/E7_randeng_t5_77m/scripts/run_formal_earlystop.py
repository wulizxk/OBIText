"""Default: verified plan only. --execute: sequential GPU-0 seeds and final Test."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'shared'))

from ed1.formal_controller import main

if __name__ == '__main__':
    freeze = json.loads((ROOT / 'verification/source_freeze_v150.json').read_text())
    raise SystemExit(main(expected_code_sha=freeze['code_sha256']))
