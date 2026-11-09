import os
from pathlib import Path
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'shared'))

from d1.cli import main

if __name__ == '__main__':
    main(ROOT)
