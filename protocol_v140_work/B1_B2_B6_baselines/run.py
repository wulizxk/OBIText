#!/usr/bin/env python3
"""CPU-only v1.5.0 statistical baselines. No experiment queue integration."""
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / 'src'), str(root.parent / 'shared')]

if __name__ == '__main__':
    from baselines.runner import main
    main()
