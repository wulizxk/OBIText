"""Thin entry point for the fingerprinted E2-A formal controller."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]

from e2.formal_controller import main


CODE_SHA = '52dfe6c67b38f0ed4e7564f1ea68ff5e817d622485552ae84e5d92d01a90ced5'


if __name__ == '__main__':
    sys.exit(main(expected_code_sha=CODE_SHA))
