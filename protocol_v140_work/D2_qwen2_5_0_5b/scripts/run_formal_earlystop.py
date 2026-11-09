#!/usr/bin/env python3
"""Thin, immutable launcher for the verified D2-A formal controller."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]

from d2.formal_controller import main


# Patched only after deployment and final source-fingerprint verification. The
# launcher itself is deliberately outside that fingerprint to avoid recursion.
CODE_SHA256 = '9d46a22f950b30fd002f27da7121bdd274761fcdcb89585a7fe9c4969012ab47'


if __name__ == '__main__':
    raise SystemExit(main(expected_code_sha=CODE_SHA256))
