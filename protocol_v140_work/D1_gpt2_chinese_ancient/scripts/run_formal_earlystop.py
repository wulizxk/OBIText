#!/usr/bin/env python3
"""Thin, immutable launcher for the verified D1-A formal controller."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]

from d1.formal_controller import main


# Patched only after deployment and final source-fingerprint verification. The
# launcher itself is deliberately outside that fingerprint to avoid recursion.
CODE_SHA256 = 'f6a7629b18dad026926b062189dc4d6cc5f7dc6bbb006bb981895686091919d6'


if __name__ == '__main__':
    raise SystemExit(main(expected_code_sha=CODE_SHA256))
