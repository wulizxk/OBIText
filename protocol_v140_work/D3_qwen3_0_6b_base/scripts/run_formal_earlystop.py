#!/usr/bin/env python3
"""Thin, immutable launcher for the verified D3-A formal controller."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]

from d3.formal_controller import main


# Patched only after deployment and final source-fingerprint verification. The
# launcher itself is deliberately outside that fingerprint to avoid recursion.
CODE_SHA256 = '0a809562c8fec4309540bd186a0fc6c65a79bbf9166b433a087a4984d8e6b4dc'


if __name__ == '__main__':
    raise SystemExit(main(expected_code_sha=CODE_SHA256))
