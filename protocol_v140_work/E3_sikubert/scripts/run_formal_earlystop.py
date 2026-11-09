#!/usr/bin/env python3
"""Thin, immutable launcher for the verified E3-A formal controller."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]

from e3.formal_controller import main


# Patched only after deployment and final source-fingerprint verification. The
# launcher itself is deliberately outside that fingerprint to avoid recursion.
CODE_SHA256 = '7a5bbd0e2ce586575886aa25d519d33d50131a6e1e4a829e81f832db4b8a327d'


if __name__ == '__main__':
    raise SystemExit(main(expected_code_sha=CODE_SHA256))
