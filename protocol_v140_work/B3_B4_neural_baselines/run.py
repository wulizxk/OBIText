#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "shared"))

from b34.cli import main

if __name__ == "__main__":
    main(PROJECT)
