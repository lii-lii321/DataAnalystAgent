"""Run the full pytest suite for this repository, regardless of caller cwd."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

raise SystemExit(pytest.main([str(ROOT), "-q"]))
