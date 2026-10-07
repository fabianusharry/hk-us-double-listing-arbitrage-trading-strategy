"""Shared pytest setup: make the repo root importable so tests can `import config` and `import src.*`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
