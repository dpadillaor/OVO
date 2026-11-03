import sys
from pathlib import Path

# The ovo package has no __init__.py: make the repo root importable when running plain `pytest`.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
