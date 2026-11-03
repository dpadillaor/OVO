"""The mapping modules import on a CPU machine without the perception_models submodule (`core`).

Run in a fresh interpreter: modules cached by other tests would hide a module-level import of `core`,
and a `core` installed on a dev machine is blocked so the check means the same everywhere.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = ["ovo.utils.clip_utils", "ovo.entities.ovo", "ovo.entities.ovomapping"]


@pytest.mark.parametrize("module", MODULES)
def test_imports_without_perception_models(module):
    code = (
        "import sys\n"
        "sys.modules['core'] = None  # makes `import core...` raise ImportError\n"
        f"import {module}\n"
        "assert not any(m == 'core' or m.startswith('core.') for m, v in sys.modules.items() if v is not None)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
