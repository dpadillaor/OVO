"""I/O: read the verdicts block (totals / counts / by_criterion) from fusion_eval_summary.json."""
from __future__ import annotations

import json
import pathlib


def load_verdicts(exp_path: pathlib.Path, scene: str) -> dict:
    """The whole ``verdicts`` block from fusion_eval_summary.json. Empty dict if absent."""
    path = pathlib.Path(exp_path) / scene / "fusion" / "fusion_LC" / "fusion_eval_summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("verdicts", {})


def load_by_criterion(exp_path: pathlib.Path, scene: str) -> list[dict]:
    """The by_criterion cascade block from fusion_eval_summary.json. Empty if absent."""
    return load_verdicts(exp_path, scene).get("by_criterion", [])
