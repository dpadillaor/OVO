"""Instance AP of an experiment: reads its predictions and the GT, writes ``instance_ap.json``.

    {"benchmark": "replica",
     "all":    {"agnostic": {"AP", "AP_50", "AP_25"},
                "class":    {"AP", "AP_50", "AP_25", "AP_head", "AP_common", "AP_tail"}},
     "scenes": {"<scene>": <same as "all">}}

A metric is null when it is undefined (no GT instance of any evaluated class).
"""

import json
import math
from pathlib import Path

import numpy as np

from ovo.eval import instance_ap, predictions
from ovo.eval.benchmarks import InstanceBenchmark
from ovo.layout import ExperimentLayout


def evaluate_experiment(
    layout: ExperimentLayout, scenes: list[str], benchmark: InstanceBenchmark, gt_dir: Path
) -> dict:
    if not scenes:
        raise ValueError("no scenes to evaluate")
    scans = {scene: load_scan(layout.instance_pred, Path(gt_dir), scene, benchmark) for scene in scenes}
    result = {
        "benchmark": benchmark.name,
        "all": _metrics(scans, benchmark),
        "scenes": {scene: _metrics({scene: scan}, benchmark) for scene, scan in scans.items()},
    }
    layout.instance_ap.parent.mkdir(parents=True, exist_ok=True)
    layout.instance_ap.write_text(json.dumps(nan_to_none(result), indent=2))
    return result


def load_scan(pred_dir: Path, gt_dir: Path, scene: str, benchmark: InstanceBenchmark) -> instance_ap.Scan:
    gt_file = gt_dir / f"{scene}.txt"
    if not gt_file.is_file():
        raise FileNotFoundError(f"missing instance GT for {scene}: {gt_file}")
    if not (pred_dir / f"{scene}.classes.json").is_file():
        raise FileNotFoundError(f"no instance predictions for {scene} in {pred_dir}; run --segment first")
    preds = predictions.read(pred_dir, scene)
    if preds.class_names != benchmark.class_names:
        raise ValueError(
            f"{scene}: predictions were classified against a different class list than the "
            f"{benchmark.name} benchmark expects; re-run --segment"
        )
    gt_ids = np.loadtxt(gt_file, dtype=np.int64, ndmin=1)
    return instance_ap.Scan(gt_ids=gt_ids, masks=preds.masks, classes=preds.classes, scores=preds.scores)


def _metrics(scans: dict, benchmark: InstanceBenchmark) -> dict:
    splits = {"head": benchmark.head, "common": benchmark.common, "tail": benchmark.tail}
    return {
        "agnostic": instance_ap.evaluate_agnostic(scans, benchmark.class_ids),
        "class": instance_ap.evaluate(scans, benchmark.class_ids, benchmark.class_names, splits),
    }


def nan_to_none(value):
    if isinstance(value, dict):
        return {k: nan_to_none(v) for k, v in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
