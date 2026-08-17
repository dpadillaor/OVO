"""Load fusion evaluation artifacts at scene / experiment / cross-experiment levels."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import pandas as pd

OUTPUT_ROOT = pathlib.Path(__file__).resolve().parents[3] / "data" / "output" / "Replica"


@dataclass
class SceneFusionData:
    """All fusion-eval artifacts for a single scene."""

    exp_id: str
    scene: str
    summary: dict
    instance_stats: pd.DataFrame
    decisions: pd.DataFrame

    @property
    def verdicts(self) -> dict:
        return self.summary.get("verdicts", {})

    @property
    def counts(self) -> dict:
        return self.verdicts.get("counts", {})

    @property
    def rates(self) -> dict:
        return self.verdicts.get("rates", {})

    @property
    def by_epoch(self) -> dict[str, dict]:
        return self.verdicts.get("by_epoch", {})

    @property
    def by_criterion(self) -> list[dict]:
        return self.verdicts.get("by_criterion", [])

    @property
    def totals(self) -> dict:
        return self.verdicts.get("totals", {})

    @property
    def run_info(self) -> dict:
        return self.summary.get("run", {})

    @property
    def eval_info(self) -> dict:
        return self.summary.get("evaluation", {})

    def agnostic_impact(self, mode: str = "objects") -> dict:
        return self.summary.get("agnostic_impact", {}).get(mode, {})

    @property
    def non_background_stats(self) -> pd.DataFrame:
        mask = ~self.instance_stats["name"].str.match(r"^class\d*$", na=False)
        return self.instance_stats[mask]


def discover_scenes(exp_path: str | pathlib.Path) -> list[str]:
    p = pathlib.Path(exp_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {p}")
    scenes = []
    for entry in sorted(p.iterdir()):
        if entry.is_dir() and (entry / "fusion" / "fusion_LC" / "fusion_eval_summary.json").exists():
            scenes.append(entry.name)
    return scenes


def load_scene(exp_path: str | pathlib.Path, scene: str) -> SceneFusionData:
    base = pathlib.Path(exp_path) / scene / "fusion" / "fusion_LC"

    summary_path = base / "fusion_eval_summary.json"
    stats_path = base / "fusion_instance_stats.csv"
    decisions_path = base / "fusion_decisions_eval.csv"

    missing = []
    for p in [summary_path, stats_path, decisions_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(f"Missing artifacts for {exp_path}/{scene}: {missing}")

    summary = json.loads(summary_path.read_text())
    instance_stats = pd.read_csv(stats_path)
    decisions = pd.read_csv(decisions_path)

    return SceneFusionData(
        exp_id=pathlib.Path(exp_path).name,
        scene=scene,
        summary=summary,
        instance_stats=instance_stats,
        decisions=decisions,
    )


def load_experiment(exp_path: str | pathlib.Path) -> list[SceneFusionData]:
    scenes = discover_scenes(exp_path)
    if not scenes:
        raise FileNotFoundError(f"No scenes with fusion_eval_summary.json in {exp_path}")
    return [load_scene(exp_path, s) for s in scenes]


def load_experiments(exp_paths: list[str | pathlib.Path]) -> dict[str, list[SceneFusionData]]:
    return {pathlib.Path(p).name: load_experiment(p) for p in exp_paths}


def pool_verdicts(scenes: list[SceneFusionData]) -> dict:
    pooled = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "pairs_scored": 0, "pairs_total": 0}
    for s in scenes:
        pooled["TP"] += s.counts.get("TP", 0)
        pooled["FP"] += s.counts.get("FP", 0)
        pooled["FN"] += s.counts.get("FN", 0)
        pooled["TN"] += s.counts.get("TN", 0)
        pooled["pairs_scored"] += s.eval_info.get("pairs_scored", 0)
        pooled["pairs_total"] += s.eval_info.get("pairs_total", 0)
    tp, fp, fn = pooled["TP"], pooled["FP"], pooled["FN"]
    pooled["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pooled["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pooled["f1"] = (2 * pooled["precision"] * pooled["recall"]
                    / (pooled["precision"] + pooled["recall"])
                    if (pooled["precision"] + pooled["recall"]) > 0 else 0.0)
    return pooled


_CASCADE_KEYS = ("eval", "pass", "reject", "TP", "FP", "FN", "TN")


def _pool_criterion_lists(lists: list[list[dict]]) -> list[dict]:
    """Micro-average a set of by_criterion cascades: sum each gate's counts,
    aligned by cascade position. A list whose gate name diverges from the first
    non-empty list's is skipped for that gate."""
    lists = [bc for bc in lists if bc]
    if not lists:
        return []
    template = lists[0]
    pooled = []
    for i, g in enumerate(template):
        agg = {"criterion": g["criterion"], **{k: 0 for k in _CASCADE_KEYS}}
        for bc in lists:
            if i < len(bc) and bc[i].get("criterion") == g["criterion"]:
                for k in _CASCADE_KEYS:
                    agg[k] += bc[i].get(k, 0)
        # Rates are recomputed from the pooled counts (micro-average), never summed.
        tp, fp, fn = agg["TP"], agg["FP"], agg["FN"]
        agg["precision"] = round(tp / (tp + fp), 4) if tp + fp else None
        agg["recall"] = round(tp / (tp + fn), 4) if tp + fn else None
        p, r = agg["precision"], agg["recall"]
        agg["f1"] = round(2 * p * r / (p + r), 4) if p and r else None
        pooled.append(agg)
    return pooled


def pool_by_criterion(scenes: list[SceneFusionData]) -> list[dict]:
    """Micro-average cascade over scenes (all runs share the chain), shaped for
    gate_sankey."""
    return _pool_criterion_lists([s.by_criterion for s in scenes])


def pool_by_criterion_by_epoch(scenes: list[SceneFusionData]) -> dict[str, list[dict]]:
    """Per drift epoch, the micro-average cascade pooled across scenes.
    Returns {epoch: by_criterion}; only epochs present in some scene appear."""
    epochs = []
    for s in scenes:
        for ep in s.by_epoch:
            if ep not in epochs:
                epochs.append(ep)
    out = {}
    for ep in epochs:
        pooled = _pool_criterion_lists(
            [s.by_epoch[ep].get("by_criterion", []) for s in scenes if ep in s.by_epoch])
        if pooled:
            out[ep] = pooled
    return out


def pool_counts_by_epoch(scenes: list[SceneFusionData]) -> dict[str, dict]:
    """Per drift epoch, the confusion counts (TP/FP/FN/TN) summed across scenes.
    Shaped as {epoch: {"counts": {...}}} to feed confusion_donut_by_epoch."""
    epochs = []
    for s in scenes:
        for ep in s.by_epoch:
            if ep not in epochs:
                epochs.append(ep)
    out = {}
    for ep in epochs:
        agg = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for s in scenes:
            c = s.by_epoch.get(ep, {}).get("counts", {})
            for k in agg:
                agg[k] += c.get(k, 0)
        out[ep] = {"counts": agg}
    return out


def pool_instance_stats(scenes: list[SceneFusionData]) -> pd.DataFrame:
    frames = []
    for s in scenes:
        df = s.instance_stats.copy()
        df["exp_id"] = s.exp_id
        df["scene"] = s.scene
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
