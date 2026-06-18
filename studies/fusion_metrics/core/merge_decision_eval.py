"""Pair analysis: dominant GT instance under each prediction + merge/split scoring."""
from __future__ import annotations

from enum import Enum
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from core.loaders import (
    SceneData,
    FusionDecision,
    instance_mask,
    gt_ids_under_mask,
    obj_id_to_column,
)

class Verdict(str, Enum):
    """Confusion-matrix outcome of a merge/split decision (positive = MERGE)."""

    TP = "TP"  # correct merge
    FP = "FP"  # over-merge: fused two different objects
    FN = "FN"  # over-split: kept one object apart
    TN = "TN"  # correct separation


@dataclass
class PairAnalysis:
    """How two predictions fall in the GT. dominant_gt_* is None if all-void."""

    col_a: int
    col_b: int
    mask_a: np.ndarray
    mask_b: np.ndarray
    gt_under_a: dict[int, int]
    gt_under_b: dict[int, int]
    dominant_gt_a: int | None
    dominant_gt_b: int | None

    @property
    def same_object(self) -> bool:
        """True when both predictions land on the same dominant GT instance."""
        return (
            self.dominant_gt_a is not None
            and self.dominant_gt_a == self.dominant_gt_b
        )

@dataclass
class EvaluatedPair:
    """A fusion decision scored against the GT: decision + GT truth + verdict."""

    decision: FusionDecision
    same_object: bool
    verdict: Verdict

    @property
    def group(self) -> str:
        """Bucket for the summary: 'ACCEPTED' or 'REJECTED/<reason>'."""
        if self.decision.merged:
            return "ACCEPTED"
        return f"REJECTED/{self.decision.reason or 'unknown'}"


def classify_merge_decision(same_object: bool, merged: bool) -> Verdict:
    """Score a merge/split decision (positive = MERGE) -> TP / FP / FN / TN."""
    if merged and same_object:
        return Verdict.TP
    if merged and not same_object:
        return Verdict.FP
    if not merged and same_object:
        return Verdict.FN
    return Verdict.TN


def dominant_gt(scene: SceneData, mask: np.ndarray) -> int | None:
    """Majority (most-covered) GT instance id under ``mask``, or ``None``."""
    return next(iter(gt_ids_under_mask(scene, mask)), None)


def analyze_pair(scene: SceneData, col_a: int, col_b: int) -> PairAnalysis:
    """Compare predicted instances ``col_a`` and ``col_b`` against the ground truth."""
    mask_a = instance_mask(scene, col_a)
    mask_b = instance_mask(scene, col_b)
    gt_a = gt_ids_under_mask(scene, mask_a)
    gt_b = gt_ids_under_mask(scene, mask_b)
    return PairAnalysis(
        col_a=col_a,
        col_b=col_b,
        mask_a=mask_a,
        mask_b=mask_b,
        gt_under_a=gt_a,
        gt_under_b=gt_b,
        dominant_gt_a=next(iter(gt_a), None),
        dominant_gt_b=next(iter(gt_b), None),
    )

def evaluate_decision(
    scene: SceneData, col_obj_ids: np.ndarray, decision: FusionDecision
) -> EvaluatedPair:
    """Score one fusion decision against the GT. Raises if an obj_id is absent."""
    col_a = obj_id_to_column(col_obj_ids, decision.i1)
    col_b = obj_id_to_column(col_obj_ids, decision.i2)
    pair = analyze_pair(scene, col_a, col_b)
    verdict = classify_merge_decision(pair.same_object, decision.merged)
    return EvaluatedPair(decision=decision, same_object=pair.same_object, verdict=verdict)


def _ratio(num: int, den: int) -> float | None:
    """num/den rounded to 4 dp, or None when undefined (den == 0)."""
    return round(num / den, 4) if den else None


def summarize(
    pairs: list[EvaluatedPair],
    *,
    experiment: str,
    scene: str,
    frame_id: int,
    n_instances_pre: int,
    n_instances_post: int,
) -> dict:
    """Aggregate scored pairs into the report dict (counts, rates, by_group)."""
    counts = Counter(p.verdict for p in pairs)
    tp, fp, fn, tn = (counts[v] for v in Verdict)

    by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        by_group[p.group][p.verdict] += 1
    by_group_out = {
        g: {"total": sum(v.values()), **dict(sorted(v.items()))}
        for g, v in sorted(by_group.items())
    }

    return {
        "experiment": experiment,
        "scene": scene,
        "frame_id": frame_id,
        "n_instances_pre": n_instances_pre,
        "n_instances_post": n_instances_post,
        "n_pairs": len(pairs),
        "n_accepted": sum(1 for p in pairs if p.decision.merged),
        "n_rejected": sum(1 for p in pairs if not p.decision.merged),
        "counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "rates": {
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        },
        "by_group": by_group_out,
    }
