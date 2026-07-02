"""Pair analysis: dominant GT instance under each prediction + merge/split scoring."""
from __future__ import annotations

from enum import Enum
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from core.agnostic_impact.loaders import (
    SceneData,
    FusionDecision,
    instance_mask,
    gt_ids_under_mask,
    obj_id_to_column,
)
from core.agnostic_impact.cascade import cascade_by_criterion

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


class Epoch(str, Enum):
    """Drift epoch of an instance, relative to the single jump's keyframe."""

    PREDRIFT = "predrift"    # created before the jump
    POSTDRIFT = "postdrift"  # created at/after the jump (ghost)


# Pair-class name orders epochs PREDRIFT-first (lexical sort would reverse it).
_EPOCH_ORDER = (Epoch.PREDRIFT, Epoch.POSTDRIFT)


def instance_epoch(created_at_frame: int, jump_frame: int) -> Epoch:
    """Drift epoch of an instance: born at/after the jump frame is POSTDRIFT (ghost)."""
    return Epoch.POSTDRIFT if created_at_frame >= jump_frame else Epoch.PREDRIFT


def epoch_by_obj_id(created_at_frame: dict[int, int], jump_frame: int) -> dict[int, Epoch]:
    """obj_id -> Epoch, derived from each instance's creation frame vs the jump frame."""
    return {oid: instance_epoch(kf, jump_frame) for oid, kf in created_at_frame.items()}


def pair_class(a: Epoch, b: Epoch) -> str:
    """Symmetric drift bucket of a pair: ``{lo}_{hi}`` over the two epochs."""
    lo, hi = sorted((a, b), key=_EPOCH_ORDER.index)
    return f"{lo.value}_{hi.value}"


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


def summarize(pairs: list[EvaluatedPair], chain: list[str] | None = None) -> dict:
    """Verdicts block for scored pairs: totals, counts, rates, by_criterion.

    ``chain`` (ordered criterion names) adds the ``by_criterion`` cascade: each
    gate scored as a classifier on its incoming stream (TP/FP it lets pass +
    TN/FN it rejects + precision/recall/f1). The OR branch of the terminal gate
    lands in its ``accept_modes``. Omitted when ``chain`` is None.
    """
    counts = Counter(p.verdict for p in pairs)
    tp, fp, fn, tn = (counts[v] for v in Verdict)

    out = {
        "totals": {"eval": len(pairs), "accepted": tp + fp, "rejected": fn + tn},
        "counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "rates": {
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        },
    }
    if chain is not None:
        out["by_criterion"] = cascade_by_criterion(pairs, chain)
    return out


def summarize_by_epoch(
    pairs: list[EvaluatedPair], epochs: dict[int, Epoch], chain: list[str] | None = None
) -> dict[str, dict]:
    """Verdicts split by drift pair-class; each bucket summarized like the top block.

    The ``prev_post`` bucket isolates the loop-closure mechanism (original vs ghost).
    ``chain`` propagates the by_criterion cascade into each epoch bucket.
    """
    buckets: dict[str, list[EvaluatedPair]] = defaultdict(list)
    for p in pairs:
        try:
            cls = pair_class(epochs[p.decision.i1], epochs[p.decision.i2])
        except KeyError as missing:
            raise ValueError(
                f"obj_id {missing} in a fusion decision has no drift epoch "
                f"(absent from the checkpoint's instances)."
            ) from None
        buckets[cls].append(p)
    return {cls: summarize(grp, chain=chain) for cls, grp in sorted(buckets.items())}
