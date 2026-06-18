"""Class-agnostic AP before vs after fusion, to quantify each run's merge impact.

Both sides come from the SAME projected pre-fusion masks: ``post`` is built by
OR-ing the masks of every connected component of ACCEPTED merges, so the delta
isolates the effect of the recorded decisions (same vertices, same projection).
AP is class-agnostic: only mask geometry matters, scores are uniform.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from core.loaders import FusionDecision

# 9 strict thresholds (averaged into ap_mean) + a lenient 0.25 reported apart,
# mirroring ins_eval_utils (all_ap excludes 0.25).
DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = (
    0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
)
_LENIENT_IOU = 0.25


@dataclass
class ThresholdResult:
    """AP and confusion counts at one IoU threshold (positive = a GT was matched)."""

    iou: float
    ap: float
    matched: int   # GT instances hit
    spurious: int  # predictions that matched no GT
    missed: int    # GT instances no prediction reached

    def to_dict(self) -> dict:
        return {
            "iou": self.iou, "ap": self.ap, "matched": self.matched,
            "spurious": self.spurious, "missed": self.missed,
        }


@dataclass
class AgnosticAP:
    """Per-threshold results; ap_mean/ap50/ap25 are derived, never stored."""

    per_threshold: list[ThresholdResult]

    def _ap_at(self, iou: float) -> float:
        for r in self.per_threshold:
            if abs(r.iou - iou) < 1e-6:
                return r.ap
        return float("nan")

    @property
    def ap_mean(self) -> float:
        """Mean AP over the strict thresholds (excludes the lenient 0.25)."""
        vals = [r.ap for r in self.per_threshold if abs(r.iou - _LENIENT_IOU) > 1e-6]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    @property
    def ap50(self) -> float:
        return self._ap_at(0.5)

    @property
    def ap25(self) -> float:
        return self._ap_at(_LENIENT_IOU)

    def to_dict(self) -> dict:
        return {
            "ap_mean": self.ap_mean,
            "ap50": self.ap50,
            "ap25": self.ap25,
            "per_threshold": [r.to_dict() for r in self.per_threshold],
        }


def _gt_instance_masks(gt_ids: np.ndarray) -> np.ndarray:
    """Boolean (V, G) one column per ground-truth instance; void ids (<=0) dropped."""
    ids = np.unique(gt_ids[gt_ids > 0])
    return ids[None, :] == gt_ids[:, None]


def _iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """(M, G) IoU between every predicted mask and every GT instance."""
    pred = pred_masks.astype(np.int64)
    gt = gt_masks.astype(np.int64)
    inter = pred.T @ gt                                  # (M, G)
    union = pred.sum(0)[:, None] + gt.sum(0)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def _match_at(iou: np.ndarray, threshold: float) -> tuple[np.ndarray, int, int, int]:
    """Greedy 1:1 match by descending IoU above ``threshold`` (uniform confidence).

    Returns (y_true per prediction, matched, spurious, missed).
    """
    n_pred, n_gt = iou.shape
    y_true = np.zeros(n_pred, dtype=np.int64)
    if n_pred and n_gt:
        ps, gs = np.where(iou >= threshold)
        order = np.argsort(iou[ps, gs])[::-1]
        used_pred, used_gt = set(), set()
        for k in order:
            p, g = int(ps[k]), int(gs[k])
            if p in used_pred or g in used_gt:
                continue
            used_pred.add(p)
            used_gt.add(g)
            y_true[p] = 1
    matched = int(y_true.sum())
    return y_true, matched, n_pred - matched, n_gt - matched


def _average_precision(y_true: np.ndarray, missed: int) -> float:
    """Area under the PR curve (ScanNet integral) for uniform-confidence predictions.

    With one confidence level the curve is a single operating point; this is the
    same integration ins_eval_utils uses, so numbers stay comparable.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    tp = float(y_true.sum())
    fp = n - tp
    precision = np.array([tp / (tp + fp) if tp + fp else 0.0, 1.0])
    recall = np.array([tp / (tp + missed) if tp + missed else 0.0, 0.0])
    recall_conv = np.concatenate(([recall[0]], recall, [0.0]))
    step_widths = np.convolve(recall_conv, [-0.5, 0, 0.5], "valid")
    return float(np.dot(precision, step_widths))


def compute_agnostic_ap(
    pred_masks: np.ndarray,
    gt_ids: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
) -> AgnosticAP:
    """Class-agnostic AP of ``pred_masks`` (V, M) against per-vertex ``gt_ids`` (V,)."""
    gt_masks = _gt_instance_masks(gt_ids)
    iou = _iou_matrix(pred_masks, gt_masks)
    results = []
    for t in thresholds:
        y_true, matched, spurious, missed = _match_at(iou, t)
        ap = _average_precision(y_true, missed)
        results.append(ThresholdResult(
            iou=round(t, 2), ap=round(ap, 4),
            matched=matched, spurious=spurious, missed=missed,
        ))
    return AgnosticAP(per_threshold=results)


def merged_groups(decisions: list[FusionDecision]) -> list[set[int]]:
    """Connected components of ACCEPTED merges, as sets of obj_ids (union-find)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d in decisions:
        if d.merged:
            parent[find(d.i1)] = find(d.i2)

    groups: dict[int, set[int]] = defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)
    return list(groups.values())


def build_post_masks(
    masks_pre: np.ndarray, col_obj_ids: np.ndarray, groups: list[set[int]]
) -> np.ndarray:
    """Post-fusion masks: OR the columns of each merged group, pass the rest through.

    obj_ids in a group but absent from ``col_obj_ids`` (lost in projection) are
    simply skipped; a group with no surviving column yields no output column.
    """
    obj_to_group = {oid: gi for gi, grp in enumerate(groups) for oid in grp}
    group_cols: dict[int, list[int]] = defaultdict(list)
    singletons: list[int] = []
    for col, oid in enumerate(col_obj_ids):
        gi = obj_to_group.get(int(oid))
        (singletons.append(col) if gi is None else group_cols[gi].append(col))

    columns = [masks_pre[:, cols].any(axis=1) for cols in group_cols.values()]
    columns += [masks_pre[:, col] for col in singletons]
    if not columns:
        return np.zeros((masks_pre.shape[0], 0), dtype=bool)
    return np.stack(columns, axis=1)


def fusion_impact(
    masks_pre: np.ndarray,
    col_obj_ids: np.ndarray,
    decisions: list[FusionDecision],
    gt_ids: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
) -> dict:
    """Agnostic-AP block for the summary JSON: pre, post and their deltas."""
    pre = compute_agnostic_ap(masks_pre, gt_ids, thresholds)
    masks_post = build_post_masks(masks_pre, col_obj_ids, merged_groups(decisions))
    post = compute_agnostic_ap(masks_post, gt_ids, thresholds)
    return {
        "delta_ap_mean": round(post.ap_mean - pre.ap_mean, 4),
        "delta_ap50": round(post.ap50 - pre.ap50, 4),
        "delta_ap25": round(post.ap25 - pre.ap25, 4),
        "pre": pre.to_dict(),
        "post": post.to_dict(),
    }
