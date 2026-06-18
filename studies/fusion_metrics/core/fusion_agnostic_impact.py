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

# 9 strict thresholds (ap_mean calculated from this average) + 0.25 threshold reported apart,
# I've mirrored what ins_eval_utils was doing (all_ap excludes 0.25).
DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = (
    0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
)
_LENIENT_IOU = 0.25


@dataclass
class ThresholdResult:
    """AP and confusion counts at one IoU threshold (positive = a GT was matched)."""

    iou: float
    ap: float
    matched: int   # GT instances hit and won by a prediction
    spurious: int  # Predictions that didn't win this GT isntance
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

    def at(self, iou: float) -> ThresholdResult:
        """The per-threshold result at ``iou`` (raises if not computed)."""
        for r in self.per_threshold:
            if abs(r.iou - iou) < 1e-6:
                return r
        raise ValueError(f"IoU threshold {iou} not in per_threshold results.")

    def _ap_at(self, iou: float) -> float:
        return self.at(iou).ap

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


@dataclass
class InstanceStat:
    """Per-GT-instance segmentation quality, class-agnostic, pre vs post fusion."""

    gt_id: int
    iou_pre: float
    iou_post: float
    acc_pre: float
    acc_post: float
    matched_obj_id: int | None   # pred obj_id matching this GT pre-fusion (IoU >= thr)
    spurious_obj_ids: list[int]  # over-split fragments dominated by this GT (pre-fusion)

    @property
    def n_spurious(self) -> int:
        return len(self.spurious_obj_ids)

    @property
    def status(self) -> str:
        """How fusion moved this object's IoU: improved / worsened / unchanged."""
        if self.iou_post > self.iou_pre + 1e-6:
            return "improved"
        if self.iou_post < self.iou_pre - 1e-6:
            return "worsened"
        return "unchanged"


def _gt_instance_masks(gt_ids: np.ndarray) -> np.ndarray:
    """Boolean (V, G) one column per ground-truth instance; void ids (<=0) dropped."""
    ids = np.unique(gt_ids[gt_ids > 0])
    return ids[None, :] == gt_ids[:, None]


def _intersection(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """(M, G) overlap counts. float32 matmul: BLAS-accelerated, exact below 2**24."""
    return pred_masks.astype(np.float32).T @ gt_masks.astype(np.float32)


def _iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """(M, G) IoU between every predicted mask and every GT instance."""
    inter = _intersection(pred_masks, gt_masks)          # (M, G)
    union = pred_masks.sum(0)[:, None] + gt_masks.sum(0)[None, :] - inter
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


def _greedy_gt_to_pred(iou: np.ndarray, threshold: float) -> dict[int, int]:
    """Greedy 1:1 assignment by descending IoU >= ``threshold``: gt_index -> pred col."""
    won: dict[int, int] = {}
    if not iou.size:
        return won
    ps, gs = np.where(iou >= threshold)
    used_pred, used_gt = set(), set()
    for k in np.argsort(iou[ps, gs])[::-1]:
        p, g = int(ps[k]), int(gs[k])
        if p in used_pred or g in used_gt:
            continue
        used_pred.add(p)
        used_gt.add(g)
        won[g] = p
    return won


def instances_for_gt(
    pred_masks: np.ndarray, gt_ids: np.ndarray, gt_id: int, iou_threshold: float = 0.5
) -> tuple[int | None, list[int]]:
    """For one GT instance: (matched pred column | None, spurious pred columns).

    Spurious = predictions that won no GT at all but whose dominant (most-covered)
    GT is ``gt_id`` -- i.e. the losing fragments of this object's over-split.
    """
    ids = np.unique(gt_ids[gt_ids > 0])
    where = np.flatnonzero(ids == gt_id)
    if where.size == 0:
        raise ValueError(f"GT id {gt_id} not present. Available: {ids.tolist()}")
    gi = int(where[0])

    gt_masks = _gt_instance_masks(gt_ids)
    iou = _iou_matrix(pred_masks, gt_masks)
    won = _greedy_gt_to_pred(iou, iou_threshold)
    matched_col = won.get(gi)
    matched_preds = set(won.values())

    inter = _intersection(pred_masks, gt_masks)  # (M, G) counts
    dominant = inter.argmax(axis=1)
    spurious = [
        p for p in range(pred_masks.shape[1])
        if p not in matched_preds and inter[p].max() > 0 and int(dominant[p]) == gi
    ]
    return matched_col, spurious


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


def _gt_overlap(
    masks: np.ndarray, gt_masks: np.ndarray, gt_sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per GT: (best IoU, acc of best-IoU pred, best pred col, inter (M,G), iou (M,G))."""
    n_gt = gt_masks.shape[1]
    if masks.shape[1] == 0:
        z = np.zeros(n_gt)
        empty = np.zeros((0, n_gt))
        return z, z, np.full(n_gt, -1), empty, empty
    inter = _intersection(masks, gt_masks)                       # (M, G)
    psize = masks.sum(0).astype(np.float64)
    union = psize[:, None] + gt_sizes[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    g = np.arange(n_gt)
    best = iou.argmax(0)
    return iou[best, g], inter[best, g] / np.maximum(gt_sizes, 1), best, inter, iou


def per_instance_stats(
    masks_pre: np.ndarray,
    col_obj_ids: np.ndarray,
    decisions: list[FusionDecision],
    gt_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[InstanceStat]:
    """Per-GT-instance IoU/acc pre vs post fusion + matched/spurious predictions."""
    masks_post = build_post_masks(masks_pre, col_obj_ids, merged_groups(decisions))
    gt_unique = np.unique(gt_ids[gt_ids > 0])
    gt_masks = _gt_instance_masks(gt_ids)
    gt_sizes = gt_masks.sum(0).astype(np.float64)

    iou_pre, acc_pre, best_pre, inter_pre, iou_pre_mat = _gt_overlap(masks_pre, gt_masks, gt_sizes)
    iou_post, acc_post, *_ = _gt_overlap(masks_post, gt_masks, gt_sizes)

    # Greedy 1:1 GT->pred match (same as AP). matched_obj_id is who won this GT;
    # spurious = predictions that won no GT whose dominant GT is this one.
    won = _greedy_gt_to_pred(iou_pre_mat, iou_threshold)
    matched_preds = set(won.values())
    spurious_by_gt: dict[int, list[int]] = defaultdict(list)
    for p in range(masks_pre.shape[1]):
        if p not in matched_preds and inter_pre[p].max() > 0:
            spurious_by_gt[int(inter_pre[p].argmax())].append(p)

    stats = []
    for gi, gid in enumerate(gt_unique):
        matched = int(col_obj_ids[won[gi]]) if gi in won else None
        spurious = [int(col_obj_ids[p]) for p in spurious_by_gt.get(gi, [])]
        stats.append(InstanceStat(
            gt_id=int(gid),
            iou_pre=round(float(iou_pre[gi]), 4), iou_post=round(float(iou_post[gi]), 4),
            acc_pre=round(float(acc_pre[gi]), 4), acc_post=round(float(acc_post[gi]), 4),
            matched_obj_id=matched, spurious_obj_ids=spurious,
        ))
    return stats


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
    pre50, post50 = pre.at(0.5), post.at(0.5)
    return {
        "delta_ap_mean": round(post.ap_mean - pre.ap_mean, 4),
        "delta_ap50": round(post.ap50 - pre.ap50, 4),
        "delta_ap25": round(post.ap25 - pre.ap25, 4),
        "delta_matched50": post50.matched - pre50.matched,
        "delta_spurious50": post50.spurious - pre50.spurious,
        "delta_missed50": post50.missed - pre50.missed,
        "pre": pre.to_dict(),
        "post": post.to_dict(),
    }
