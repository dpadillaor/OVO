"""Instance-segmentation AP over mesh vertices.

Adapted from the ScanNet benchmark evaluator (MIT licence, Copyright 2017 Angela Dai, Angel X.
Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Niessner)
https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
through its derivatives in Mask3D (MIT licence)
https://github.com/JonasSchult/Mask3D/blob/main/benchmark/evaluate_semantic_instance.py
and OpenMask3D (MIT licence, head/common/tail splits)
https://github.com/OpenMask3D/openmask3d/blob/main/openmask3d/evaluation/eval_semantic_instance.py
and the dict-based variant added to https://github.com/tberriel/OVO in commit b2c3448
(``ins_eval_example/ins_eval_utils.py``).

Change from the original: when several predictions overlap the same GT instance, the one with
the highest IoU claims it (the original took them in load order).

GT: one id per mesh vertex, ``class_id * 1000 + k`` (0 = void). Only classes in the benchmark
count; vertices of any other class are void (predictions overlapping them are not penalised).
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

# IoU thresholds: AP averages 0.50:0.05:0.90; AP_50 and AP_25 use one threshold each.
OVERLAPS = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# GT instances and predictions smaller than this (in vertices) are ignored.
MIN_REGION_SIZE = 100


@dataclass(frozen=True)
class Scan:
    gt_ids: np.ndarray  # (V,) int
    masks: np.ndarray  # (N, V) bool
    classes: np.ndarray  # (N,) int, index into class_names, -1 = unclassified
    scores: np.ndarray  # (N,) float


def evaluate(
    scans: Mapping[str, Scan],
    class_ids: tuple[int, ...],
    class_names: tuple[str, ...],
    splits: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, float]:
    """Class-aware AP. ``splits`` (e.g. head/common/tail) adds ``AP_<split>`` per group."""
    id_to_label = dict(zip(class_ids, class_names))
    pred_label = lambda c: class_names[c] if 0 <= c < len(class_names) else None
    aps = _ap_table(scans, class_ids, class_names, id_to_label, pred_label, lambda scores: scores)
    metrics = _summary(aps)
    per_class = _per_class_ap(aps)
    for split, names in (splits or {}).items():
        values = [per_class[i] for i, name in enumerate(class_names) if name in names]
        metrics[f"AP_{split}"] = _nanmean(values)
    return metrics


def evaluate_agnostic(scans: Mapping[str, Scan], class_ids: tuple[int, ...]) -> dict[str, float]:
    """Class-agnostic AP: one class for every GT instance and prediction, and all scores equal,
    so only the masks are judged."""
    id_to_label = {cid: "object" for cid in class_ids}
    aps = _ap_table(scans, class_ids, ("object",), id_to_label, lambda c: "object", np.ones_like)
    return _summary(aps)


_IS_25 = np.isclose(OVERLAPS, 0.25)
_IS_50 = np.isclose(OVERLAPS, 0.5)


def _summary(aps: np.ndarray) -> dict[str, float]:
    return {
        "AP": _nanmean(aps[:, ~_IS_25]),
        "AP_50": _nanmean(aps[:, _IS_50]),
        "AP_25": _nanmean(aps[:, _IS_25]),
    }


def _per_class_ap(aps: np.ndarray) -> np.ndarray:
    # np.average (not nanmean): a class is nan only when it has no GT, at every threshold alike.
    # Row by row, not axis=1: keeps the summation order, so results match to the last bit.
    return np.array([np.average(row[~_IS_25]) for row in aps])


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float("nan") if np.all(np.isnan(values)) else float(np.nanmean(values))


# ---------- matching ----------


def _ap_table(scans, class_ids, labels, id_to_label, pred_label, scores_of) -> np.ndarray:
    """AP per (label, overlap threshold); nan where a label has no GT."""
    matches = [
        _match_scan(si, scan, class_ids, labels, id_to_label, pred_label, scores_of(scan.scores))
        for si, scan in enumerate(scans.values())
    ]
    aps = np.zeros((len(labels), len(OVERLAPS)))
    for oi, overlap_th in enumerate(OVERLAPS):
        visited = set()  # predictions already claimed by a GT at this threshold
        for li, label in enumerate(labels):
            aps[li, oi] = _label_ap(matches, label, overlap_th, visited)
    return aps


def _match_scan(scan_index, scan, class_ids, labels, id_to_label, pred_label, scores):
    """GT instances and predictions of one scan, grouped by label, with their intersections."""
    gt_ids = scan.gt_ids
    gt_by_label = {label: [] for label in labels}
    for inst_id in np.unique(gt_ids):
        if inst_id == 0 or inst_id // 1000 not in id_to_label:
            continue
        gt_by_label[id_to_label[inst_id // 1000]].append(
            {"instance_id": int(inst_id), "vert_count": int((gt_ids == inst_id).sum()), "matched_pred": []}
        )

    void = ~np.isin(gt_ids // 1000, class_ids)
    pred_by_label = {label: [] for label in labels}
    for i, mask in enumerate(scan.masks):
        label = pred_label(int(scan.classes[i]))
        if label is None:
            continue
        if len(mask) != len(gt_ids):
            raise ValueError(f"prediction mask has {len(mask)} vertices, GT has {len(gt_ids)}")
        mask = mask != 0
        count = int(np.count_nonzero(mask))
        if count < MIN_REGION_SIZE:
            continue
        pred = {
            "key": (scan_index, i),
            "vert_count": count,
            "confidence": scores[i],
            "void_intersection": int(np.count_nonzero(void & mask)),
            "matched_gt": [],
        }
        for gt in gt_by_label[label]:
            inter = int(np.count_nonzero((gt_ids == gt["instance_id"]) & mask))
            if inter > 0:
                pred["matched_gt"].append({**gt, "intersection": inter})
                gt["matched_pred"].append({**pred, "intersection": inter})
        pred_by_label[label].append(pred)
    return gt_by_label, pred_by_label


def _iou(gt, pred, inter) -> float:
    return float(inter) / (gt["vert_count"] + pred["vert_count"] - inter)


def _label_ap(matches, label, overlap_th, visited) -> float:
    y_true, y_score = [], []
    hard_false_negatives = 0
    has_gt = has_pred = False

    for gt_by_label, pred_by_label in matches:
        preds = pred_by_label[label]
        gts = [g for g in gt_by_label[label] if g["instance_id"] >= 1000 and g["vert_count"] >= MIN_REGION_SIZE]
        has_gt |= bool(gts)
        has_pred |= bool(preds)

        cur_true = np.ones(len(gts))
        cur_score = np.full(len(gts), -np.inf)
        cur_match = np.zeros(len(gts), dtype=bool)
        for gi, gt in enumerate(gts):
            found = False
            for pred in sorted(gt["matched_pred"], key=lambda p: _iou(gt, p, p["intersection"]), reverse=True):
                if pred["key"] in visited:
                    continue
                if _iou(gt, pred, pred["intersection"]) > overlap_th:
                    conf = pred["confidence"]
                    if cur_match[gi]:
                        # a second prediction for a matched GT: the lower score is a false positive
                        cur_score[gi], low = max(cur_score[gi], conf), min(cur_score[gi], conf)
                        cur_true = np.append(cur_true, 0)
                        cur_score = np.append(cur_score, low)
                        cur_match = np.append(cur_match, True)
                    else:
                        found = True
                        cur_match[gi] = True
                        cur_score[gi] = conf
                        visited.add(pred["key"])
            if not found:
                hard_false_negatives += 1
        cur_true, cur_score = cur_true[cur_match], cur_score[cur_match]

        # unmatched predictions are false positives unless they mostly cover ignored vertices
        for pred in preds:
            if any(_iou(gt, pred, gt["intersection"]) > overlap_th for gt in pred["matched_gt"]):
                continue
            ignored = pred["void_intersection"]
            for gt in pred["matched_gt"]:
                if gt["instance_id"] < 1000:
                    ignored += gt["intersection"]
                if gt["vert_count"] < MIN_REGION_SIZE:
                    ignored += gt["intersection"]
            if float(ignored) / pred["vert_count"] <= overlap_th:
                cur_true = np.append(cur_true, 0)
                cur_score = np.append(cur_score, pred["confidence"])

        y_true.append(cur_true)
        y_score.append(cur_score)

    if not has_gt:
        return float("nan")
    if not has_pred:
        return 0.0
    return _average_precision(np.concatenate(y_true), np.concatenate(y_score), hard_false_negatives)


def _average_precision(y_true, y_score, hard_false_negatives) -> float:
    order = np.argsort(y_score)
    y_score, y_true = y_score[order], y_true[order]
    cumsum = np.cumsum(y_true)
    _, unique_indices = np.unique(y_score, return_index=True)
    n = len(unique_indices) + 1
    num_examples = len(y_score)
    # https://github.com/ScanNet/ScanNet/pull/26: every prediction may be unmatched and ignored
    num_true = cumsum[-1] if len(cumsum) > 0 else 0
    precision, recall = np.zeros(n), np.zeros(n)
    cumsum = np.append(cumsum, 0)
    for k, idx in enumerate(unique_indices):
        before = cumsum[idx - 1]
        tp = num_true - before
        fp = num_examples - idx - tp
        fn = before + hard_false_negatives
        precision[k] = float(tp) / (tp + fp)
        recall[k] = float(tp) / (tp + fn)
    # the first point of the curve is artificial
    precision[-1], recall[-1] = 1.0, 0.0
    padded = np.append(np.append(recall[0], recall), 0.0)
    step_widths = np.convolve(padded, [-0.5, 0, 0.5], "valid")
    return float(np.dot(precision, step_widths))
