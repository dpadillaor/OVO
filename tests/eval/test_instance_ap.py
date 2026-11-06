"""Instance AP (ovo.eval.instance_ap) on hand-built scenes with a known answer."""

import math

import numpy as np
import pytest

from ovo.eval import instance_ap
from ovo.eval.instance_ap import MIN_REGION_SIZE, Scan

CLASS_IDS = (3, 7)
CLASS_NAMES = ("a", "b")
A, B = 0, 1  # class indices of the predictions
V = 1000
NOT_A_CLASS = 5  # GT class outside the benchmark: its vertices are void


def gt(*instances, n=V):
    """GT ids from (instance_id, start, stop) triples; everything else is 0 (void)."""
    ids = np.zeros(n, dtype=np.int64)
    for inst, lo, hi in instances:
        ids[lo:hi] = inst
    return ids


def mask(lo, hi, n=V):
    m = np.zeros(n, dtype=bool)
    m[lo:hi] = True
    return m


def scan(gt_ids, preds):
    """``preds``: list of (mask, class index, score)."""
    n = len(gt_ids)
    return Scan(
        gt_ids=gt_ids,
        masks=np.array([p[0] for p in preds], dtype=bool).reshape(-1, n),
        classes=np.array([p[1] for p in preds], dtype=np.int64),
        scores=np.array([p[2] for p in preds], dtype=float),
    )


def class_ap(scans, splits=None):
    return instance_ap.evaluate(scans, CLASS_IDS, CLASS_NAMES, splits)


def agnostic_ap(scans):
    return instance_ap.evaluate_agnostic(scans, CLASS_IDS)


TWO_OBJECTS = gt((3000, 0, 200), (7000, 200, 400))


def test_thresholds_are_the_scannet_ones():
    assert MIN_REGION_SIZE == 100
    np.testing.assert_allclose(instance_ap.OVERLAPS, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.25])


def test_perfect_masks_score_one():
    s = {"s": scan(TWO_OBJECTS, [(mask(0, 200), A, 0.9), (mask(200, 400), B, 0.8)])}
    assert class_ap(s) == {"AP": 1.0, "AP_50": 1.0, "AP_25": 1.0}
    assert agnostic_ap(s) == {"AP": 1.0, "AP_50": 1.0, "AP_25": 1.0}


def test_no_predictions_score_zero():
    s = {"s": scan(TWO_OBJECTS, [])}
    assert class_ap(s) == {"AP": 0.0, "AP_50": 0.0, "AP_25": 0.0}
    assert agnostic_ap(s) == {"AP": 0.0, "AP_50": 0.0, "AP_25": 0.0}


def test_unclassified_predictions_count_as_missing():
    s = {"s": scan(TWO_OBJECTS, [(mask(0, 200), -1, 0.9), (mask(200, 400), -1, 0.8)])}
    assert class_ap(s)["AP"] == 0.0
    assert agnostic_ap(s)["AP"] == 1.0


def test_class_without_gt_is_nan():
    only_a = {"s": scan(gt((3000, 0, 200)), [(mask(0, 200), A, 0.9)])}
    metrics = class_ap(only_a, {"with_a": ("a",), "only_b": ("b",)})
    assert metrics["AP"] == 1.0  # b (no GT) is left out of the mean, not counted as 0
    assert metrics["AP_with_a"] == 1.0
    assert math.isnan(metrics["AP_only_b"])


def test_no_gt_at_all_is_nan():
    metrics = class_ap({"s": scan(gt(), [(mask(0, 200), A, 0.9)])})
    assert all(math.isnan(v) for v in metrics.values())


@pytest.mark.parametrize("size, counted", [(MIN_REGION_SIZE - 1, False), (MIN_REGION_SIZE, True)])
def test_small_gt_instances_are_ignored(size, counted):
    metrics = class_ap({"s": scan(gt((3000, 0, size)), [])})
    if counted:
        assert metrics["AP"] == 0.0  # a missed instance
    else:
        assert math.isnan(metrics["AP"])  # as if there were no GT at all


@pytest.mark.parametrize("size, expected", [(MIN_REGION_SIZE - 1, 0.0), (MIN_REGION_SIZE, 1.0)])
def test_small_predictions_are_ignored(size, expected):
    # inside a 200-vertex GT, IoU is size / 200 > 0.25: only the region size decides
    s = {"s": scan(gt((3000, 0, 200)), [(mask(0, size), A, 0.9)])}
    assert class_ap(s)["AP_25"] == expected


@pytest.mark.parametrize("void_region", [
    gt(),  # unlabelled vertices
    gt((NOT_A_CLASS * 1000, 500, 800)),  # a class outside the benchmark
])
def test_prediction_on_void_is_not_a_false_positive(void_region):
    gt_ids = void_region + gt((3000, 0, 200))
    # the void prediction has the higher score: were it a false positive, precision would drop
    s = {"s": scan(gt_ids, [(mask(0, 200), A, 0.5), (mask(500, 800), A, 0.9)])}
    assert class_ap(s) == {"AP": 1.0, "AP_50": 1.0, "AP_25": 1.0}


def test_prediction_on_another_benchmark_class_is_a_false_positive():
    gt_ids = gt((3000, 0, 200), (7000, 500, 800))
    s = {"s": scan(gt_ids, [(mask(0, 200), A, 0.5), (mask(500, 800), A, 0.9)])}
    assert class_ap(s, {"a": ("a",)})["AP_a"] < 1.0


# Contested GT: g1 = [0, 200) and g2 = [200, 400), both class a.
# WIDE = [100, 300) overlaps both with IoU 1/3; TIGHT = [0, 140) overlaps g1 with IoU 0.7.
# At IoU 0.25, g1 must go to TIGHT (higher IoU), which leaves WIDE free to match g2.
# Taking g1's candidates in load order instead lets WIDE claim g1 when it is loaded first,
# so g2 is missed and AP_25 drops to 0.5.
CONTESTED_GT = gt((3000, 0, 200), (3001, 200, 400))
WIDE = (mask(100, 300), A, 0.9)
TIGHT = (mask(0, 140), A, 0.8)


@pytest.mark.parametrize("order", [[WIDE, TIGHT], [TIGHT, WIDE]], ids=["wide_first", "tight_first"])
def test_contested_gt_goes_to_the_highest_iou_prediction(order):
    metrics = class_ap({"s": scan(CONTESTED_GT, order)})
    assert metrics["AP_25"] == 1.0
    # from IoU 0.5 up only TIGHT matches anything (g1, until 0.7): g2 is missed in both orders
    assert metrics["AP_50"] == 0.125
    assert metrics["AP"] == 0.05555555555555555


def test_agnostic_ignores_classes():
    s = {"s": scan(TWO_OBJECTS, [(mask(0, 200), B, 0.9), (mask(200, 400), A, 0.8)])}  # swapped
    assert class_ap(s)["AP"] == 0.0
    assert agnostic_ap(s) == {"AP": 1.0, "AP_50": 1.0, "AP_25": 1.0}


@pytest.mark.parametrize("tp_score, fp_score", [(0.9, 0.1), (0.1, 0.9)])
def test_agnostic_ignores_scores(tp_score, fp_score):
    gt_ids = gt((3000, 0, 200), (7000, 500, 800))
    preds = [(mask(0, 200), A, tp_score), (mask(500, 800), A, fp_score)]
    tp_first = scan(gt_ids, [(mask(0, 200), A, 0.9), (mask(500, 800), A, 0.1)])
    assert agnostic_ap({"s": scan(gt_ids, preds)}) == agnostic_ap({"s": tp_first})


def test_class_aware_ap_depends_on_scores():
    gt_ids = gt((3000, 0, 200), (7000, 500, 800))
    tp_high = scan(gt_ids, [(mask(0, 200), A, 0.9), (mask(500, 800), A, 0.1)])
    tp_low = scan(gt_ids, [(mask(0, 200), A, 0.1), (mask(500, 800), A, 0.9)])
    split = {"a": ("a",)}
    assert class_ap({"s": tp_high}, split)["AP_a"] > class_ap({"s": tp_low}, split)["AP_a"]


def test_splits_add_one_key_each():
    s = {"s": scan(TWO_OBJECTS, [(mask(0, 200), A, 0.9)])}
    assert set(class_ap(s)) == {"AP", "AP_50", "AP_25"}
    metrics = class_ap(s, {"head": ("a",), "tail": ("b",)})
    assert set(metrics) == {"AP", "AP_50", "AP_25", "AP_head", "AP_tail"}
    assert metrics["AP_head"] == 1.0
    assert metrics["AP_tail"] == 0.0
    assert metrics["AP"] == 0.5


def test_scans_are_pooled():
    found = scan(gt((3000, 0, 200)), [(mask(0, 200), A, 0.9)])
    missed = scan(gt((3000, 0, 200)), [])
    pooled = class_ap({"x": found, "y": missed})
    assert 0.0 < pooled["AP"] < 1.0


def test_mask_length_mismatch_raises():
    bad = Scan(gt_ids=TWO_OBJECTS, masks=np.ones((1, V + 1), bool), classes=np.array([A]), scores=np.array([0.9]))
    with pytest.raises(ValueError, match=f"{V + 1} vertices, GT has {V}"):
        class_ap({"s": bad})
    with pytest.raises(ValueError, match="vertices"):
        agnostic_ap({"s": bad})


# ---------- match_agnostic ----------

MATCH_GT = gt((3000, 0, 200), (3001, 200, 400), (7000, 400, 600), (3002, 800, 850), (5000, 850, 1000))
MATCH_PREDS = [
    (mask(0, 200), A, 1.0),  # claims 3000
    (mask(0, 150), A, 1.0),  # IoU 0.75 with 3000, already claimed: false positive
    (mask(150, 300), A, 1.0),  # IoU 0.4 with 3001, at most: false positive
    (mask(600, 800), A, 1.0),  # all void: ignored
]


def test_match_agnostic_claims_counts_and_gt_set():
    m = instance_ap.match_agnostic(scan(MATCH_GT, MATCH_PREDS), CLASS_IDS, 0.5)
    # 3002 is below MIN_REGION_SIZE and 5000 is not a benchmark class: neither counts
    assert m.pred_of_gt == {3000: 0, 3001: None, 7000: None}
    assert (m.matched, m.missed, m.spurious) == (1, 2, 2)


def test_match_agnostic_agrees_with_ap_50():
    s = scan(MATCH_GT, MATCH_PREDS)
    m = instance_ap.match_agnostic(s, CLASS_IDS, 0.5)
    # equal scores: one point on the PR curve, AP = recall * (precision + 1) / 2
    precision, recall = m.matched / (m.matched + m.spurious), m.matched / (m.matched + m.missed)
    assert agnostic_ap({"s": s})["AP_50"] == pytest.approx(recall * (precision + 1) / 2)
    assert recall * (precision + 1) / 2 == pytest.approx(2 / 9)
