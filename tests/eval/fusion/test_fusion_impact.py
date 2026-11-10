"""ovo.eval.fusion.impact: masks after the merges, agnostic AP and per-GT stats before vs after.

Scene: a chair split in two (1 and 2, 2 too small to count for the AP), a table (3) and a
prediction on void (4). The fusion merges 1 and 2.
"""

import numpy as np
import pytest

from ovo.eval.benchmarks import REPLICA
from ovo.eval.fusion import impact

CHAIR = REPLICA.class_ids[REPLICA.class_names.index("chair")] * 1000 + 1
TABLE = REPLICA.class_ids[REPLICA.class_names.index("table")] * 1000 + 1
V = 500


def _span(lo, hi):
    m = np.zeros(V, dtype=bool)
    m[lo:hi] = True
    return m


GT = np.zeros(V, dtype=np.int64)
GT[0:200], GT[200:400] = CHAIR, TABLE
MASKS = np.array([_span(0, 120), _span(120, 200), _span(200, 400), _span(400, 500)])
IDS = np.array([1, 2, 3, 4])
GROUPS = {1: [1, 2]}


def _sides():
    pre = impact.side(MASKS, IDS, GT, REPLICA.class_ids)
    post = impact.side(*impact.post_masks(MASKS, IDS, GROUPS), GT, REPLICA.class_ids)
    return pre, post


def test_post_masks_or_each_group_under_its_survivor():
    masks, ids = impact.post_masks(MASKS, IDS, {3: [3, 1]})
    assert ids.tolist() == [2, 3, 4]
    assert np.array_equal(masks[1], _span(0, 120) | _span(200, 400))
    assert np.array_equal(masks[0], MASKS[1])


def test_agnostic_ap_before_and_after():
    pre, post = _sides()
    ap = impact.agnostic_ap(GT, pre, post, REPLICA.class_ids)
    # before: the chair half has IoU 0.6, matched up to 0.55; above, it is a false positive and
    # the chair is missed: AP = recall * (precision + 1) / 2 = 0.5 * 1.5 / 2
    assert ap["pre"]["AP_50"] == 1.0
    assert ap["pre"]["AP"] == pytest.approx((2 * 1.0 + 7 * 0.375) / 9)
    assert (ap["pre"]["matched_50"], ap["pre"]["spurious_50"], ap["pre"]["missed_50"]) == (2, 0, 0)
    assert ap["post"]["AP"] == 1.0
    assert ap["delta"]["AP"] == pytest.approx(1.0 - ap["pre"]["AP"])
    assert ap["delta"]["matched_50"] == 0


def test_instance_stats():
    pre, post = _sides()
    rows = {r["gt_id"]: r for r in impact.instance_stats(GT, pre, post, REPLICA.class_ids, REPLICA.class_names)}
    assert list(rows) == [CHAIR, TABLE]
    chair, table = rows[CHAIR], rows[TABLE]
    assert chair["class_name"] == "chair"
    assert (chair["iou_pre"], chair["iou_post"]) == pytest.approx((0.6, 1.0))
    assert (chair["coverage_pre"], chair["coverage_post"]) == pytest.approx((0.6, 1.0))
    assert (chair["change"], chair["match_50"], chair["pred_pre"], chair["pred_post"]) == ("improved", "KEPT", 1, 1)
    assert chair["fragments_pre"] == "2"  # claims nothing, lies on the chair
    assert (table["change"], table["match_50"], table["pred_pre"], table["pred_post"], table["fragments_pre"]) == (
        "unchanged", "KEPT", 3, 3, "")


def test_an_over_merge_loses_the_table():
    pre = impact.side(MASKS, IDS, GT, REPLICA.class_ids)
    post = impact.side(*impact.post_masks(MASKS, IDS, {1: [1, 3]}), GT, REPLICA.class_ids)
    rows = {r["gt_id"]: r for r in impact.instance_stats(GT, pre, post, REPLICA.class_ids, REPLICA.class_names)}
    # 1+3 has IoU 120/400 with the chair and 200/320 with the table: it keeps the table
    assert (rows[TABLE]["match_50"], rows[TABLE]["pred_post"], rows[TABLE]["change"]) == ("KEPT", 1, "worsened")
    assert (rows[CHAIR]["match_50"], rows[CHAIR]["pred_post"]) == ("LOST", "")
