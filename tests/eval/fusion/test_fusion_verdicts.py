"""ovo.eval.fusion.verdicts: dominant GT, verdict per pair, counts and cascade."""

import numpy as np
import pytest

from ovo.eval.fusion import decisions, verdicts
from ovo.eval.fusion.verdicts import Scored

CLASS_IDS = (3, 7)
#                 v0    v1    v2    v3    v4    v5    v6
GT = np.array([3001, 3001, 3002, 7001, 5001, 5001, 0])  # 5xxx: not a benchmark class (void)


def test_dominant_gt_ignores_void_and_breaks_ties_by_id():
    masks = np.array([
        [1, 1, 1, 0, 0, 0, 0],  # 3001 x2, 3002 x1
        [0, 0, 0, 1, 1, 1, 1],  # 7001 x1 beats two void classes and 0
        [0, 0, 1, 1, 0, 0, 0],  # tie 3002 / 7001 -> 3002
        [0, 0, 0, 0, 1, 1, 1],  # only void
    ], dtype=bool)
    assert verdicts.dominant_gt(masks, np.array([10, 20, 30, 40]), GT, CLASS_IDS) == {
        10: 3001, 20: 7001, 30: 3002, 40: None}


def _row(first, second, accepted, reason="overlap", mode=None):
    text = f"frame_id,first,second,accepted,reason,mode\n1,{first},{second},{accepted},{reason},{mode or ''}\n"
    return decisions.parse(text).rows[0]


def test_score_needs_both_masks_and_void_is_never_the_same():
    dominant = {1: 3001, 2: 3001, 3: None, 4: None}
    assert verdicts.score(_row(1, 2, True, mode="strong"), dominant).verdict == "TP"
    assert verdicts.score(_row(3, 4, True, mode="weak"), dominant).verdict == "FP"
    assert verdicts.score(_row(1, 2, False, "cos_sim"), dominant).verdict == "FN"
    assert verdicts.score(_row(1, 3, False, "cos_sim"), dominant).verdict == "TN"
    assert verdicts.score(_row(1, 9, False, "cos_sim"), dominant) is None


SCORED = [
    Scored(False, "centroid", None, True),   # FN at the gate
    Scored(False, "centroid", None, False),  # TN at the gate
    Scored(False, "cos_sim", None, False),
    Scored(False, "overlap", None, True),
    Scored(True, "overlap", "strong", True),
    Scored(True, "overlap", "weak", False),
    Scored(True, "overlap", "strong", True),
]


def test_counts_and_rates():
    assert verdicts.counts(SCORED) == {"TP": 2, "FP": 1, "FN": 2, "TN": 2,
                                       "precision": 2 / 3, "recall": 0.5, "f1": 4 / 7}
    assert verdicts.counts([])["precision"] is None


def test_cascade_follows_the_pipeline():
    steps = verdicts.cascade(SCORED, ["centroid", "cos_sim", "overlap"])
    picked = [(s["step"], s["in"], s["pass"], s["reject"], s["TP"], s["FP"], s["FN"], s["TN"]) for s in steps]
    assert picked == [
        ("centroid", 7, 5, 2, 3, 2, 1, 1),
        ("cos_sim", 5, 4, 1, 3, 1, 0, 1),
        ("overlap", 4, 3, 1, 2, 1, 1, 0),
    ]
    assert steps[2]["modes"] == {"strong": {"pass": 2, "TP": 2, "FP": 0}, "weak": {"pass": 1, "TP": 0, "FP": 1}}
    assert "modes" not in steps[0]
    assert steps[1]["recall"] == 1.0 and steps[1]["precision"] == 0.75


def test_cascade_rejects_a_reason_that_is_not_a_step():
    with pytest.raises(ValueError, match="centroid"):
        verdicts.cascade(SCORED, ["cos_sim", "overlap"])
