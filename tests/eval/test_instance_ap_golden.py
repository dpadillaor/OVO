"""Instance AP against frozen values of the evaluator it was ported from.

The expected numbers were computed once with the dict-based ``ins_eval_utils.py`` evaluator
(with the highest-IoU matching) on the synthetic scenes below, and are compared for exact float
equality. The scenes come from a small linear congruential generator, so they do
not depend on numpy's random streams.
"""

import math

import numpy as np
import pytest

from ovo.eval import instance_ap
from ovo.eval.benchmarks import REPLICA


ID_OF = dict(zip(REPLICA.class_names, REPLICA.class_ids))


class Lcg:
    def __init__(self, seed: int):
        self.state = seed

    def next(self) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) % 2**64
        return self.state >> 33

    def below(self, n: int) -> int:
        return self.next() % n

    def uniform(self) -> float:
        return self.next() / 2**31


def synthetic_scan(rng: Lcg) -> instance_ap.Scan:
    """GT instances of a few Replica classes (plus non-benchmark ones), noisy copies of them as
    predictions with sometimes-wrong classes, and a few spurious masks."""
    n = 1500 + rng.below(1500)
    gt_ids = np.zeros(n, dtype=np.int64)
    pool = [ID_OF[name] for name in ("chair", "table", "bed", "bin", "book", "clock")]
    pool += [1, 2]  # not benchmark classes: void for the evaluator
    counts = {}
    start = rng.below(60)
    while start < n - 50:
        size = 40 + rng.below(320)
        cls = pool[rng.below(len(pool))]
        gt_ids[start:start + size] = cls * 1000 + counts.get(cls, 0)
        counts[cls] = counts.get(cls, 0) + 1
        start += size + rng.below(120)

    masks, classes = [], []
    for inst in np.unique(gt_ids[gt_ids >= 1000]):
        for _ in range(rng.below(3)):
            m = gt_ids == inst
            idx = np.flatnonzero(m)
            m[idx[: int(len(idx) * 0.5 * rng.uniform())]] = False  # drop a prefix
            lo = rng.below(n)
            m[lo:lo + rng.below(300)] = True  # bleed into a random stretch
            masks.append(m)
            cid = int(inst) // 1000
            right = REPLICA.class_ids.index(cid) if cid in REPLICA.class_ids else 0
            classes.append(right if rng.uniform() < 0.7 else rng.below(len(REPLICA.class_ids) + 1) - 1)
    for _ in range(rng.below(4)):
        m = np.zeros(n, dtype=bool)
        lo = rng.below(n)
        m[lo:lo + 80 + rng.below(400)] = True
        masks.append(m)
        classes.append(rng.below(len(REPLICA.class_ids)))
    scores = [round(rng.uniform(), 1 + 3 * rng.below(2)) for _ in masks]  # ties at 1 decimal
    return instance_ap.Scan(
        gt_ids=gt_ids,
        masks=np.array(masks, dtype=bool).reshape(-1, n),
        classes=np.array(classes, dtype=np.int64),
        scores=np.array(scores, dtype=float),
    )


def scenario(seed: int) -> dict:
    rng = Lcg(seed)
    return {f"scene{i}": synthetic_scan(rng) for i in range(1 + rng.below(3))}


SPLITS = {"head": REPLICA.head, "common": REPLICA.common, "tail": REPLICA.tail}

GOLDEN = {
    1: {  # 3 scans, 35 predictions
        "class.AP": 0.12034465020576134,
        "class.AP_50": 0.3028935185185185,
        "class.AP_25": 0.40648148148148144,
        "class.AP_head": 0.16219135802469134,
        "class.AP_common": 0.06481481481481481,
        "class.AP_tail": 0.13402777777777777,
        "agnostic.AP": 0.20908189556015644,
        "agnostic.AP_50": 0.5499999999999999,
        "agnostic.AP_25": 0.5920000000000001,
    },
    2: {  # 2 scans, 26 predictions
        "class.AP": 0.06222222222222223,
        "class.AP_50": 0.17333333333333334,
        "class.AP_25": 0.36904761904761907,
        "class.AP_head": math.nan,
        "class.AP_common": 0.010000000000000002,
        "class.AP_tail": 0.16666666666666666,
        "agnostic.AP": 0.1575416991180665,
        "agnostic.AP_50": 0.26960784313725494,
        "agnostic.AP_25": 0.5978260869565217,
    },
    3: {  # 3 scans, 43 predictions
        "class.AP": 0.04888117283950617,
        "class.AP_50": 0.1440972222222222,
        "class.AP_25": 0.7092427248677248,
        "class.AP_head": 0.13831018518518517,
        "class.AP_common": 0.0,
        "class.AP_tail": 0.008333333333333333,
        "agnostic.AP": 0.08951995533378623,
        "agnostic.AP_50": 0.29333333333333333,
        "agnostic.AP_25": 0.695216049382716,
    },
    4: {  # 3 scans, 25 predictions
        "class.AP": 0.08603395061728396,
        "class.AP_50": 0.23263888888888892,
        "class.AP_25": 0.40277777777777773,
        "class.AP_head": 0.06018518518518518,
        "class.AP_common": 0.16666666666666666,
        "class.AP_tail": 0.03125,
        "agnostic.AP": 0.14414427209834846,
        "agnostic.AP_50": 0.38235294117647056,
        "agnostic.AP_25": 0.5744949494949495,
    },
    5: {  # 2 scans, 19 predictions
        "class.AP": 0.02222222222222222,
        "class.AP_50": 0.13333333333333333,
        "class.AP_25": 0.6416666666666666,
        "class.AP_head": 0.0,
        "class.AP_common": 0.05555555555555555,
        "class.AP_tail": 0.027777777777777776,
        "agnostic.AP": 0.13564263668430332,
        "agnostic.AP_50": 0.35714285714285715,
        "agnostic.AP_25": 0.7384259259259259,
    },
}


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_matches_the_original_evaluator(seed):
    scans = scenario(seed)
    got = {
        **{f"class.{k}": v for k, v in instance_ap.evaluate(scans, REPLICA.class_ids, REPLICA.class_names, SPLITS).items()},
        **{f"agnostic.{k}": v for k, v in instance_ap.evaluate_agnostic(scans, REPLICA.class_ids).items()},
    }
    expected = GOLDEN[seed]
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert (math.isnan(got[key]) and math.isnan(value)) or got[key] == value, key


def test_golden_values_are_not_trivial():
    values = [v for metrics in GOLDEN.values() for v in metrics.values() if not math.isnan(v)]
    assert len(GOLDEN) == 5
    assert any(0.0 < v < 1.0 for v in values)
