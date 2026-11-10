"""Were the fusion's decisions right? Each judged pair against the GT, positive = merge.

The dominant GT of an instance is the GT instance covering most of its mask. Every GT instance
counts, also those of classes outside the AP benchmark (wall, floor...): the question is whether two
instances are the same physical object, and the GT tells two walls apart. Only id 0 (unlabelled) is
void and never dominates. Ties go to the smallest id. Two instances are the same object when they
share a dominant GT.

    accepted, same object       TP      accepted, different     FP
    rejected, same object       FN      rejected, different     TN

The cascade scores each step of the pipeline (gates in config order, then cos_sim, then overlap)
on the pairs that reached it, positive = let the pair through:

    cascade(scored, ["centroid", "cos_sim", "overlap"])
    [{"step": "centroid", "in": 12, "pass": 9, "reject": 3, "TP": 4, ...}, ...]
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Scored:
    """One judged pair whose two instances both have a mask."""
    accepted: bool
    reason: str
    mode: Optional[str]
    same_object: bool

    @property
    def verdict(self) -> str:
        if self.accepted:
            return "TP" if self.same_object else "FP"
        return "FN" if self.same_object else "TN"


def dominant_gt(masks: np.ndarray, ids: np.ndarray, gt_ids: np.ndarray) -> dict[int, Optional[int]]:
    """{instance id: dominant GT id, or None when its mask covers only void}."""
    valid = gt_ids > 0
    out = {}
    for ins_id, mask in zip(ids, masks):
        under = gt_ids[mask & valid]
        if under.size == 0:
            out[int(ins_id)] = None
            continue
        values, counts = np.unique(under, return_counts=True)
        out[int(ins_id)] = int(values[np.argmax(counts)])
    return out


def score(row, dominant: dict[int, Optional[int]]) -> Optional[Scored]:
    """`row`: a decisions.Row. None when either instance has no mask (it cannot be judged)."""
    if row.first not in dominant or row.second not in dominant:
        return None
    a, b = dominant[row.first], dominant[row.second]
    return Scored(row.accepted, row.reason, row.mode, a is not None and a == b)


def counts(scored: list[Scored]) -> dict:
    c = Counter(s.verdict for s in scored)
    return {"TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"], **_rates(c["TP"], c["FP"], c["FN"])}


def cascade(scored: list[Scored], steps: list[str]) -> list[dict]:
    """One entry per step. A rejected pair leaves at the step named by its reason; an accepted
    pair passes them all. The last step also splits what it passed by accept mode."""
    index = {name: i for i, name in enumerate(steps)}
    unknown = {s.reason for s in scored if not s.accepted} - set(index)
    if unknown:
        raise ValueError(f"rejected by {sorted(unknown)}, which is not a step of {steps}")
    left_at = [len(steps) if s.accepted else index[s.reason] for s in scored]

    out = []
    for i, step in enumerate(steps):
        passed = [s for s, at in zip(scored, left_at) if at > i]
        rejected = [s for s, at in zip(scored, left_at) if at == i]
        tp = sum(s.same_object for s in passed)
        fn = sum(s.same_object for s in rejected)
        entry = {
            "step": step, "in": len(passed) + len(rejected), "pass": len(passed), "reject": len(rejected),
            "TP": tp, "FP": len(passed) - tp, "FN": fn, "TN": len(rejected) - fn,
            **_rates(tp, len(passed) - tp, fn),
        }
        if i == len(steps) - 1:
            entry["modes"] = _modes(passed)
        out.append(entry)
    return out


def _modes(passed: list[Scored]) -> dict[str, dict[str, int]]:
    modes: dict[str, dict[str, int]] = {}
    for s in passed:
        m = modes.setdefault(s.mode, {"pass": 0, "TP": 0, "FP": 0})
        m["pass"] += 1
        m["TP" if s.same_object else "FP"] += 1
    return dict(sorted(modes.items()))


def _rates(tp: int, fp: int, fn: int) -> dict[str, Optional[float]]:
    """precision, recall and f1; None when undefined."""
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }
