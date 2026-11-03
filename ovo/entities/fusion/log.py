"""Fusion decisions log: one CSV row per judged pair, with what was measured and decided.

    frame_id,first,second,accepted,reason,mode,cos_sim,overlap,aabb,centroid
    120,4,9,True,overlap,strong,0.93,0.71,,0.12
    120,4,15,False,cos_sim,,0.62,,,0.30
    120,7,9,False,centroid,,,,,2.41

Columns are the fields of ``Decision`` plus one per known gate, always present: a cell is empty
when the value was not needed to decide (an earlier step rejected the pair) or the gate is off.
Pairs the apply policy skipped are not judged and have no row.
"""

import csv
from pathlib import Path

from .gates import GATES
from .pipeline import Decision, FusionReport

GATE_COLUMNS = tuple(sorted(GATES))
COLUMNS = ("frame_id", "first", "second", "accepted", "reason", "mode", "cos_sim", "overlap") + GATE_COLUMNS


def decision_row(frame_id: int, d: Decision) -> dict:
    row = {
        "frame_id": frame_id,
        "first": d.first,
        "second": d.second,
        "accepted": d.accepted,
        "reason": d.reason,
        "mode": d.mode,
        "cos_sim": d.cos_sim,
        "overlap": d.overlap,
    }
    row.update({gate: d.gate_values.get(gate) for gate in GATE_COLUMNS})
    unknown = set(d.gate_values) - set(GATE_COLUMNS)
    if unknown:
        raise ValueError(f"gate values without a column: {sorted(unknown)}")
    return row


class FusionLog:
    """Appends the decisions of every fusion run to one CSV.

    A file left by an earlier run is removed on creation, and the header is written with the
    first decisions: no file means no pair was judged.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.unlink(missing_ok=True)
        self._started = False

    def write(self, frame_id: int, report: FusionReport) -> None:
        if not report.decisions:
            return
        if not self._started:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a" if self._started else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if not self._started:
                writer.writeheader()
                self._started = True
            writer.writerows(decision_row(frame_id, d) for d in report.decisions)
