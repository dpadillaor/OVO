"""Drift epochs of a SimulatedSLAM run: which pose error was an instance born with?

Every event in `{scene}/simulation.json` (a jump or a correction) changes the pose error, so it
starts a new epoch; within one, the error is the same (continuous drift aside). Epoch k holds
the frames after k events:

    jumps at 30 and 70, correction at 50          e0: 0-29  e1: 30-49  e2: 50-69  e3: 70-

An instance belongs to the epoch of its `created_at_frame` (a fused instance, to the earliest of
its parts). An event at a frame already affects that frame: the jump moves its pose, the
correction snaps it back to GT before it is segmented. A pair of instances is labelled by its
two epochs, lowest first: "e0_e1" is an original against a copy born after the jump.

Events after the fusion change nothing: every instance in its pre-fusion checkpoint was born
before it, so they never count.
"""

import json
from pathlib import Path
from typing import Optional


def boundaries(path: Path) -> Optional[list[int]]:
    """Frames at which an epoch starts, sorted; None when there is no simulation.json (not a
    SimulatedSLAM run: there are no epochs to tell)."""
    if not path.is_file():
        return None
    events = json.loads(path.read_text())
    return sorted({int(e["frame"]) for e in events["jumps"] + events["corrections"]})


def epoch(created_at_frame: int, bounds: list[int]) -> int:
    return sum(1 for b in bounds if b <= created_at_frame)


def pair(a: int, b: int) -> str:
    lo, hi = sorted((a, b))
    return f"e{lo}_e{hi}"
