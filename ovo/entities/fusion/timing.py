"""Fusion timing: where the time of each fusion goes, step by step.

Each step has up to two costs: preparing it, once per fusion for every instance, and judging the
pairs that reach it, summed over those pairs. With `overlap: voxel`, for example, preparing
computes every overlap and a pair is a lookup. Steps, in the order they run:

    extract     the point cloud of each instance          prepare only (OVO._fuse_instances)
    <gates>     in config order                           prepare + pairs
    cos_sim     reading the descriptors; their similarity prepare + pairs
    overlap     building the overlap method; measuring    prepare + pairs
    apply       merging the accepted pairs                prepare only (OVO._fuse_instances)

How many pairs reached each step is not counted here: the decisions log already says it (a
rejected pair stops at its `reason`). Log, one row per fusion and step, written for every fusion
(also one that judged no pair: preparing was paid anyway):

    {scene}/fusion/fusion_timings.csv
    frame_id,step,prepare_s,pairs_s
    120,extract,1.52,
    120,centroid,0.002,0.010

With the map on the GPU each prepared step waits for it (torch.cuda.synchronize), or the time
would only be that of queueing the work. Per-pair steps already wait on every pair.
"""

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch

COLUMNS = ("frame_id", "step", "prepare_s", "pairs_s")


@dataclass
class StepTime:
    prepare_s: float = 0.0
    pairs_s: Optional[float] = None  # None: the step has no per-pair work


@contextmanager
def preparing(step: StepTime, sync: bool) -> Iterator[None]:
    """Adds the time of the block to `step.prepare_s`; with `sync`, waits for the GPU first."""
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if sync:
        torch.cuda.synchronize()
    step.prepare_s += time.perf_counter() - t0


class TimingLog:
    """Appends the step times of every fusion to one CSV. A file left by an earlier run is
    removed on creation."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.unlink(missing_ok=True)
        self._started = False

    def write(self, frame_id: int, steps: dict[str, StepTime]) -> None:
        if not self._started:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a" if self._started else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if not self._started:
                writer.writeheader()
                self._started = True
            writer.writerows({"frame_id": frame_id, "step": name, "prepare_s": t.prepare_s, "pairs_s": t.pairs_s}
                             for name, t in steps.items())
