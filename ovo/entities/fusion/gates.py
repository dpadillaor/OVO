"""Gates: optional cheap vetoes, checked before anything else is measured for a pair.
A gate can only say "not the same". None are active by default.

A gate has a `name` (its key in the YAML list `gates`), a `Params` dataclass with its defaults,
`prepare` (once per pass, e.g. per-instance precomputation) and `check`, which returns whether
the pair passes plus the value it measured (logged with the decision).

    gates:
      - centroid: {max_dist: 1.5}     # distance between centroids
      - aabb: {max_dist: 0.3}         # gap between bounding boxes (fairer to elongated objects)
"""

from dataclasses import dataclass

import torch


class CentroidGate:
    """Veto pairs whose centroids are farther apart than max_dist (m). From the original OVO."""

    name = "centroid"

    @dataclass(frozen=True)
    class Params:
        max_dist: float = 1.5

    def __init__(self, params: "CentroidGate.Params"):
        self.params = params

    def prepare(self, instances, pcds: dict[int, torch.Tensor]) -> None:
        self._centroids = {i: p.mean(0) for i, p in pcds.items()}

    def check(self, first: int, second: int) -> tuple[bool, float]:
        dist = float((self._centroids[first] - self._centroids[second]).norm())
        return dist <= self.params.max_dist, dist


class AabbGate:
    """Veto pairs whose axis-aligned bounding boxes are farther apart than max_dist (m).

    The distance is the gap between the boxes (0 if they touch or intersect). It is the
    counterpart of `centroid` that is fair to elongated objects (a long table and a chair at its
    end have distant centroids but touching boxes).

    Meant for `point` overlap. With `voxel` overlap it is pointless: every candidate pair shares
    a voxel, so its gap is below the voxel diagonal and the gate never vetoes it.
    """

    name = "aabb"

    @dataclass(frozen=True)
    class Params:
        max_dist: float = 0.3

    def __init__(self, params: "AabbGate.Params"):
        self.params = params

    def prepare(self, instances, pcds: dict[int, torch.Tensor]) -> None:
        self._boxes = {i: (p.amin(0), p.amax(0)) for i, p in pcds.items() if len(p)}

    def check(self, first: int, second: int) -> tuple[bool, float]:
        if first not in self._boxes or second not in self._boxes:
            return False, float("inf")  # an empty instance has no box: it cannot be the same object
        (min1, max1), (min2, max2) = self._boxes[first], self._boxes[second]
        gap = torch.clamp(torch.maximum(min1 - max2, min2 - max1), min=0)
        dist = float(gap.norm())
        return dist <= self.params.max_dist, dist


GATES = {CentroidGate.name: CentroidGate, AabbGate.name: AabbGate}


def build_gate(name: str, params: dict):
    if name not in GATES:
        raise ValueError(f"instance_fusion.gates: unknown gate {name!r}; valid: {sorted(GATES)}")
    cls = GATES[name]
    try:
        return cls(cls.Params(**params))
    except TypeError as e:
        raise ValueError(f"instance_fusion.gates.{name}: {e}") from None
