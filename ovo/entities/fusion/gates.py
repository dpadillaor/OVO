"""Gates: optional cheap vetoes, checked before anything else is measured for a pair.
A gate can only say "not the same". None are active by default.

A gate has a `name` (its key in the YAML list `gates`), a `Params` dataclass with its defaults,
`prepare` (once per pass, e.g. per-instance precomputation) and `check`, which returns whether
the pair passes plus the value it measured (logged with the decision).

    gates:
      - centroid: {max_dist: 1.5}
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


GATES = {CentroidGate.name: CentroidGate}


def build_gate(name: str, params: dict):
    if name not in GATES:
        raise ValueError(f"instance_fusion.gates: unknown gate {name!r}; valid: {sorted(GATES)}")
    cls = GATES[name]
    try:
        return cls(cls.Params(**params))
    except TypeError as e:
        raise ValueError(f"instance_fusion.gates.{name}: {e}") from None
