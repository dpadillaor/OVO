"""Decide which instances are the same object and in which order to merge them.

For each candidate pair, cheapest first, stopping as soon as the pair is rejected:
    policy (still worth judging?) -> gates -> semantics -> overlap -> accept rule
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .apply import POLICIES
from .config import FusionCfg
from .gates import build_gate
from .overlap import build_overlap
from .rule import accept, similar_enough


@dataclass
class Decision:
    """What was measured about a pair and what was decided. One row of the decisions log.
    Values that were not needed to decide stay None."""
    first: int                          # survives if the pair is fused
    second: int
    accepted: bool = False
    reason: str = "overlap"             # who decided: a gate name, "cos_sim", or "overlap"
    mode: Optional[str] = None          # accept branch: strong | weak | both
    cos_sim: Optional[float] = None
    overlap: Optional[float] = None
    gate_values: dict[str, float] = field(default_factory=dict)


@dataclass
class FusionReport:
    decisions: list[Decision]           # pairs actually judged, in double-loop order
    merges: list[tuple[int, int]]       # (survivor, absorbed), in the order to apply them
    seconds: float


class FusionPipeline:
    def __init__(self, cfg: FusionCfg, feature_attr: str = "clip_feature"):
        self.cfg = cfg
        self.feature_attr = feature_attr
        self.gates = [build_gate(g.name, g.params) for g in cfg.gates]
        self.overlap = build_overlap(cfg.overlap)
        if cfg.apply not in POLICIES:
            raise ValueError(f"instance_fusion.apply must be one of {sorted(POLICIES)}, got {cfg.apply!r}")
        self.policy = POLICIES[cfg.apply]

    def run(self, instances: list, pcds: dict[int, torch.Tensor]) -> FusionReport:
        t0 = time.perf_counter()
        order = [inst.id for inst in instances]
        features = {inst.id: getattr(inst, self.feature_attr)[0] for inst in instances}
        self.overlap.prepare(order, pcds)
        for gate in self.gates:
            gate.prepare(instances, pcds)
        policy = self.policy(order)

        decisions = []
        for a, b in self.overlap.candidates():
            if not policy.should_judge(a, b):
                continue  # the outcome can no longer change: measure nothing
            d = Decision(a, b)
            decisions.append(d)
            if self._judge(d, features[a], features[b]):
                policy.on_accept(a, b)

        return FusionReport(decisions, policy.merges(), time.perf_counter() - t0)

    def _judge(self, d: Decision, feat_a: torch.Tensor, feat_b: torch.Tensor) -> bool:
        for gate in self.gates:
            ok, d.gate_values[gate.name] = gate.check(d.first, d.second)
            if not ok:
                d.reason = gate.name
                return False

        d.cos_sim = torch.nn.functional.cosine_similarity(feat_a, feat_b, dim=0).item()
        if not similar_enough(d.cos_sim, self.cfg.accept):
            d.reason = "cos_sim"
            return False

        d.overlap = self.overlap.measure(d.first, d.second)
        d.mode = accept(d.overlap, d.cos_sim, self.cfg.accept)
        d.accepted = d.mode is not None
        return d.accepted
