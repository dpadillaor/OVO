"""
Fusion Strategy using Chain of Responsibility for instance comparison.

Each criterion in the chain runs sequentially. A criterion returns:
  None  → passed, continue to next criterion
  True  → ACCEPT (stop chain)
  False → REJECT (stop chain)

Configure the chain via `fusion_criteria` in config, or let the factory
pick the default chain for the chosen `fusion_method`.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
from ..utils.instance_utils import compute_pcd_overlap, compute_centroid_distance, compute_aabb_distance
from ..utils.cooccurrence_graph import CooccurrenceGraph

import logging
import time
import torch

logger = logging.getLogger("ovo.fusion")


# ---------------------------------------------------------------------------
# Criterion ABC
# ---------------------------------------------------------------------------

class Criterion(ABC):
    name: str = "criterion"

    @abstractmethod
    def check(
        self,
        instance1,
        instance2,
        points1,
        centroid1,
        points2,
        centroid2,
        ctx: dict,
    ) -> Tuple[Optional[bool], dict]:
        """
        Returns (verdict, decision_dict).
          verdict None  → pass
          verdict True  → accept
          verdict False → reject
        ctx is a shared dict for passing values between criteria (e.g. cos_sim → overlap).
        """
        pass


# ---------------------------------------------------------------------------
# Concrete criteria
# ---------------------------------------------------------------------------

class CooccurrenceCriterion(Criterion):
    name = "cooccurrence"

    def __init__(self, cooccurrence_graph: CooccurrenceGraph, threshold: int):
        self._graph = cooccurrence_graph
        self.threshold = threshold

    def check(self, i1, i2, p1, c1, p2, c2, ctx):
        shared_kfs = self._graph.weight(i1.id, i2.id)
        ctx["shared_kfs"] = shared_kfs
        if shared_kfs > self.threshold:
            logger.debug("REJECTED i1=%s i2=%s | cooccurrence: shared_kfs=%d > th=%d", i1.id, i2.id, shared_kfs, self.threshold)
            return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "cooccurrence", "shared_kfs": shared_kfs}
        return None, {}


class CentroidDistanceCriterion(Criterion):
    name = "centroid"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def check(self, i1, i2, p1, c1, p2, c2, ctx):
        dist = compute_centroid_distance(c1, c2)
        ctx["centroid_dist"] = float(dist)
        if dist > self.threshold:
            logger.debug("REJECTED i1=%s i2=%s | centroid_dist=%.3f > th=%.3f", i1.id, i2.id, dist, self.threshold)
            return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "centroid", "centroid_dist": float(dist)}
        return None, {}


class AabbDistanceCriterion(Criterion):
    name = "aabb"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def check(self, i1, i2, p1, c1, p2, c2, ctx):
        dist = compute_aabb_distance(p1, p2)
        ctx["aabb_dist"] = float(dist)
        if dist > self.threshold:
            logger.debug("REJECTED i1=%s i2=%s | aabb_dist=%.3f > th=%.3f", i1.id, i2.id, dist, self.threshold)
            return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "aabb", "aabb_dist": float(dist)}
        return None, {}


class CosSimilarityCriterion(Criterion):
    name = "cos_sim"

    def __init__(self, feature_attr: str, threshold: float):
        self.feature_attr = feature_attr
        self.threshold = threshold

    def check(self, i1, i2, p1, c1, p2, c2, ctx):
        f1 = getattr(i1, self.feature_attr)[0]
        f2 = getattr(i2, self.feature_attr)[0]
        cos_sim = torch.nn.functional.cosine_similarity(f1, f2, dim=0).item()
        ctx["cos_sim"] = cos_sim
        if cos_sim < self.threshold:
            logger.debug("REJECTED i1=%s i2=%s | cos_sim=%.3f < th=%.3f (centroid_dist=%.3f)", i1.id, i2.id, cos_sim, self.threshold, ctx.get("centroid_dist", float("nan")))
            return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "cos_sim", "centroid_dist": ctx.get("centroid_dist"), "cos_sim": cos_sim}
        return None, {}


class PointOverlapCriterion(Criterion):
    name = "overlap"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def check(self, i1, i2, p1, c1, p2, c2, ctx):
        p_dist = compute_pcd_overlap(p1, p2, self.threshold)
        ctx["p_dist"] = float(p_dist)
        cos_sim = ctx.get("cos_sim")
        centroid_dist = ctx.get("centroid_dist")
        shared_kfs = ctx.get("shared_kfs")

        accepted = p_dist > 0.5 or (cos_sim is not None and cos_sim > 0.9 and p_dist > 0.2)
        if accepted:
            logger.debug("ACCEPTED i1=%s i2=%s | cos_sim=%s centroid_dist=%s p_dist=%.3f", i1.id, i2.id, cos_sim, centroid_dist, p_dist)
            return True, {"result": "ACCEPTED", "i1": i1.id, "i2": i2.id, "centroid_dist": centroid_dist, "cos_sim": cos_sim, "p_dist": float(p_dist), "shared_kfs": shared_kfs}
        logger.debug("REJECTED i1=%s i2=%s | overlap: p_dist=%.3f (cos_sim=%s centroid_dist=%s)", i1.id, i2.id, p_dist, cos_sim, centroid_dist)
        return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "overlap", "centroid_dist": centroid_dist, "cos_sim": cos_sim, "p_dist": float(p_dist), "shared_kfs": shared_kfs}


# ---------------------------------------------------------------------------
# FusionStrategy: runs the criterion chain
# ---------------------------------------------------------------------------

class FusionStrategy:
    def __init__(self, criteria: List[Criterion]):
        self.criteria = criteria
        self._decisions: list = []
        self._criterion_times: Dict[str, float] = {}

    def same_instance(
        self,
        instance1,
        instance2,
        points_centroid1: Tuple[torch.Tensor, torch.Tensor],
        points_centroid2: Tuple[torch.Tensor, torch.Tensor],
    ) -> bool:
        points1, centroid1 = points_centroid1
        points2, centroid2 = points_centroid2
        ctx: dict = {}

        for criterion in self.criteria:
            t0 = time.time()
            verdict, decision = criterion.check(instance1, instance2, points1, centroid1, points2, centroid2, ctx)
            self._criterion_times[criterion.name] = self._criterion_times.get(criterion.name, 0.0) + (time.time() - t0)
            if verdict is not None:
                if decision:
                    self._decisions.append(decision)
                return verdict

        return False

    def pop_decisions(self) -> list:
        decisions, self._decisions = self._decisions, []
        return decisions

    def pop_timings(self) -> Dict[str, float]:
        timings, self._criterion_times = self._criterion_times, {}
        return timings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_CHAINS: Dict[str, List[str]] = {
    "clip":  ["cooccurrence", "centroid", "cos_sim", "overlap"],
    "dino":  ["cooccurrence", "centroid", "cos_sim", "overlap"],
    "pe":    ["cooccurrence", "centroid", "cos_sim", "overlap"],
    "sam3":  ["cooccurrence", "centroid", "cos_sim", "overlap"],
}

_FEATURE_ATTRS: Dict[str, str] = {
    "clip": "clip_feature",
    "dino": "dino_feature",
    "pe":   "pe_feature",
    "sam3": "sam3_feature",
}


def _build_criterion(name: str, config: Dict[str, Any], feature_attr: Optional[str], cooccurrence_graph: CooccurrenceGraph) -> Criterion:
    if name == "cooccurrence":
        return CooccurrenceCriterion(cooccurrence_graph, config.get("cooccurrence_veto_threshold", 5))
    if name == "centroid":
        return CentroidDistanceCriterion(config.get("th_centroid", 1.5))
    if name == "aabb":
        return AabbDistanceCriterion(config.get("th_aabb", 0.3))
    if name == "cos_sim":
        if feature_attr is None:
            raise ValueError("cos_sim criterion requires a semantic fusion_method (not geometric)")
        return CosSimilarityCriterion(feature_attr, config.get("th_cossim", 0.81))
    if name == "overlap":
        return PointOverlapCriterion(config.get("th_points", 0.1))
    raise ValueError(f"Unknown criterion: {name!r}. Valid: cooccurrence, centroid, aabb, cos_sim, overlap")


def create_fusion_strategy(config: Dict[str, Any], cooccurrence_graph: CooccurrenceGraph) -> FusionStrategy:
    """
    Build a FusionStrategy from config.

    fusion_method selects the default criterion chain and feature attribute.
    Override the chain by setting fusion_criteria (list of criterion names) in config.
    """
    fusion_method = config.get("fusion_method", "clip").lower()
    if fusion_method not in _DEFAULT_CHAINS:
        raise ValueError(f"Unknown fusion_method: {fusion_method!r}. Valid: {list(_DEFAULT_CHAINS)}")

    chain = config.get("fusion_criteria", _DEFAULT_CHAINS[fusion_method])
    feature_attr = _FEATURE_ATTRS.get(fusion_method)

    criteria = [_build_criterion(name, config, feature_attr, cooccurrence_graph) for name in chain]
    return FusionStrategy(criteria)
