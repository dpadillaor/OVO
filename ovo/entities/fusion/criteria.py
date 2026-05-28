from abc import ABC, abstractmethod
from typing import Tuple, Optional

from ...utils.instance_utils import compute_pcd_overlap, compute_centroid_distance, compute_aabb_distance
from ...utils.cooccurrence_graph import CooccurrenceGraph

import logging
import torch

logger = logging.getLogger("ovo.fusion")


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
