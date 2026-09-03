from abc import ABC, abstractmethod
from itertools import combinations
from typing import Tuple, Optional

from ...utils.instance_utils import compute_pcd_overlap, compute_pcd_old_overlap, compute_centroid_distance
from ...utils.cooccurrence_graph import CooccurrenceGraph
from .voxel import VoxelIndex

import logging
import torch

logger = logging.getLogger("ovo.fusion")


class Criterion(ABC):
    name: str = "criterion"

    def prepare(self, objects_list, obj_pcds) -> None:
        """Per-pass precompute hook, called once before the pair loop. No-op by default.

        Criteria that reuse per-instance state across pairs build it here from `obj_pcds`
        ({id: points}) instead of recomputing it on every pair: centroid caches the means,
        aabb the bounding boxes, voxel the occupied cells.
        """
        pass

    @abstractmethod
    def check(
        self,
        instance1,
        instance2,
        points1,
        points2,
        ctx: dict,
    ) -> Tuple[Optional[bool], dict]:
        """
        Returns (verdict, decision_dict).
          verdict None  → pass
          verdict True  → accept
          verdict False → reject
        ctx is a shared dict for passing values between criteria (e.g. cos_sim → overlap).
        Per-instance geometry is precomputed in `prepare`; `points1`/`points2` are the raw
        clouds for criteria that still need them per pair (overlap_old).
        """
        pass


class CooccurrenceCriterion(Criterion):
    name = "cooccurrence"

    def __init__(self, cooccurrence_graph: CooccurrenceGraph, threshold: int):
        self._graph = cooccurrence_graph
        self.threshold = threshold

    def check(self, i1, i2, p1, p2, ctx):
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
        self._centroid = {}

    def prepare(self, objects_list, obj_pcds) -> None:
        self._centroid = {inst.id: obj_pcds[inst.id].mean(axis=0) for inst in objects_list}

    def check(self, i1, i2, p1, p2, ctx):
        dist = compute_centroid_distance(self._centroid[i1.id], self._centroid[i2.id])
        ctx["centroid_dist"] = float(dist)
        if dist > self.threshold:
            logger.debug("REJECTED i1=%s i2=%s | centroid_dist=%.3f > th=%.3f", i1.id, i2.id, dist, self.threshold)
            return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "centroid", "centroid_dist": float(dist)}
        return None, {}


class AabbDistanceCriterion(Criterion):
    name = "aabb"

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._bbox = {}

    def prepare(self, objects_list, obj_pcds) -> None:
        self._bbox = {inst.id: (obj_pcds[inst.id].amin(dim=0), obj_pcds[inst.id].amax(dim=0)) for inst in objects_list}

    def check(self, i1, i2, p1, p2, ctx):
        (min1, max1), (min2, max2) = self._bbox[i1.id], self._bbox[i2.id]
        gap = torch.clamp(torch.maximum(min1 - max2, min2 - max1), min=0)
        dist = gap.norm().item()
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

    def check(self, i1, i2, p1, p2, ctx):
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

    def check(self, i1, i2, p1, p2, ctx):
        p_dist = compute_pcd_overlap(p1, p2, self.threshold)
        ctx["p_dist"] = float(p_dist)
        cos_sim = ctx.get("cos_sim")
        centroid_dist = ctx.get("centroid_dist")
        shared_kfs = ctx.get("shared_kfs")

        geom = p_dist > 0.5  # rama A: solape geométrico fuerte
        sem = cos_sim is not None and cos_sim > 0.9 and p_dist > 0.2  # rama B: semántica + solape débil
        if geom or sem:
            mode = "A" if geom and not sem else "B" if sem and not geom else "AB"
            logger.debug("ACCEPTED i1=%s i2=%s | mode=%s cos_sim=%s centroid_dist=%s p_dist=%.3f", i1.id, i2.id, mode, cos_sim, centroid_dist, p_dist)
            return True, {"result": "ACCEPTED", "accept_mode": mode, "i1": i1.id, "i2": i2.id, "centroid_dist": centroid_dist, "cos_sim": cos_sim, "p_dist": float(p_dist), "shared_kfs": shared_kfs}
        logger.debug("REJECTED i1=%s i2=%s | overlap: p_dist=%.3f (cos_sim=%s centroid_dist=%s)", i1.id, i2.id, p_dist, cos_sim, centroid_dist)
        return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "overlap", "centroid_dist": centroid_dist, "cos_sim": cos_sim, "p_dist": float(p_dist), "shared_kfs": shared_kfs}


class PointOverlapOldCriterion(Criterion):
    """Asymmetric overlap: fraction of points1 within th_points of pcd2."""
    name = "overlap_old"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def check(self, i1, i2, p1, p2, ctx):
        p_dist = compute_pcd_old_overlap(p1, p2, self.threshold)
        ctx["p_dist"] = float(p_dist)
        cos_sim = ctx.get("cos_sim")
        centroid_dist = ctx.get("centroid_dist")
        shared_kfs = ctx.get("shared_kfs")

        geom = p_dist > 0.5  # rama A: solape geométrico fuerte
        sem = cos_sim is not None and cos_sim > 0.9 and p_dist > 0.2  # rama B: semántica + solape débil
        if geom or sem:
            mode = "A" if geom and not sem else "B" if sem and not geom else "AB"
            logger.debug("ACCEPTED i1=%s i2=%s | mode=%s cos_sim=%s centroid_dist=%s p_dist=%.3f (overlap_old)", i1.id, i2.id, mode, cos_sim, centroid_dist, p_dist)
            return True, {"result": "ACCEPTED", "accept_mode": mode, "i1": i1.id, "i2": i2.id, "centroid_dist": centroid_dist, "cos_sim": cos_sim, "p_dist": float(p_dist), "shared_kfs": shared_kfs}
        logger.debug("REJECTED i1=%s i2=%s | overlap_old: p_dist=%.3f (cos_sim=%s centroid_dist=%s)", i1.id, i2.id, p_dist, cos_sim, centroid_dist)
        return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "overlap_old", "centroid_dist": centroid_dist, "cos_sim": cos_sim, "p_dist": float(p_dist), "shared_kfs": shared_kfs}


class VoxelOverlapCriterion(Criterion):
    """Overlap over a shared voxel grid: |A n B| / min(|A|, |B|) in occupied cells.

    Same two-branch accept rule as overlap_old (strong geometric, or weak + semantic), but
    the ratio is integer-set intersection of voxel cells instead of point nearest-neighbour
    distance. Cells are built once per pass in `prepare`, not recomputed per pair.
    """
    name = "overlap_voxel"

    def __init__(self, voxel_size: float, th_geom: float = 0.5, th_sem: float = 0.2, origin: float = 0.0):
        self.voxel_size = voxel_size
        self.th_geom = th_geom
        self.th_sem = th_sem
        self.origin = origin
        self._index = {}

    def prepare(self, objects_list, obj_pcds) -> None:
        self._index = {
            inst.id: VoxelIndex.build(obj_pcds[inst.id], self.voxel_size, self.origin)
            for inst in objects_list
        }

    def candidate_pairs(self):
        """Pairs of instances that share at least one voxel cell (the broadphase set).

        A pair that shares no cell has zero overlap, so this criterion can never accept it;
        enumerating only shared-cell pairs is therefore lossless and replaces the O(N^2) loop.
        Built from the cells already in `_index`: concatenate every (cell, instance), sort by
        cell, and within each run of equal cells emit all instance pairs.
        """
        keys, ids = [], []
        for inst_id, idx in self._index.items():
            if len(idx):
                keys.append(idx.cells)
                ids.append(torch.full_like(idx.cells, inst_id))
        if not keys:
            return set()
        keys = torch.cat(keys)
        ids = torch.cat(ids)
        order = torch.argsort(keys)
        keys, ids = keys[order], ids[order]
        _, counts = torch.unique_consecutive(keys, return_counts=True)
        pairs = set()
        start = 0
        for cnt in counts.tolist():
            if cnt > 1:  # a cell shared by 2+ instances -> those instances are candidates
                group = ids[start:start + cnt].tolist()
                for a, b in combinations(group, 2):
                    pairs.add((a, b) if a < b else (b, a))
            start += cnt
        return pairs

    def check(self, i1, i2, p1, p2, ctx):
        a, b = self._index[i1.id], self._index[i2.id]
        denom = min(len(a), len(b))
        vox = a.shared(b) / denom if denom else 0.0
        ctx["p_dist"] = float(vox)
        cos_sim = ctx.get("cos_sim")
        centroid_dist = ctx.get("centroid_dist")
        shared_kfs = ctx.get("shared_kfs")

        geom = vox > self.th_geom  # rama A: solape geométrico fuerte
        sem = cos_sim is not None and cos_sim > 0.9 and vox > self.th_sem  # rama B: semántica + solape débil
        if geom or sem:
            mode = "A" if geom and not sem else "B" if sem and not geom else "AB"
            logger.debug("ACCEPTED i1=%s i2=%s | mode=%s cos_sim=%s centroid_dist=%s vox=%.3f (overlap_voxel)", i1.id, i2.id, mode, cos_sim, centroid_dist, vox)
            return True, {"result": "ACCEPTED", "accept_mode": mode, "i1": i1.id, "i2": i2.id, "centroid_dist": centroid_dist, "cos_sim": cos_sim, "p_dist": float(vox), "shared_kfs": shared_kfs}
        logger.debug("REJECTED i1=%s i2=%s | overlap_voxel: vox=%.3f (cos_sim=%s centroid_dist=%s)", i1.id, i2.id, vox, cos_sim, centroid_dist)
        return False, {"result": "REJECTED", "i1": i1.id, "i2": i2.id, "reason": "overlap_voxel", "centroid_dist": centroid_dist, "cos_sim": cos_sim, "p_dist": float(vox), "shared_kfs": shared_kfs}
