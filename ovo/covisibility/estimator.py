"""Frustum overlap covisibility estimator.

Delegates all geometry to ``ovo.utils.geometry_utils`` — does not reimplement
frustum maths. The estimator answers a single question: do two keyframes see
each other enough to compare their semantic content.
"""
from __future__ import annotations

from typing import List, Protocol

import torch

from ovo.utils import geometry_utils

from .dataclasses import CovisibilityEdge, KeyFrameData


class IVisibilityEstimator(Protocol):
    def estimate_pair(self, kf_a: KeyFrameData, kf_b: KeyFrameData) -> float: ...

    def estimate_candidates(
        self, query: KeyFrameData, candidates: List[KeyFrameData]
    ) -> List[CovisibilityEdge]: ...


class FrustumOverlapEstimator:
    """Compute covisibility between two keyframes via frustum-corner overlap.

    The 8 frustum corners of ``kf_b`` are tested against the frustum of
    ``kf_a``; ``overlap_ratio`` is the fraction that fall inside.
    """

    def __init__(self, min_overlap_ratio: float = 0.3, max_distance: float = 5.0) -> None:
        self._min_overlap_ratio = float(min_overlap_ratio)
        self._max_distance = float(max_distance)

    def estimate_pair(self, kf_a: KeyFrameData, kf_b: KeyFrameData) -> float:
        if not self._is_within_distance(kf_a, kf_b):
            return 0.0
        return self._compute_overlap(kf_a, kf_b)

    def estimate_candidates(
        self, query: KeyFrameData, candidates: List[KeyFrameData]
    ) -> List[CovisibilityEdge]:
        edges: List[CovisibilityEdge] = []
        for cand in candidates:
            if cand.kf_id == query.kf_id:
                continue
            ratio = self.estimate_pair(query, cand)
            if ratio >= self._min_overlap_ratio:
                edges.append(
                    CovisibilityEdge(
                        kf_id_a=query.kf_id,
                        kf_id_b=cand.kf_id,
                        overlap_ratio=ratio,
                    )
                )
        return edges

    def _get_frustum_corners(self, kf: KeyFrameData) -> torch.Tensor:
        return geometry_utils.compute_camera_frustum_corners(
            kf.depth, kf.c2w, kf.cam_intrinsics
        )

    def _compute_overlap(self, kf_a: KeyFrameData, kf_b: KeyFrameData) -> float:
        # NOTE: geometry_utils.compute_camera_frustum_planes has a known latent
        # bug in plane-D index assignment. Tracked separately; not blocking here.
        corners_a = self._get_frustum_corners(kf_a)
        corners_b = self._get_frustum_corners(kf_b)
        device = corners_a.device
        ids_inside = geometry_utils.compute_frustum_point_ids(
            corners_b, corners_a, device=str(device)
        )
        return float(ids_inside.numel()) / float(corners_b.shape[0])

    def _is_within_distance(self, kf_a: KeyFrameData, kf_b: KeyFrameData) -> bool:
        t_a = kf_a.c2w[:3, 3]
        t_b = kf_b.c2w[:3, 3]
        return torch.linalg.norm(t_a - t_b).item() <= self._max_distance
