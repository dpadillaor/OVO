"""NetworkX-backed covisibility graph and its event bus."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Protocol

import networkx as nx
import torch

from .dataclasses import CovisibilityEdge, CovisiblePair, KeyFrameData
from .estimator import IVisibilityEstimator


_EVENT_KF_ADDED = "on_keyframe_added"
_EVENT_KF_REMOVED = "on_keyframe_removed"
_EVENT_EDGE_UPDATED = "on_edge_updated"


class CovisibilityEventBus:
    """Lightweight pub/sub so the graph can notify OVO without coupling."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = {
            _EVENT_KF_ADDED: [],
            _EVENT_KF_REMOVED: [],
            _EVENT_EDGE_UPDATED: [],
        }

    def on_keyframe_added(self, handler: Callable[[KeyFrameData], None]) -> None:
        self._register(_EVENT_KF_ADDED, handler)

    def on_keyframe_removed(self, handler: Callable[[int], None]) -> None:
        self._register(_EVENT_KF_REMOVED, handler)

    def on_edge_updated(self, handler: Callable[[CovisibilityEdge], None]) -> None:
        self._register(_EVENT_EDGE_UPDATED, handler)

    def emit(self, event: str, payload: Any) -> None:
        for handler in self._handlers.get(event, []):
            handler(payload)

    def _register(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)


class ICovisibilityGraph(Protocol):
    def add_keyframe(self, kf: KeyFrameData) -> None: ...
    def remove_keyframe(self, kf_id: int) -> None: ...
    def get_covisible(self, kf_id: int, top_n: int) -> List[CovisibilityEdge]: ...
    def get_covisible_pairs_for_instances(
        self, instances: Iterable[Any], min_overlap: float
    ) -> List[CovisiblePair]: ...


class CovisibilityGraph:
    """Maintain and query a covisibility graph between keyframes."""

    def __init__(
        self,
        estimator: IVisibilityEstimator,
        event_bus: CovisibilityEventBus | None = None,
        logger: Any | None = None,
    ) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._keyframes: Dict[int, KeyFrameData] = {}
        self._estimator = estimator
        self._event_bus = event_bus or CovisibilityEventBus()
        self._logger = logger

    def _record_time(self, op: str, duration: float) -> None:
        if self._logger is not None and hasattr(self._logger, "record_covisibility_time"):
            self._logger.record_covisibility_time(op, duration)

    @property
    def event_bus(self) -> CovisibilityEventBus:
        return self._event_bus

    def add_keyframe(self, kf: KeyFrameData) -> None:
        if kf.kf_id in self._keyframes:
            return
        t0 = time.perf_counter()
        self._graph.add_node(kf.kf_id)
        self._keyframes[kf.kf_id] = kf
        self._recompute_edges(kf)
        self._record_time("add_keyframe", time.perf_counter() - t0)
        self._event_bus.emit(_EVENT_KF_ADDED, kf)

    def remove_keyframe(self, kf_id: int) -> None:
        if kf_id not in self._keyframes:
            return
        self._graph.remove_node(kf_id)
        self._keyframes.pop(kf_id)
        self._event_bus.emit(_EVENT_KF_REMOVED, kf_id)

    def get_covisible(self, kf_id: int, top_n: int) -> List[CovisibilityEdge]:
        if kf_id not in self._graph:
            return []
        edges = [
            self._build_edge(kf_id, neighbor, data["overlap_ratio"])
            for neighbor, data in self._graph[kf_id].items()
        ]
        edges.sort(key=lambda e: e.overlap_ratio, reverse=True)
        return edges[: max(top_n, 0)]

    def get_covisible_pairs_for_instances(
        self, instances: Iterable[Any], min_overlap: float
    ) -> List[CovisiblePair]:
        t0 = time.perf_counter()
        pairs = self._resolve_instance_pairs(list(instances), min_overlap)
        self._record_time("pairs_query", time.perf_counter() - t0)
        return pairs

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._serialize(), f)

    def load(self, path: Path) -> None:
        with Path(path).open("r") as f:
            self._deserialize(json.load(f))

    def _recompute_edges(self, kf: KeyFrameData) -> None:
        candidates = [other for kid, other in self._keyframes.items() if kid != kf.kf_id]
        if not candidates:
            return
        edges = self._estimator.estimate_candidates(kf, candidates)
        for edge in edges:
            self._graph.add_edge(
                edge.kf_id_a, edge.kf_id_b, overlap_ratio=edge.overlap_ratio
            )
            self._event_bus.emit(_EVENT_EDGE_UPDATED, edge)

    def _build_edge(self, kf_a: int, kf_b: int, overlap_ratio: float) -> CovisibilityEdge:
        return CovisibilityEdge(kf_id_a=kf_a, kf_id_b=kf_b, overlap_ratio=overlap_ratio)

    def _resolve_instance_pairs(
        self, instances: List[Any], min_overlap: float
    ) -> List[CovisiblePair]:
        pairs: List[CovisiblePair] = []
        seen: set[tuple[int, int]] = set()

        for i, inst_a in enumerate(instances):
            kfs_a = getattr(inst_a, "kfs_ids", None) or []
            if not kfs_a:
                continue
            for inst_b in instances[i + 1 :]:
                kfs_b = getattr(inst_b, "kfs_ids", None) or []
                if not kfs_b:
                    continue
                key = (inst_a.id, inst_b.id) if inst_a.id < inst_b.id else (inst_b.id, inst_a.id)
                if key in seen:
                    continue
                best = self._find_best_edge(kfs_a, kfs_b, min_overlap)
                if best is None:
                    continue
                ka, kb, ratio = best
                pairs.append(
                    CovisiblePair(
                        instance_id_a=inst_a.id,
                        instance_id_b=inst_b.id,
                        kf_id_a=ka,
                        kf_id_b=kb,
                        overlap_ratio=ratio,
                    )
                )
                seen.add(key)
        return pairs

    def _find_best_edge(
        self, kfs_a: Iterable[int], kfs_b: Iterable[int], min_overlap: float
    ) -> tuple[int, int, float] | None:
        best: tuple[int, int, float] | None = None
        kfs_b_set = set(kfs_b)
        for ka in kfs_a:
            if ka not in self._graph:
                continue
            for kb, data in self._graph[ka].items():
                if kb not in kfs_b_set:
                    continue
                ratio = data["overlap_ratio"]
                if ratio < min_overlap:
                    continue
                if best is None or ratio > best[2]:
                    best = (ka, kb, ratio)
        return best

    def _serialize(self) -> dict:
        return {
            "keyframes": [
                {
                    "kf_id": kf.kf_id,
                    "frame_id": kf.frame_id,
                    "c2w": kf.c2w.detach().cpu().tolist(),
                    "depth_shape": list(kf.depth.shape),
                    "cam_intrinsics": kf.cam_intrinsics.detach().cpu().tolist(),
                }
                for kf in self._keyframes.values()
            ],
            "edges": [
                {"kf_id_a": u, "kf_id_b": v, "overlap_ratio": d["overlap_ratio"]}
                for u, v, d in self._graph.edges(data=True)
            ],
        }

    def _deserialize(self, data: dict) -> None:
        self._graph = nx.Graph()
        self._keyframes = {}
        for kf in data.get("keyframes", []):
            kf_id = int(kf["kf_id"])
            self._graph.add_node(kf_id)
            self._keyframes[kf_id] = KeyFrameData(
                kf_id=kf_id,
                frame_id=int(kf["frame_id"]),
                c2w=torch.tensor(kf["c2w"]),
                depth=torch.zeros(*kf["depth_shape"]),
                cam_intrinsics=torch.tensor(kf["cam_intrinsics"]),
            )
        for edge in data.get("edges", []):
            self._graph.add_edge(
                int(edge["kf_id_a"]),
                int(edge["kf_id_b"]),
                overlap_ratio=float(edge["overlap_ratio"]),
            )
