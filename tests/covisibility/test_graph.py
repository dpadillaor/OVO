"""Unit tests for CovisibilityGraph and CovisibilityEventBus."""
from dataclasses import dataclass
from typing import List

import torch

from ovo.covisibility.dataclasses import CovisibilityEdge, KeyFrameData
from ovo.covisibility.graph import CovisibilityEventBus, CovisibilityGraph


class StubEstimator:
    """Estimator stub: returns predefined overlaps for any pair lookup."""

    def __init__(self, overlaps: dict[tuple[int, int], float]):
        self._overlaps = overlaps

    def estimate_pair(self, kf_a: KeyFrameData, kf_b: KeyFrameData) -> float:
        return self._lookup(kf_a.kf_id, kf_b.kf_id)

    def estimate_candidates(
        self, query: KeyFrameData, candidates: List[KeyFrameData]
    ) -> List[CovisibilityEdge]:
        edges = []
        for cand in candidates:
            if cand.kf_id == query.kf_id:
                continue
            ratio = self._lookup(query.kf_id, cand.kf_id)
            if ratio > 0.0:
                edges.append(
                    CovisibilityEdge(
                        kf_id_a=query.kf_id,
                        kf_id_b=cand.kf_id,
                        overlap_ratio=ratio,
                    )
                )
        return edges

    def _lookup(self, a: int, b: int) -> float:
        key = (a, b) if (a, b) in self._overlaps else (b, a)
        return self._overlaps.get(key, 0.0)


@dataclass
class _StubInstance:
    id: int
    kfs_ids: list


def _kf(kf_id: int) -> KeyFrameData:
    return KeyFrameData(
        kf_id=kf_id,
        frame_id=kf_id,
        c2w=torch.eye(4),
        depth=torch.ones(4, 4),
        cam_intrinsics=torch.eye(3),
    )


def test_add_keyframe_creates_node_and_edges():
    est = StubEstimator({(0, 1): 0.7})
    g = CovisibilityGraph(est)
    g.add_keyframe(_kf(0))
    g.add_keyframe(_kf(1))
    edges = g.get_covisible(0, top_n=10)
    assert len(edges) == 1
    assert edges[0].kf_id_b == 1
    assert edges[0].overlap_ratio == 0.7


def test_remove_keyframe_drops_node_and_edges():
    est = StubEstimator({(0, 1): 0.5})
    g = CovisibilityGraph(est)
    g.add_keyframe(_kf(0))
    g.add_keyframe(_kf(1))
    g.remove_keyframe(1)
    assert g.get_covisible(0, top_n=10) == []


def test_get_covisible_sorted_by_overlap_desc():
    est = StubEstimator({(0, 1): 0.4, (0, 2): 0.9, (0, 3): 0.6})
    g = CovisibilityGraph(est)
    for kid in (0, 1, 2, 3):
        g.add_keyframe(_kf(kid))
    edges = g.get_covisible(0, top_n=2)
    assert [e.kf_id_b for e in edges] == [2, 3]


def test_get_covisible_pairs_for_instances_empty_when_no_shared_kfs():
    est = StubEstimator({})
    g = CovisibilityGraph(est)
    g.add_keyframe(_kf(0))
    g.add_keyframe(_kf(1))
    instances = [_StubInstance(id=10, kfs_ids=[0]), _StubInstance(id=11, kfs_ids=[1])]
    assert g.get_covisible_pairs_for_instances(instances, min_overlap=0.1) == []


def test_get_covisible_pairs_returns_pair_when_kfs_covisible():
    est = StubEstimator({(0, 1): 0.8})
    g = CovisibilityGraph(est)
    g.add_keyframe(_kf(0))
    g.add_keyframe(_kf(1))
    instances = [
        _StubInstance(id=10, kfs_ids=[0]),
        _StubInstance(id=11, kfs_ids=[1]),
    ]
    pairs = g.get_covisible_pairs_for_instances(instances, min_overlap=0.5)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.instance_id_a == 10 and p.instance_id_b == 11
    assert p.overlap_ratio == 0.8


def test_get_covisible_pairs_respects_min_overlap_filter():
    est = StubEstimator({(0, 1): 0.2})
    g = CovisibilityGraph(est)
    g.add_keyframe(_kf(0))
    g.add_keyframe(_kf(1))
    instances = [
        _StubInstance(id=10, kfs_ids=[0]),
        _StubInstance(id=11, kfs_ids=[1]),
    ]
    assert g.get_covisible_pairs_for_instances(instances, min_overlap=0.5) == []


def test_event_bus_dispatch():
    bus = CovisibilityEventBus()
    added: list = []
    removed: list = []
    edges: list = []
    bus.on_keyframe_added(added.append)
    bus.on_keyframe_removed(removed.append)
    bus.on_edge_updated(edges.append)

    g = CovisibilityGraph(StubEstimator({(0, 1): 0.6}), event_bus=bus)
    g.add_keyframe(_kf(0))
    g.add_keyframe(_kf(1))
    g.remove_keyframe(0)

    assert len(added) == 2
    assert len(removed) == 1 and removed[0] == 0
    assert len(edges) == 1 and edges[0].overlap_ratio == 0.6


def test_save_and_load_roundtrip(tmp_path):
    est = StubEstimator({(0, 1): 0.55})
    g = CovisibilityGraph(est)
    g.add_keyframe(_kf(0))
    g.add_keyframe(_kf(1))
    path = tmp_path / "graph.json"
    g.save(path)

    g2 = CovisibilityGraph(StubEstimator({}))
    g2.load(path)
    edges = g2.get_covisible(0, top_n=10)
    assert len(edges) == 1
    assert edges[0].overlap_ratio == 0.55
