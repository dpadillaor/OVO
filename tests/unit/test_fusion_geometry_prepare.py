"""Centroid and aabb criteria precompute their per-instance summary in `prepare`."""

from types import SimpleNamespace

import torch

from ovo.entities.fusion import CentroidDistanceCriterion, AabbDistanceCriterion


def _pts(coords):
    return torch.tensor(coords, dtype=torch.float32)


def _prepared(criterion, clouds):
    objs = [SimpleNamespace(id=i) for i in clouds]
    criterion.prepare(objs, dict(clouds))
    return {i: SimpleNamespace(id=i) for i in clouds}


class TestCentroidPrepare:
    def test_centroid_is_the_mean_of_the_cloud(self):
        crit = CentroidDistanceCriterion(threshold=1.5)
        _prepared(crit, {1: _pts([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])})
        assert torch.allclose(crit._centroid[1], _pts([1.0, 0.0, 0.0]))

    def test_close_centroids_pass(self):
        crit = CentroidDistanceCriterion(threshold=1.5)
        ins = _prepared(crit, {1: _pts([[0.0, 0, 0]]), 2: _pts([[0.5, 0, 0]])})
        verdict, _ = crit.check(ins[1], ins[2], None, None, {})
        assert verdict is None  # within threshold -> pass to next criterion

    def test_far_centroids_reject(self):
        crit = CentroidDistanceCriterion(threshold=1.5)
        ins = _prepared(crit, {1: _pts([[0.0, 0, 0]]), 2: _pts([[3.0, 0, 0]])})
        verdict, dec = crit.check(ins[1], ins[2], None, None, {})
        assert verdict is False and dec["reason"] == "centroid"


class TestAabbPrepare:
    def test_bbox_is_min_max_of_the_cloud(self):
        crit = AabbDistanceCriterion(threshold=0.3)
        _prepared(crit, {1: _pts([[0.0, -1.0, 2.0], [1.0, 0.0, 3.0]])})
        lo, hi = crit._bbox[1]
        assert torch.allclose(lo, _pts([0.0, -1.0, 2.0]))
        assert torch.allclose(hi, _pts([1.0, 0.0, 3.0]))

    def test_overlapping_boxes_pass(self):
        crit = AabbDistanceCriterion(threshold=0.3)
        a = _pts([[0.0, 0, 0], [1.0, 1.0, 1.0]])
        b = _pts([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])  # boxes overlap -> gap 0
        ins = _prepared(crit, {1: a, 2: b})
        verdict, _ = crit.check(ins[1], ins[2], None, None, {})
        assert verdict is None

    def test_far_boxes_reject(self):
        crit = AabbDistanceCriterion(threshold=0.3)
        a = _pts([[0.0, 0, 0], [1.0, 1.0, 1.0]])
        b = _pts([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])  # far -> gap > 0.3
        ins = _prepared(crit, {1: a, 2: b})
        verdict, dec = crit.check(ins[1], ins[2], None, None, {})
        assert verdict is False and dec["reason"] == "aabb"
