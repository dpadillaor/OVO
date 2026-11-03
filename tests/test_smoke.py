"""Smoke test: the lightweight ovo modules import and run on CPU."""

import torch

from ovo.utils import instance_utils


class _FakeInstance:
    def __init__(self, clip_feature):
        self.clip_feature = clip_feature


def test_same_instance_on_identical_clouds():
    points = torch.rand(50, 3)
    centroid = points.mean(0)
    feature = torch.ones(1, 8)
    instance = _FakeInstance(feature)

    assert instance_utils.same_instance(
        instance, _FakeInstance(feature.clone()),
        (points, centroid), (points.clone(), centroid.clone()),
        th_centroid=0.1, th_cossim=0.81, th_points=0.01,
    )


def test_same_instance_rejects_distant_centroids():
    points = torch.rand(50, 3)
    far = points + 10.0
    feature = torch.ones(1, 8)

    assert not instance_utils.same_instance(
        _FakeInstance(feature), _FakeInstance(feature),
        (points, points.mean(0)), (far, far.mean(0)),
        th_centroid=0.1, th_cossim=0.81, th_points=0.01,
    )
