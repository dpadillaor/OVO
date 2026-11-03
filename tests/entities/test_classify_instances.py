"""OVO.classify_instances (ovo.entities.ovo): one class and one confidence per instance.

The OVO is built without its constructor and `query` is stubbed with a fixed (N, K) similarity
map, so no CLIP model is needed.
"""

import numpy as np
import pytest
import torch

from ovo.entities.ovo import OVO


def classify(sim_map, th=0):
    ovo = OVO.__new__(OVO)
    ovo.query = lambda classes, template: sim_map
    classes = [f"class{k}" for k in range(sim_map.shape[1])]
    return ovo.classify_instances(classes, th=th)


@pytest.mark.parametrize("sim_map", [
    torch.tensor([[0.1, 0.7, 0.2]]),
    torch.tensor([[0.1, 0.7, 0.2], [0.9, 0.0, 0.1], [0.3, 0.3, 0.4]]),
], ids=["one_instance", "three_instances"])
def test_one_class_and_conf_per_instance(sim_map):
    info = classify(sim_map)
    n = sim_map.shape[0]
    assert info["classes"].shape == (n,)
    assert info["conf"].shape == (n,)
    np.testing.assert_array_equal(info["classes"], sim_map.argmax(dim=1).numpy())
    np.testing.assert_allclose(info["conf"], sim_map.max(dim=1).values.numpy())


def test_single_instance_conf_can_be_indexed():
    info = classify(torch.tensor([[0.2, 0.8]]))
    assert info["classes"][0] == 1
    assert info["conf"][0] == pytest.approx(0.8)


def test_below_threshold_is_unclassified_with_zero_conf():
    sim_map = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.2, 0.25], [0.5, 0.0, 0.1]])
    info = classify(sim_map, th=0.5)
    np.testing.assert_array_equal(info["classes"], [1, -1, -1])  # 0.5 itself is not above th
    np.testing.assert_allclose(info["conf"], [0.7, 0.0, 0.0])


def test_single_instance_below_threshold():
    info = classify(torch.tensor([[0.2, 0.3]]), th=0.5)
    np.testing.assert_array_equal(info["classes"], [-1])
    np.testing.assert_allclose(info["conf"], [0.0])
