"""Gates (ovo.entities.fusion.gates): centroid veto and gate construction."""

import pytest
import torch

from ovo.entities.fusion.gates import GATES, CentroidGate, build_gate


def _gate(max_dist, pcds):
    gate = build_gate("centroid", {"max_dist": max_dist})
    gate.prepare([], pcds)
    return gate


def _at(*centre, n=4):
    """A cloud of n points symmetric around `centre`, so its mean is exactly `centre`."""
    c = torch.tensor(centre, dtype=torch.float64)
    offsets = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float64)[:n]
    return c + 0.01 * offsets


def test_registry_and_defaults():
    assert GATES == {"centroid": CentroidGate}
    assert build_gate("centroid", {}).params == CentroidGate.Params(max_dist=1.5)


def test_check_returns_the_centroid_distance():
    gate = _gate(1.5, {1: _at(0, 0, 0), 2: _at(0.3, 0.4, 0)})
    ok, dist = gate.check(1, 2)
    assert ok
    assert dist == pytest.approx(0.5)


def test_vetoes_above_max_dist():
    gate = _gate(1.0, {1: _at(0, 0, 0), 2: _at(2, 0, 0)})
    assert gate.check(1, 2) == (False, pytest.approx(2.0))
    assert gate.check(2, 1)[0] is False


def test_distance_exactly_max_dist_passes():
    gate = _gate(1.5, {1: _at(0, 0, 0), 2: _at(1.5, 0, 0)})
    ok, dist = gate.check(1, 2)
    assert dist == 1.5
    assert ok


def test_unknown_gate():
    with pytest.raises(ValueError, match=r"instance_fusion\.gates: unknown gate 'centroidd'"):
        build_gate("centroidd", {})


def test_bad_gate_param():
    with pytest.raises(ValueError, match=r"instance_fusion\.gates\.centroid:.*max"):
        build_gate("centroid", {"max": 1.0})
