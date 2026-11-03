"""Fusion config (ovo.entities.fusion.config): defaults, partial overrides, validation."""

import pytest

from ovo.entities.fusion.config import FusionCfg, GateSpec
from ovo.entities.fusion.overlap import OverlapCfg, PointCfg, VoxelCfg
from ovo.entities.fusion.rule import AcceptCfg


def test_missing_section_gives_defaults():
    assert FusionCfg.from_dict(None) == FusionCfg()
    assert FusionCfg.from_dict({}) == FusionCfg()


def test_defaults():
    cfg = FusionCfg()
    assert cfg.gates == ()
    assert cfg.overlap == OverlapCfg(method="voxel", voxel=VoxelCfg(0.05, (0.0, 0.0, 0.0)), point=PointCfg(0.1))
    assert cfg.accept == AcceptCfg(min_cos=0.81, strong=0.5, weak=0.2, weak_min_cos=0.9)
    assert cfg.apply == "greedy"


def test_partial_override_keeps_other_defaults():
    cfg = FusionCfg.from_dict({"accept": {"strong": 0.6}, "overlap": {"point": {"max_dist": 0.2}}})
    assert cfg.accept == AcceptCfg(strong=0.6)
    assert cfg.overlap == OverlapCfg(point=PointCfg(0.2))
    assert cfg.gates == () and cfg.apply == "greedy"


def test_does_not_mutate_the_input():
    raw = {"gates": ["centroid"], "apply": "chained"}
    FusionCfg.from_dict(raw)
    assert raw == {"gates": ["centroid"], "apply": "chained"}


@pytest.mark.parametrize("raw, path", [
    ({"gate": []}, r"instance_fusion: unknown key\(s\) \['gate'\]"),
    ({"accept": {"min_cs": 0.8}}, r"instance_fusion\.accept: unknown key\(s\) \['min_cs'\]"),
    ({"overlap": {"voxel": {"sise": 0.1}}}, r"instance_fusion\.overlap\.voxel: unknown key\(s\) \['sise'\]"),
])
def test_unknown_keys_raise_with_their_path(raw, path):
    with pytest.raises(ValueError, match=path):
        FusionCfg.from_dict(raw)


def test_origin_list_becomes_tuple():
    cfg = FusionCfg.from_dict({"overlap": {"voxel": {"origin": [0.01, 0.0, 0.0]}}})
    assert cfg.overlap.voxel.origin == (0.01, 0.0, 0.0)
    assert isinstance(cfg.overlap.voxel.origin, tuple)


@pytest.mark.parametrize("entry, spec", [
    ("centroid", GateSpec("centroid")),
    ({"centroid": {"max_dist": 1.0}}, GateSpec("centroid", {"max_dist": 1.0})),
    ({"centroid": None}, GateSpec("centroid")),
])
def test_gate_entry_formats(entry, spec):
    assert FusionCfg.from_dict({"gates": [entry]}).gates == (spec,)


@pytest.mark.parametrize("entry", [{"centroid": {}, "other": {}}, 3, ["centroid"]])
def test_bad_gate_entry(entry):
    with pytest.raises(ValueError, match=r"instance_fusion\.gates: each entry"):
        FusionCfg.from_dict({"gates": [entry]})
