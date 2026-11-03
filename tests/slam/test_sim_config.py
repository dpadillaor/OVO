"""config: the YAML section slam.simulated, validated."""

import pytest

from ovo.slam.simulated import SimulatedCfg
from ovo.slam.simulated.config import CorrectionCfg
from ovo.slam.simulated.drift import DriftCfg
from ovo.slam.simulated.jumps import JumpCfg
from ovo.slam.simulated.keyframes import KeyframeCfg


def test_defaults():
    cfg = SimulatedCfg()
    assert cfg.keyframes == KeyframeCfg(dist=0.1, rot_deg=5.0)
    assert cfg.drift is None
    assert cfg.jumps == ()
    assert cfg.jump_seed == 42
    assert cfg.correction == CorrectionCfg(at_frames=())


@pytest.mark.parametrize("raw", [None, {}])
def test_empty_is_defaults(raw):
    assert SimulatedCfg.from_dict(raw) == SimulatedCfg()


def test_nested_building():
    cfg = SimulatedCfg.from_dict({
        "keyframes": {"dist": 0.2},
        "drift": {"translation_std": 0.02, "seed": 1},
        "jumps": [{"at_frame": 300, "translation": [0, 0, -0.1]},
                  {"at_frame": 10, "rotation": 5, "axes": "world"}],
        "jump_seed": "7",
        "correction": {"at_frames": [500, "100"]},
    })
    assert cfg.keyframes == KeyframeCfg(dist=0.2, rot_deg=5.0)
    assert cfg.drift == DriftCfg(translation_std=0.02, rotation_std_deg=0.5, seed=1)
    assert cfg.jumps == (JumpCfg(at_frame=300, translation=[0, 0, -0.1]),
                         JumpCfg(at_frame=10, rotation=5, axes="world"))
    assert cfg.jump_seed == 7
    assert cfg.correction.at_frames == (500, 100)


@pytest.mark.parametrize("raw", [{"drift": None}, {}])
def test_drift_null_or_absent_is_none(raw):
    assert SimulatedCfg.from_dict(raw).drift is None


def test_drift_empty_dict_is_default_drift():
    assert SimulatedCfg.from_dict({"drift": {}}).drift == DriftCfg()


@pytest.mark.parametrize("raw", [{"jumps": None}, {"jumps": []}, {"correction": None},
                                 {"correction": {"at_frames": None}}, {"keyframes": None}])
def test_null_sections_are_defaults(raw):
    assert SimulatedCfg.from_dict(raw) == SimulatedCfg()


@pytest.mark.parametrize("raw, path", [
    ({"jumpz": []}, "slam.simulated:"),
    ({"keyframes": {"distance": 0.1}}, "slam.simulated.keyframes"),
    ({"drift": {"std": 0.1}}, "slam.simulated.drift"),
    ({"jumps": [{"at_frame": 1}, {"at_frame": 2, "frame": 3}]}, r"slam.simulated.jumps\[1\]"),
    ({"correction": {"frames": [1]}}, "slam.simulated.correction"),
])
def test_unknown_keys_raise_at_every_level(raw, path):
    with pytest.raises(ValueError, match=path + ".*unknown key"):
        SimulatedCfg.from_dict(raw)


def test_unknown_key_message_lists_valid_keys():
    with pytest.raises(ValueError, match="valid: \\['dist', 'rot_deg'\\]"):
        SimulatedCfg.from_dict({"keyframes": {"distance": 0.1}})


def test_invalid_values_raise():
    with pytest.raises(ValueError, match="std"):
        SimulatedCfg.from_dict({"drift": {"translation_std": -1}})
    with pytest.raises(ValueError, match="axes"):
        SimulatedCfg.from_dict({"jumps": [{"at_frame": 1, "axes": "body"}]})


def test_from_dict_does_not_mutate_input():
    raw = {"correction": {"at_frames": [3]}, "jumps": [{"at_frame": 1}]}
    SimulatedCfg.from_dict(raw)
    assert raw == {"correction": {"at_frames": [3]}, "jumps": [{"at_frame": 1}]}


@pytest.mark.parametrize("raw, match", [
    ({"jumps": [{"translation": 0.1}]}, r"jumps\[0\]: each jump needs `at_frame`"),
    ({"jumps": ["oops"]}, r"jumps\[0\]: each jump needs `at_frame`"),
    ({"jumps": [{"at_frame": 1, "translation": "abc"}]}, "translation must be a number or 3 numbers"),
    ({"correction": {"at_frames": [10, -1]}}, "at_frames: must be >= 0"),
    ({"keyframes": {"dist": -0.1}}, "thresholds must be >= 0"),
])
def test_stricter_validation(raw, match):
    with pytest.raises(ValueError, match=match):
        SimulatedCfg.from_dict(raw)
