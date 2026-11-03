"""The shipped config spells out the SimulatedSLAM defaults: the YAML block must match the code."""

from pathlib import Path

import yaml

from ovo.slam.simulated import SimulatedCfg

OVO_YAML = Path(__file__).resolve().parents[2] / "data" / "working" / "configs" / "ovo.yaml"


def test_ovo_yaml_simulated_block_equals_defaults():
    slam = yaml.safe_load(OVO_YAML.read_text())["slam"]
    assert "simulated" in slam
    assert SimulatedCfg.from_dict(slam["simulated"]) == SimulatedCfg()
    # every default is spelled out, not left implicit
    assert set(slam["simulated"]) == {"keyframes", "drift", "jumps", "jump_seed", "correction"}
