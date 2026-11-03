"""The shipped config spells out the fusion defaults: the YAML block must match the code."""

from pathlib import Path

import yaml

from ovo.entities.fusion import FusionCfg

OVO_YAML = Path(__file__).resolve().parents[2] / "data" / "working" / "configs" / "ovo.yaml"


def test_ovo_yaml_instance_fusion_block_equals_defaults():
    semantic = yaml.safe_load(OVO_YAML.read_text())["semantic"]
    assert "instance_fusion" in semantic
    assert FusionCfg.from_dict(semantic["instance_fusion"]) == FusionCfg()
