"""ovo.replay.checkpoint: modes, files and the `replay` block of the config."""

from pathlib import Path

import pytest
import torch
import yaml

from ovo import layout
from ovo.replay import checkpoint, state
from ovo.utils import io_utils
from replay_fakes import config, scene

OVO_YAML = Path(__file__).resolve().parents[2] / "data" / "working" / "configs" / "ovo.yaml"


def _save_at(mode: str, scene_dir: Path, frames) -> checkpoint.Checkpointer:
    ovo, _, slam = scene()
    saver = checkpoint.Checkpointer(mode, scene_dir)
    for frame in frames:
        saver.save(frame, slam, ovo, config())
    return saver


@pytest.mark.parametrize("raw, mode", [
    ({}, "off"),
    ({"replay": None}, "off"),
    ({"replay": {}}, "off"),
    ({"replay": {"checkpoints": False}}, "off"),  # YAML `off`
    ({"replay": {"checkpoints": "off"}}, "off"),
    ({"replay": {"checkpoints": "last"}}, "last"),
    ({"replay": {"checkpoints": "all"}}, "all"),
])
def test_mode_from_config(raw, mode):
    assert checkpoint.mode_from_config(raw) == mode


def test_yaml_bare_off_reads_as_false_and_means_off():
    raw = yaml.safe_load("replay:\n  checkpoints: off\n")
    assert raw["replay"]["checkpoints"] is False
    assert checkpoint.mode_from_config(raw) == "off"


@pytest.mark.parametrize("raw", [
    {"replay": {"checkpoints": "every"}},
    {"replay": {"checkpoints": True}},  # YAML `on`
    {"replay": {"checkpoints": "last", "keep": 3}},
])
def test_invalid_mode_or_unknown_key_raises(raw):
    with pytest.raises(ValueError, match="replay"):
        checkpoint.mode_from_config(raw)


def test_checkpointer_rejects_an_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="replay.checkpoints"):
        checkpoint.Checkpointer("every", tmp_path)


def test_off_writes_nothing(tmp_path):
    _save_at("off", tmp_path, [100, 200])
    assert not layout.fusion_dir(tmp_path).exists()


def test_all_keeps_every_checkpoint(tmp_path):
    _save_at("all", tmp_path, [100, 200, 300])
    assert sorted(checkpoint.available(tmp_path)) == [100, 200, 300]


def test_last_keeps_only_the_latest(tmp_path):
    _save_at("last", tmp_path, [100, 200, 300])
    assert checkpoint.available(tmp_path) == {300: layout.pre_fusion_checkpoint(tmp_path, 300)}


def test_last_twice_at_the_same_frame_keeps_that_file(tmp_path):
    _save_at("last", tmp_path, [100, 100])
    assert sorted(checkpoint.available(tmp_path)) == [100]


def test_checkpoints_of_an_earlier_run_are_removed_on_creation(tmp_path):
    _save_at("all", tmp_path, [100, 2000])
    _save_at("all", tmp_path, [150])
    assert sorted(checkpoint.available(tmp_path)) == [150]
    checkpoint.Checkpointer("off", tmp_path)
    assert checkpoint.available(tmp_path) == {}


def test_file_content(tmp_path):
    ovo, map_data, slam = scene()
    cfg = config()
    checkpoint.Checkpointer("all", tmp_path).save(120, slam, ovo, cfg)
    path = layout.pre_fusion_checkpoint(tmp_path, 120)
    assert path == tmp_path / "fusion" / "pre_fusion_00120.ckpt"

    ckpt = checkpoint.load(path)
    assert set(ckpt) == {"format", "frame_id", "config", "map", "ovo"}
    assert ckpt["format"] == checkpoint.FORMAT and ckpt["frame_id"] == 120
    assert ckpt["config"] == cfg and ckpt["config"] is not cfg
    assert torch.equal(ckpt["map"]["xyz"], map_data[0])
    assert torch.equal(ckpt["map"]["obj_ids"].squeeze(-1), map_data[2])
    assert set(ckpt["ovo"]["objects"]) == set(state.capture(ovo)["objects"])


def test_save_runs_the_pending_semantic_work_first(tmp_path):
    ovo, _, slam = scene()
    calls = []
    ovo.complete_semantic_info = lambda: calls.append("done")
    checkpoint.Checkpointer("all", tmp_path).save(100, slam, ovo, config())
    assert calls == ["done"]
    calls.clear()
    checkpoint.Checkpointer("off", tmp_path).save(100, slam, ovo, config())
    assert calls == []  # off changes nothing in the run


def test_load_rejects_another_format(tmp_path):
    path = tmp_path / "pre_fusion_00001.ckpt"
    torch.save({"format": 99, "frame_id": 1}, path)
    with pytest.raises(ValueError, match="format"):
        checkpoint.load(path)


def test_load_rejects_a_format_1_checkpoint_and_says_to_re_run(tmp_path):
    """Format 1 (before created_at_frame) cannot be upgraded: the frames are not in it."""
    assert checkpoint.FORMAT == 2
    path = tmp_path / "pre_fusion_00001.ckpt"
    torch.save({"format": 1, "frame_id": 1}, path)
    with pytest.raises(ValueError, match=r"format 1, expected 2; re-run the scene with replay.checkpoints"):
        checkpoint.load(path)


def test_saved_instances_carry_created_at_frame(tmp_path):
    ovo, _, slam = scene()
    ovo.objects[10].created_at_frame, ovo.objects[20].created_at_frame = 5, 12
    checkpoint.Checkpointer("all", tmp_path).save(120, slam, ovo, config())
    objects = checkpoint.load(layout.pre_fusion_checkpoint(tmp_path, 120))["ovo"]["objects"]
    assert {i: o["created_at_frame"] for i, o in objects.items()} == {10: 5, 20: 12, 30: None, 40: None, 50: None}


def test_available(tmp_path):
    assert checkpoint.available(tmp_path / "missing") == {}
    fusion = layout.fusion_dir(tmp_path)
    fusion.mkdir()
    for name in ("pre_fusion_00100.ckpt", "pre_fusion_01990.ckpt", "pre_fusion_tmp.ckpt",
                 "fusion_decisions.csv"):
        (fusion / name).touch()
    assert checkpoint.available(tmp_path) == {100: fusion / "pre_fusion_00100.ckpt",
                                               1990: fusion / "pre_fusion_01990.ckpt"}


def test_ovo_yaml_replay_block_is_the_code_default():
    raw = yaml.safe_load(OVO_YAML.read_text())
    assert set(raw["replay"]) == {"checkpoints"}
    assert checkpoint.mode_from_config(raw) == checkpoint.mode_from_config({}) == "off"
    # read as the runs read it
    assert checkpoint.mode_from_config(io_utils.load_config(str(OVO_YAML), inherit=False)) == "off"
