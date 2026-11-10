"""ovo.replay.run: a replay with the same fusion repeats the run's fusion exactly; another
fusion changes it; replay_experiment picks the checkpoints and lays the output out."""

import csv

import pytest
import torch
import yaml

from ovo import layout
from ovo.entities.fusion import FusionLog
from ovo.entities.ovo import OVO
from ovo.replay import checkpoint
from ovo.replay.run import replay_experiment, replay_scene
from replay_fakes import SCENE_NAME, config, scene

STRICT = {"accept": {"min_cos": 1.01}}  # nothing reaches it: no merge


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _original_run(scene_dir, frame=100, cfg=None):
    """Checkpoint right before a real update_map, as OVOSemMap does, then run it."""
    cfg = cfg or config()
    ovo, map_data, slam = scene(cfg)
    checkpoint.Checkpointer("all", scene_dir).save(frame, slam, ovo, cfg)
    ids, report, _ = ovo.update_map(map_data)
    FusionLog(layout.fusion_decisions(scene_dir)).write(frame, report)
    return ovo, ids.clone(), report


def _replayed(out_dir):
    ckpt = torch.load(out_dir / "ovo_map.ckpt", weights_only=False)
    ovo = OVO.for_fusion(config()["semantic"], device="cpu")
    ovo.restore_dict(ckpt["ovo_map_params"], debug_info=False)
    return ckpt, ovo


def test_same_fusion_repeats_the_run_exactly(tmp_path):
    run_dir, out_dir = tmp_path / "run", tmp_path / "replay"
    original, original_ids, _ = _original_run(run_dir)

    replay_scene(layout.pre_fusion_checkpoint(run_dir, 100), out_dir, device="cpu")

    ckpt, replayed = _replayed(out_dir)
    assert torch.equal(ckpt["map_params"]["obj_ids"].squeeze(-1), original_ids)
    assert _rows(layout.fusion_decisions(out_dir)) == _rows(layout.fusion_decisions(run_dir))
    assert sorted(int(i) for i in replayed.objects) == sorted(original.objects) == [10, 30, 40]
    for ins_id, obj in original.objects.items():
        assert torch.equal(replayed.objects[ins_id].clip_feature, obj.clip_feature)
        assert replayed.objects[ins_id].clip_feature_kf == obj.clip_feature_kf


def test_replayed_map_keeps_the_earliest_creation_frame_of_a_fusion(tmp_path):
    """10 absorbs 20, born before it: the fused 10 is as old as 20, in the replay's ovo_map.ckpt too."""
    run_dir, out_dir = tmp_path / "run", tmp_path / "replay"
    ovo, map_data, slam = scene()
    for ins_id, frame in {10: 40, 20: 5, 30: 60, 40: 70}.items():
        ovo.objects[ins_id].created_at_frame = frame
    checkpoint.Checkpointer("all", run_dir).save(100, slam, ovo, config())

    replay_scene(layout.pre_fusion_checkpoint(run_dir, 100), out_dir, device="cpu")

    _, replayed = _replayed(out_dir)
    assert {int(i): o.created_at_frame for i, o in replayed.objects.items()} == {10: 5, 30: 60, 40: 70}


def test_a_stricter_fusion_changes_the_result(tmp_path):
    run_dir, out_dir = tmp_path / "run", tmp_path / "replay"
    _, original_ids, _ = _original_run(run_dir)

    replay_scene(layout.pre_fusion_checkpoint(run_dir, 100), out_dir, fusion=STRICT, device="cpu")

    ckpt, replayed = _replayed(out_dir)
    assert not torch.equal(ckpt["map_params"]["obj_ids"].squeeze(-1), original_ids)
    assert sorted(int(i) for i in replayed.objects) == [10, 20, 30, 40]
    assert all(r["accepted"] == "False" for r in _rows(layout.fusion_decisions(out_dir)))


def test_output_files_and_config(tmp_path):
    run_dir, out_dir = tmp_path / "run", tmp_path / "replay"
    _original_run(run_dir, frame=120)
    ckpt_path = layout.pre_fusion_checkpoint(run_dir, 120)

    replay_scene(ckpt_path, out_dir, fusion=STRICT, device="cpu")

    cfg = yaml.safe_load((out_dir / "config.yaml").read_text())
    original = config()
    assert cfg["semantic"]["instance_fusion"] == STRICT
    assert cfg["output_path"] == str(out_dir)
    assert cfg["replay"] == {"checkpoints": "off"}
    assert cfg["replayed_from"] == {"checkpoint": str(ckpt_path), "frame_id": 120}
    assert {k: v for k, v in cfg.items() if k not in ("output_path", "replay", "replayed_from", "semantic")} \
        == {k: v for k, v in original.items() if k not in ("replay", "semantic")}
    assert {r["frame_id"] for r in _rows(layout.fusion_decisions(out_dir))} == {"120"}
    for stat in ("t_fusion", "t_fusion_judge"):
        assert len((out_dir / "logger" / f"{stat}.log").read_text().split()) == 1
    ckpt = torch.load(out_dir / "ovo_map.ckpt", weights_only=False)
    assert set(ckpt) == {"map_params", "ovo_map_params"}
    assert ckpt["map_params"]["obj_ids"].shape == (len(ckpt["map_params"]["xyz"]), 1)
    assert "frame_id" not in ckpt["ovo_map_params"]  # saved as a run with debug off


@pytest.mark.parametrize("fusion, steps", [
    (None, ["cos_sim", "overlap"]),
    ({"gates": [{"aabb": {}}, {"centroid": {}}]}, ["aabb", "centroid", "cos_sim", "overlap"])])
def test_the_step_times_of_the_replayed_fusion_are_written(tmp_path, fusion, steps):
    run_dir, out_dir = tmp_path / "run", tmp_path / "replay"
    _original_run(run_dir, frame=120)
    path = layout.fusion_timings(out_dir)
    path.parent.mkdir(parents=True)
    path.write_text("left by an earlier replay\n")

    replay_scene(layout.pre_fusion_checkpoint(run_dir, 120), out_dir, fusion=fusion, device="cpu")

    assert path.read_text().splitlines()[0] == "frame_id,step,prepare_s,pairs_s"
    rows = _rows(path)
    assert [r["step"] for r in rows] == ["extract", *steps, "apply"]
    assert {r["frame_id"] for r in rows} == {"120"}
    assert all(float(r["prepare_s"]) >= 0 for r in rows)
    assert [r["pairs_s"] == "" for r in rows] == [True] + [False] * len(steps) + [True]
    assert all(float(r["pairs_s"]) >= 0 for r in rows[1:-1])


def test_an_invalid_fusion_writes_nothing(tmp_path):
    run_dir, out_dir = tmp_path / "run", tmp_path / "replay"
    _original_run(run_dir)
    with pytest.raises(ValueError, match="unknown key"):
        replay_scene(layout.pre_fusion_checkpoint(run_dir, 100), out_dir, fusion={"nope": 1}, device="cpu")
    assert not out_dir.exists()


def test_the_checkpoint_is_not_changed(tmp_path):
    run_dir = tmp_path / "run"
    _original_run(run_dir)
    path = layout.pre_fusion_checkpoint(run_dir, 100)
    before = path.read_bytes()
    replay_scene(path, tmp_path / "replay", device="cpu")
    assert path.read_bytes() == before


# replay_experiment


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Replica/exp with two scenes: office0 fuses at 100 and 200 (all), room0 at 150 (last), and
    a scene without checkpoints."""
    monkeypatch.setattr(layout, "OUTPUT_ROOT", tmp_path)
    exp = layout.ExperimentLayout.named("Replica", "exp")

    ovo, map_data, slam = scene()
    office = checkpoint.Checkpointer("all", exp.scene(SCENE_NAME))
    office.save(100, slam, ovo, config())
    ovo.update_map(map_data)              # 10 absorbs 20
    office.save(200, slam, ovo, config())  # nothing left to merge

    ovo, map_data, slam = scene()
    room = checkpoint.Checkpointer("last", exp.scene("room0"))
    room.save(90, slam, ovo, config())
    room.save(150, slam, ovo, config())

    exp.scene("room1").mkdir()
    (exp.root / "instance_pred").mkdir()
    return exp


def _accepted(path):
    return [(r["first"], r["second"]) for r in _rows(path) if r["accepted"] == "True"]


def test_default_scenes_are_those_with_checkpoints_from_their_last_one(experiment):
    out = replay_experiment("Replica", "exp", "same", device="cpu")

    assert out == experiment.replay("same")
    assert out.root == experiment.root / "replays" / "same"
    assert sorted(p.name for p in out.root.iterdir()) == [SCENE_NAME, "room0"]
    office = yaml.safe_load((out.scene(SCENE_NAME) / "config.yaml").read_text())
    room = yaml.safe_load((out.scene("room0") / "config.yaml").read_text())
    assert office["replayed_from"]["frame_id"] == 200
    assert office["replayed_from"]["checkpoint"] == str(layout.pre_fusion_checkpoint(experiment.scene(SCENE_NAME), 200))
    assert room["replayed_from"]["frame_id"] == 150
    # the state at 200: 20 was absorbed by the fusion at 100
    assert all("20" not in (r["first"], r["second"]) for r in _rows(layout.fusion_decisions(out.scene(SCENE_NAME))))
    assert _accepted(layout.fusion_decisions(out.scene("room0"))) == [("10", "20")]


def test_a_second_replay_skips_the_replays_folder(experiment):
    replay_experiment("Replica", "exp", "first", device="cpu")
    out = replay_experiment("Replica", "exp", "second", device="cpu")
    assert sorted(p.name for p in out.root.iterdir()) == [SCENE_NAME, "room0"]


def test_frame_picks_that_checkpoint(experiment):
    out = replay_experiment("Replica", "exp", "at100", scenes=[SCENE_NAME], frame=100, device="cpu")
    cfg = yaml.safe_load((out.scene(SCENE_NAME) / "config.yaml").read_text())
    assert cfg["replayed_from"]["frame_id"] == 100
    assert _accepted(layout.fusion_decisions(out.scene(SCENE_NAME))) == [("10", "20")]
    assert sorted(p.name for p in out.root.iterdir()) == [SCENE_NAME]


@pytest.mark.parametrize("kwargs, match", [
    (dict(frame=100), "exactly one scene"),
    (dict(scenes=[SCENE_NAME, "room0"], frame=100), "exactly one scene"),
    (dict(scenes=[SCENE_NAME], frame=150), r"available: \[100, 200\]"),
    (dict(scenes=["room1"]), "no pre-fusion checkpoints"),
    (dict(scenes=["missing"]), "no pre-fusion checkpoints"),
    (dict(scenes=[SCENE_NAME, "room1"]), "no pre-fusion checkpoints"),  # fails before office0 is written
    (dict(fusion={"nope": 1}), "unknown key"),
    (dict(fusion={"accept": {"nope": 1}}), "unknown key"),
])
def test_errors_write_nothing(experiment, kwargs, match):
    with pytest.raises(ValueError, match=match):
        replay_experiment("Replica", "exp", "bad", device="cpu", **kwargs)
    assert not (experiment.root / "replays").exists()


def test_an_experiment_without_checkpoints_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "OUTPUT_ROOT", tmp_path)
    (tmp_path / "Replica" / "exp" / SCENE_NAME).mkdir(parents=True)
    with pytest.raises(ValueError, match="no scene has pre-fusion checkpoints"):
        replay_experiment("Replica", "exp", "x", device="cpu")
