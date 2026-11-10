"""ovo.eval.fusion.run and ``python -m ovo eval fusion`` on a tiny fake experiment.

The scene of test_fusion_impact.py, on a mesh whose vertices lie on a line, one map point per
vertex (so the projection gives back the point labels), plus instance 5, whose two points are far
from every vertex. Fusion of frame 100: 1~2 accepted (a chair in two halves), 1~3 rejected on
cos_sim (chair vs table), 3~4 rejected on overlap (4 is on void), 1~5 rejected on cos_sim.

Instances are born at CREATED; with a simulation.json of a jump at 30 and a correction at 50 their
epochs are 1: e0, 2: e1, 3: e0, 4: e1, 5: e2 (6 is not judged).
"""

import csv
import json

import numpy as np
import open3d as o3d
import pytest
import torch
import yaml

from ovo import layout
from ovo.__main__ import main
from ovo.eval.benchmarks import REPLICA
from ovo.eval.fusion.run import evaluate_experiment

CHAIR = REPLICA.class_ids[REPLICA.class_names.index("chair")] * 1000 + 1
TABLE = REPLICA.class_ids[REPLICA.class_names.index("table")] * 1000 + 1
V = 500
SCENE = "office0"
HEADER = "frame_id,first,second,accepted,reason,mode,cos_sim,overlap,aabb,centroid\n"
FRAME_100 = """100,1,2,True,overlap,strong,0.93,0.71,,
100,1,3,False,cos_sim,,0.40,,,
100,3,4,False,overlap,,0.90,0.05,,
100,1,5,False,cos_sim,,0.10,,,
"""
CREATED = {1: 0, 2: 40, 3: 10, 4: 35, 5: 55, 6: 0}


def _gt_and_mesh():
    gt = np.zeros(V, dtype=np.int64)
    gt[0:200], gt[200:400] = CHAIR, TABLE
    gt_dir = layout.instance_gt_dir("Replica")
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(gt_dir / f"{SCENE}.txt", gt, fmt="%d")
    vertices = np.stack([np.arange(V, dtype=float), np.zeros(V), np.zeros(V)], axis=1)
    o3d.io.write_point_cloud(str(layout.replica_mesh(SCENE)), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices)))
    return vertices


def _run_scene(scene_dir, vertices):
    labels = np.repeat([1, 2, 3, 4], [120, 80, 200, 100])
    xyz = np.concatenate([vertices, [[1e4, 0, 0], [1e4, 1, 0]]])
    obj_ids = np.concatenate([labels, [5, 5]])
    path = layout.pre_fusion_checkpoint(scene_dir, 100)
    path.parent.mkdir(parents=True)
    torch.save({  # the format of ovo.replay.checkpoint, with only what the evaluation reads
        "format": 2,
        "frame_id": 100,
        "config": {"semantic": {"instance_fusion": {"apply": "greedy"}}},
        "map": {"xyz": torch.tensor(xyz), "obj_ids": torch.tensor(obj_ids).unsqueeze(-1)},
        "ovo": {"objects": {i: {"created_at_frame": f} for i, f in CREATED.items()}},  # 6: removed from the map
    }, path)
    layout.fusion_decisions(scene_dir).write_text(HEADER + "50,1,3,False,cos_sim,,0.2,,,\n" + FRAME_100)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = layout.ExperimentLayout.named("Replica", "exp")
    _run_scene(exp.scene(SCENE), _gt_and_mesh())
    return exp


def _simulation(scene_dir, jumps, corrections):
    """simulation.json as SimulatedSLAM writes it (only `frame` matters here)."""
    layout.simulation(scene_dir).write_text(json.dumps({
        "jumps": [{"frame": f, "at_frame": f, "axes": "camera"} for f in jumps],
        "corrections": [{"frame": f, "keyframes_moved": 1} for f in corrections]}))


def _set_created(scene_dir, frames, frame_id=100):
    path = layout.pre_fusion_checkpoint(scene_dir, frame_id)
    ckpt = torch.load(path, weights_only=False)
    for i, f in frames.items():
        ckpt["ovo"]["objects"][i]["created_at_frame"] = f
    torch.save(ckpt, path)


def _summary(scene_dir, frame_id=100):
    return json.loads(layout.fusion_eval_summary(scene_dir, frame_id).read_text())


def _csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_outputs(experiment, capsys):
    main(["eval", "fusion", "--dataset", "Replica", "--experiment", "exp"])
    scene_dir = experiment.scene(SCENE)
    assert f"Written to {layout.fusion_eval_dir(scene_dir, 100)}" in capsys.readouterr().out

    rows = _csv(layout.fusion_eval_decisions(scene_dir, 100))
    assert list(rows[0]) == HEADER.strip().split(",") + [
        "gt_first", "gt_second", "same_object", "verdict", "epoch_first", "epoch_second"]
    assert [r["frame_id"] for r in rows] == ["100"] * 4
    assert rows[0]["cos_sim"] == "0.93"  # passed through as written
    assert [(r["gt_first"], r["gt_second"], r["same_object"], r["verdict"]) for r in rows] == [
        (str(CHAIR), str(CHAIR), "True", "TP"),
        (str(CHAIR), str(TABLE), "False", "TN"),
        (str(TABLE), "", "False", "TN"),
        (str(CHAIR), "", "", ""),  # 5 has no mask
    ]
    assert [(r["epoch_first"], r["epoch_second"]) for r in rows] == [("", "")] * 4  # no simulation.json

    s = json.loads(layout.fusion_eval_summary(scene_dir, 100).read_text())
    assert (s["format"], s["scene"], s["frame_id"], s["last_fusion"]) == (2, SCENE, 100, True)
    assert list(s) == ["format", "scene", "frame_id", "last_fusion", "fusion", "instances", "pairs", "verdicts",
                       "cascade", "by_epoch", "agnostic_ap"]
    assert s["by_epoch"] is None
    assert s["fusion"] == {"apply": "greedy"}
    assert s["instances"] == {"pre": 5, "post": 4, "merges": 1}
    assert s["pairs"] == {"judged": 4, "scored": 3, "unscored": 1, "unscored_ids": [5]}
    assert s["verdicts"] == {"TP": 1, "FP": 0, "FN": 0, "TN": 2, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert [(c["step"], c["in"], c["pass"]) for c in s["cascade"]] == [("cos_sim", 3, 2), ("overlap", 2, 1)]
    assert s["cascade"][1]["modes"] == {"strong": {"pass": 1, "TP": 1, "FP": 0}}
    ap = s["agnostic_ap"]
    assert ap["post"]["AP"] == 1.0 and ap["pre"]["AP"] == pytest.approx((2 + 7 * 0.375) / 9)

    stats = _csv(layout.fusion_eval_instance_stats(scene_dir, 100))
    assert [(r["gt_id"], r["change"], r["match_50"], r["pred_post"], r["fragments_pre"]) for r in stats] == [
        (str(CHAIR), "improved", "KEPT", "1", "2"), (str(TABLE), "unchanged", "KEPT", "3", "")]


def test_a_replay_is_evaluated_from_its_source_checkpoint(experiment):
    source = experiment.scene(SCENE)
    replay_dir = experiment.replay("chained").scene(SCENE)
    layout.fusion_dir(replay_dir).mkdir(parents=True)
    layout.fusion_decisions(replay_dir).write_text(HEADER + FRAME_100)
    fusion = {"apply": "chained"}
    config = {"semantic": {"instance_fusion": fusion},
              "replayed_from": {"checkpoint": str(layout.pre_fusion_checkpoint(source, 100)), "frame_id": 100}}
    (replay_dir / "config.yaml").write_text(yaml.dump(config))

    [out] = evaluate_experiment("Replica", "exp/replays/chained")

    assert out == layout.fusion_eval_dir(replay_dir, 100)
    s = json.loads(layout.fusion_eval_summary(replay_dir, 100).read_text())
    assert s["fusion"] == fusion
    assert s["instances"]["merges"] == 1
    assert not layout.fusion_eval_dir(source, 100).exists()


def test_a_replay_of_an_earlier_fusion_is_not_the_last(experiment):
    """The replay's own log only holds the replayed frame: last_fusion comes from the source run."""
    source = experiment.scene(SCENE)
    old = layout.pre_fusion_checkpoint(source, 100)
    torch.save({**torch.load(old, weights_only=False), "frame_id": 50}, layout.pre_fusion_checkpoint(source, 50))
    replay_dir = experiment.replay("early").scene(SCENE)
    layout.fusion_dir(replay_dir).mkdir(parents=True)
    layout.fusion_decisions(replay_dir).write_text(HEADER + "50,1,3,False,cos_sim,,0.2,,,\n")
    config = {"semantic": {"instance_fusion": {"apply": "greedy"}},
              "replayed_from": {"checkpoint": str(layout.pre_fusion_checkpoint(source, 50)), "frame_id": 50}}
    (replay_dir / "config.yaml").write_text(yaml.dump(config))

    evaluate_experiment("Replica", "exp/replays/early")

    assert json.loads(layout.fusion_eval_summary(replay_dir, 50).read_text())["last_fusion"] is False


def test_scannet_is_not_supported(experiment):
    with pytest.raises(ValueError, match="Replica only"):
        evaluate_experiment("ScanNet", "exp")


def test_missing_frame_lists_the_available_ones(experiment):
    with pytest.raises(ValueError, match=r"available: \[100\]"):
        evaluate_experiment("Replica", "exp", [SCENE], frame=50)
    assert not layout.fusion_eval_dir(experiment.scene(SCENE), 50).exists()


def test_frame_needs_one_scene(experiment):
    with pytest.raises(ValueError, match="exactly one scene"):
        evaluate_experiment("Replica", "exp", [SCENE, "room0"], frame=100)


def _cascade(*steps):
    """(step, in, pass, TP, FP, FN, TN) of each cascade entry."""
    return [tuple(c[k] for k in ("step", "in", "pass", "TP", "FP", "FN", "TN")) for c in steps]


def test_by_epoch_splits_the_scored_pairs_by_the_epochs_of_their_instances(experiment):
    scene_dir = experiment.scene(SCENE)
    _simulation(scene_dir, jumps=[30], corrections=[50])

    evaluate_experiment("Replica", "exp")

    rows = _csv(layout.fusion_eval_decisions(scene_dir, 100))
    assert [(r["first"], r["second"], r["epoch_first"], r["epoch_second"]) for r in rows] == [
        ("1", "2", "0", "1"), ("1", "3", "0", "0"), ("3", "4", "0", "1"),
        ("1", "5", "0", "2"),  # unscored (5 has no mask): its epochs are still written
    ]
    by_epoch = _summary(scene_dir)["by_epoch"]
    assert list(by_epoch) == ["e0_e0", "e0_e1"]  # 1~5 is unscored: no e0_e2
    assert by_epoch["e0_e1"]["verdicts"] == {"TP": 1, "FP": 0, "FN": 0, "TN": 1,
                                             "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert _cascade(*by_epoch["e0_e1"]["cascade"]) == [("cos_sim", 2, 2, 1, 1, 0, 0), ("overlap", 2, 1, 1, 0, 0, 1)]
    assert by_epoch["e0_e1"]["cascade"][1]["modes"] == {"strong": {"pass": 1, "TP": 1, "FP": 0}}
    assert by_epoch["e0_e0"]["verdicts"] == {"TP": 0, "FP": 0, "FN": 0, "TN": 1,
                                             "precision": None, "recall": None, "f1": None}
    assert _cascade(*by_epoch["e0_e0"]["cascade"]) == [("cos_sim", 1, 0, 0, 0, 0, 1), ("overlap", 0, 0, 0, 0, 0, 0)]
    assert by_epoch["e0_e0"]["cascade"][1]["modes"] == {}


def test_by_epoch_adds_up_to_the_global_verdicts_and_cascade(experiment):
    scene_dir = experiment.scene(SCENE)
    _simulation(scene_dir, jumps=[30], corrections=[50])
    evaluate_experiment("Replica", "exp")
    s = _summary(scene_dir)
    for key in ("TP", "FP", "FN", "TN"):
        assert sum(g["verdicts"][key] for g in s["by_epoch"].values()) == s["verdicts"][key]
    for i, step in enumerate(s["cascade"]):
        for key in ("in", "pass", "reject", "TP", "FP", "FN", "TN"):
            assert sum(g["cascade"][i][key] for g in s["by_epoch"].values()) == step[key], (step["step"], key)
    assert sum(g["cascade"][0]["in"] for g in s["by_epoch"].values()) == s["pairs"]["scored"]


def test_without_events_every_pair_is_e0_e0(experiment):
    scene_dir = experiment.scene(SCENE)
    _simulation(scene_dir, jumps=[], corrections=[])
    evaluate_experiment("Replica", "exp")
    s = _summary(scene_dir)
    assert list(s["by_epoch"]) == ["e0_e0"]
    assert s["by_epoch"]["e0_e0"] == {"verdicts": s["verdicts"], "cascade": s["cascade"]}


def test_by_epoch_is_in_numeric_epoch_order(experiment):
    scene_dir = experiment.scene(SCENE)
    _simulation(scene_dir, jumps=list(range(1, 11)), corrections=[])  # frame f >= 10 is in e10
    _set_created(scene_dir, {1: 2, 2: 10, 3: 3, 4: 3, 5: 0})
    evaluate_experiment("Replica", "exp")
    # (1, 2) e2_e10, (1, 3) e2_e3, (3, 4) e3_e3: as strings e2_e10 would sort before e2_e3
    assert list(_summary(scene_dir)["by_epoch"]) == ["e2_e3", "e2_e10", "e3_e3"]


def test_missing_created_at_frame_with_simulation_json_raises_before_writing(experiment):
    scene_dir = experiment.scene(SCENE)
    _simulation(scene_dir, jumps=[30], corrections=[])
    _set_created(scene_dir, {3: None, 5: None, 6: None})  # 6 is not judged: not listed
    with pytest.raises(ValueError, match=r"instances \[3, 5\] have no created_at_frame"):
        evaluate_experiment("Replica", "exp")
    assert not layout.fusion_eval_dir(scene_dir, 100).exists()


def test_missing_created_at_frame_without_simulation_json_is_not_needed(experiment):
    scene_dir = experiment.scene(SCENE)
    _set_created(scene_dir, {i: None for i in CREATED})
    evaluate_experiment("Replica", "exp")
    assert _summary(scene_dir)["by_epoch"] is None


def _replay(experiment, name):
    source = experiment.scene(SCENE)
    replay_dir = experiment.replay(name).scene(SCENE)
    layout.fusion_dir(replay_dir).mkdir(parents=True)
    layout.fusion_decisions(replay_dir).write_text(HEADER + FRAME_100)
    config = {"semantic": {"instance_fusion": {"apply": "greedy"}},
              "replayed_from": {"checkpoint": str(layout.pre_fusion_checkpoint(source, 100)), "frame_id": 100}}
    (replay_dir / "config.yaml").write_text(yaml.dump(config))
    return source, replay_dir


def test_a_replay_takes_its_epochs_from_the_source_run(experiment):
    source, replay_dir = _replay(experiment, "same")
    _simulation(source, jumps=[30], corrections=[50])
    _simulation(replay_dir, jumps=[], corrections=[])  # not the run's: ignored

    evaluate_experiment("Replica", "exp/replays/same")

    assert list(_summary(replay_dir)["by_epoch"]) == ["e0_e0", "e0_e1"]
    rows = _csv(layout.fusion_eval_decisions(replay_dir, 100))
    assert [(r["epoch_first"], r["epoch_second"]) for r in rows] == [("0", "1"), ("0", "0"), ("0", "1"), ("0", "2")]


def test_a_replay_of_a_run_without_simulation_json_has_no_epochs(experiment):
    _, replay_dir = _replay(experiment, "same")
    _simulation(replay_dir, jumps=[30], corrections=[50])  # not the run's: ignored

    evaluate_experiment("Replica", "exp/replays/same")

    assert _summary(replay_dir)["by_epoch"] is None
    assert {(r["epoch_first"], r["epoch_second"]) for r in _csv(layout.fusion_eval_decisions(replay_dir, 100))} == {("", "")}
