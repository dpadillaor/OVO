"""ovo.eval.fusion.run and ``python -m ovo eval fusion`` on a tiny fake experiment.

The scene of test_fusion_impact.py, on a mesh whose vertices lie on a line, one map point per
vertex (so the projection gives back the point labels), plus instance 5, whose two points are far
from every vertex. Fusion of frame 100: 1~2 accepted (a chair in two halves), 1~3 rejected on
cos_sim (chair vs table), 3~4 rejected on overlap (4 is on void), 1~5 rejected on cos_sim.
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
        "ovo": {"objects": {i: {} for i in (1, 2, 3, 4, 5, 6)}},  # 6: removed from the map
    }, path)
    layout.fusion_decisions(scene_dir).write_text(HEADER + "50,1,3,False,cos_sim,,0.2,,,\n" + FRAME_100)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = layout.ExperimentLayout.named("Replica", "exp")
    _run_scene(exp.scene(SCENE), _gt_and_mesh())
    return exp


def _csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_outputs(experiment, capsys):
    main(["eval", "fusion", "--dataset", "Replica", "--experiment", "exp"])
    scene_dir = experiment.scene(SCENE)
    assert f"Written to {layout.fusion_eval_dir(scene_dir, 100)}" in capsys.readouterr().out

    rows = _csv(layout.fusion_eval_decisions(scene_dir, 100))
    assert [r["frame_id"] for r in rows] == ["100"] * 4
    assert rows[0]["cos_sim"] == "0.93"  # passed through as written
    assert [(r["gt_first"], r["gt_second"], r["same_object"], r["verdict"]) for r in rows] == [
        (str(CHAIR), str(CHAIR), "True", "TP"),
        (str(CHAIR), str(TABLE), "False", "TN"),
        (str(TABLE), "", "False", "TN"),
        (str(CHAIR), "", "", ""),  # 5 has no mask
    ]

    s = json.loads(layout.fusion_eval_summary(scene_dir, 100).read_text())
    assert (s["format"], s["scene"], s["frame_id"], s["last_fusion"]) == (1, SCENE, 100, True)
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
