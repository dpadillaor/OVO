"""``python -m ovo eval instances`` (ovo.__main__) on a tiny fake experiment."""

import json

import numpy as np
import pytest

from ovo import layout
from ovo.__main__ import main
from ovo.eval import predictions
from ovo.eval.benchmarks import REPLICA

N = 400
CHAIR = REPLICA.class_names.index("chair")


def fake_scene(scene, found):
    gt_dir = layout.instance_gt_dir("Replica")
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_ids = np.zeros(N, dtype=np.int64)
    gt_ids[:150] = REPLICA.class_ids[CHAIR] * 1000
    np.savetxt(gt_dir / f"{scene}.txt", gt_ids, fmt="%d")
    masks = np.zeros((1, N), dtype=bool)
    masks[0, :150 if found else 0] = True
    preds = predictions.InstancePredictions(masks, np.array([CHAIR]), np.array([0.7]), REPLICA.class_names)
    predictions.write(layout.ExperimentLayout.named("Replica", "exp").instance_pred, scene, preds, "replica")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_scene("office0", found=True)
    fake_scene("room0", found=False)
    return tmp_path / "data/output/Replica/exp/instance_ap.json"


def test_default_scenes_are_those_with_predictions(experiment, capsys):
    main(["eval", "instances", "--dataset", "Replica", "--experiment", "exp"])
    result = json.loads(experiment.read_text())
    assert sorted(result["scenes"]) == ["office0", "room0"]
    assert result["scenes"]["office0"]["class"]["AP"] == 1.0
    assert result["scenes"]["room0"]["class"]["AP"] == 0.0
    out = capsys.readouterr().out
    assert "Written to data/output/Replica/exp/instance_ap.json" in out
    assert json.loads(out[: out.index("Written to")]) == result["all"]


def test_scenes_flag_restricts_the_evaluation(experiment):
    main(["eval", "instances", "--dataset", "Replica", "--experiment", "exp", "--scenes", "office0"])
    result = json.loads(experiment.read_text())
    assert list(result["scenes"]) == ["office0"]
    assert result["all"]["class"]["AP"] == 1.0


def test_unknown_experiment_has_no_scenes(experiment):
    with pytest.raises(ValueError, match="no scenes"):
        main(["eval", "instances", "--dataset", "Replica", "--experiment", "missing"])


def test_unknown_dataset_raises(experiment):
    with pytest.raises(ValueError, match="No instance benchmark"):
        main(["eval", "instances", "--dataset", "ScanNetPP", "--experiment", "exp", "--scenes", "x"])


@pytest.mark.parametrize("argv", [[], ["eval"], ["eval", "instances", "--experiment", "exp"]])
def test_incomplete_command_lines_exit(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
