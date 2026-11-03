"""Experiment instance AP (ovo.eval.instances): file checks and the instance_ap.json it writes."""

import json
from pathlib import Path

import numpy as np
import pytest

from ovo.eval import instances, predictions
from ovo.eval.benchmarks import REPLICA, InstanceBenchmark
from ovo.layout import ExperimentLayout

N = 600
CHAIR = REPLICA.class_names.index("chair")  # head
BOOK = REPLICA.class_names.index("book")  # tail


def write_scene(layout, gt_dir, scene, found=True, class_names=REPLICA.class_names):
    """One chair GT instance on vertices [0, 200), predicted perfectly or not at all."""
    gt_ids = np.zeros(N, dtype=np.int64)
    gt_ids[:200] = REPLICA.class_ids[CHAIR] * 1000
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(gt_dir / f"{scene}.txt", gt_ids, fmt="%d")
    masks = np.zeros((1 if found else 0, N), dtype=bool)
    masks[:, :200] = True
    preds = predictions.InstancePredictions(
        masks=masks, classes=np.full(len(masks), CHAIR), scores=np.full(len(masks), 0.9), class_names=class_names
    )
    predictions.write(layout.instance_pred, scene, preds, "replica")


@pytest.fixture
def layout(tmp_path):
    return ExperimentLayout(tmp_path / "exp")


@pytest.fixture
def gt_dir(tmp_path):
    return tmp_path / "gt"


def test_writes_global_and_per_scene_metrics(layout, gt_dir):
    write_scene(layout, gt_dir, "office0", found=True)
    write_scene(layout, gt_dir, "room0", found=False)
    result = instances.evaluate_experiment(layout, ["office0", "room0"], REPLICA, gt_dir)

    on_disk = json.loads(layout.instance_ap.read_text())
    assert set(on_disk) == {"benchmark", "all", "scenes"}
    assert on_disk["benchmark"] == "replica"
    assert set(on_disk["scenes"]) == {"office0", "room0"}
    assert set(on_disk["all"]) == {"agnostic", "class"}
    assert set(on_disk["all"]["agnostic"]) == {"AP", "AP_50", "AP_25"}
    assert set(on_disk["all"]["class"]) == {"AP", "AP_50", "AP_25", "AP_head", "AP_common", "AP_tail"}

    assert on_disk["scenes"]["office0"]["class"]["AP"] == 1.0
    assert on_disk["scenes"]["room0"]["class"]["AP"] == 0.0
    assert 0.0 < on_disk["all"]["class"]["AP"] < 1.0
    assert result["scenes"]["office0"]["agnostic"]["AP"] == 1.0


def test_nan_is_written_as_null(layout, gt_dir):
    write_scene(layout, gt_dir, "office0")
    result = instances.evaluate_experiment(layout, ["office0"], REPLICA, gt_dir)
    text = layout.instance_ap.read_text()
    assert "NaN" not in text
    assert np.isnan(result["all"]["class"]["AP_tail"])  # no tail GT: undefined
    assert json.loads(text)["all"]["class"]["AP_tail"] is None
    assert json.loads(text)["all"]["class"]["AP_head"] == 1.0


def test_zero_predictions_scene(layout, gt_dir):
    write_scene(layout, gt_dir, "office0", found=False)
    result = instances.evaluate_experiment(layout, ["office0"], REPLICA, gt_dir)
    assert result["all"]["class"]["AP"] == 0.0


def test_missing_gt_raises(layout, gt_dir):
    write_scene(layout, gt_dir, "office0")
    (gt_dir / "office0.txt").unlink()
    with pytest.raises(FileNotFoundError, match="missing instance GT for office0"):
        instances.evaluate_experiment(layout, ["office0"], REPLICA, gt_dir)
    assert not layout.instance_ap.exists()


def test_missing_predictions_raise(layout, gt_dir):
    write_scene(layout, gt_dir, "office0")
    write_scene(layout, gt_dir, "office1")
    (layout.instance_pred / "office1.classes.json").unlink()
    with pytest.raises(FileNotFoundError, match="no instance predictions for office1"):
        instances.evaluate_experiment(layout, ["office0", "office1"], REPLICA, gt_dir)


def test_class_list_mismatch_raises(layout, gt_dir):
    write_scene(layout, gt_dir, "office0", class_names=REPLICA.class_names[::-1])
    with pytest.raises(ValueError, match="different class list"):
        instances.evaluate_experiment(layout, ["office0"], REPLICA, gt_dir)


def test_class_list_of_another_benchmark_raises(layout, gt_dir):
    other = InstanceBenchmark("toy", REPLICA.class_ids, REPLICA.class_names[:-1] + ("other",), (), (), ())
    write_scene(layout, gt_dir, "office0")
    with pytest.raises(ValueError, match="toy benchmark"):
        instances.evaluate_experiment(layout, ["office0"], other, gt_dir)


def test_empty_scene_list_raises(layout, gt_dir):
    with pytest.raises(ValueError, match="no scenes"):
        instances.evaluate_experiment(layout, [], REPLICA, gt_dir)


def test_nan_to_none_recurses_and_keeps_other_values():
    value = {"a": float("nan"), "b": {"c": float("nan"), "d": 0.5}, "e": "x", "f": 0}
    assert instances.nan_to_none(value) == {"a": None, "b": {"c": None, "d": 0.5}, "e": "x", "f": 0}


def test_layout_paths():
    exp = ExperimentLayout.named("Replica", "my_exp")
    assert exp.root == Path("data/output/Replica/my_exp")
    assert exp.scene("office0") == exp.root / "office0"
    assert exp.instance_pred == exp.root / "instance_pred"
    assert exp.instance_ap == exp.root / "instance_ap.json"
