"""Instance predictions on disk (ovo.eval.predictions): roundtrip, RLE and the txt line format."""

import json
import re

import numpy as np
import pytest

from ovo.eval import predictions
from ovo.eval.predictions import InstancePredictions, rle_decode, rle_encode

NAMES = ("chair", "table", "lamp")


def make_preds(n, n_vertices=37, seed=0):
    rng = np.random.default_rng(seed)
    return InstancePredictions(
        masks=rng.random((n, n_vertices)) < 0.4,
        classes=np.array([(i % 4) - 1 for i in range(n)], dtype=np.int64),  # includes -1
        scores=rng.random(n),
        class_names=NAMES,
    )


@pytest.mark.parametrize("n", [1, 3, 10, 11])
def test_roundtrip(tmp_path, n):
    preds = make_preds(n)
    predictions.write(tmp_path, "office0", preds, "replica")
    back = predictions.read(tmp_path, "office0")
    assert back.masks.dtype == bool
    np.testing.assert_array_equal(back.masks, preds.masks)
    np.testing.assert_array_equal(back.classes, preds.classes)
    assert -1 in back.classes
    np.testing.assert_array_equal(back.scores, [float(f"{s:.4f}") for s in preds.scores])
    assert back.class_names == NAMES


def test_scores_are_stored_with_four_decimals(tmp_path):
    preds = InstancePredictions(np.ones((2, 5), bool), np.array([0, 1]), np.array([0.98765, 1 / 3]), NAMES)
    predictions.write(tmp_path, "s", preds, "replica")
    assert predictions.read(tmp_path, "s").scores.tolist() == [0.9877, 0.3333]


def test_zero_instances(tmp_path):
    empty = InstancePredictions(np.zeros((0, 12), bool), np.zeros(0, np.int64), np.zeros(0), NAMES)
    predictions.write(tmp_path, "room0", empty, "replica")
    assert (tmp_path / "room0.txt").read_text() == ""
    assert list((tmp_path / "predicted_masks").iterdir()) == []
    back = predictions.read(tmp_path, "room0")
    assert len(back.masks) == len(back.classes) == len(back.scores) == 0
    assert back.class_names == NAMES


def test_txt_line_format_is_kept(tmp_path):
    predictions.write(tmp_path, "office0", make_preds(11), "replica")
    lines = (tmp_path / "office0.txt").read_text().splitlines()
    assert len(lines) == 11
    pattern = re.compile(r"\./predicted_masks/office0_(\d+)\.json (-?\d+) (\d+\.\d{4})")
    for i, line in enumerate(lines):
        match = pattern.fullmatch(line)
        assert match, line
        assert int(match.group(1)) == i
        assert (tmp_path / line.split()[0]).is_file()
    assert lines[0].startswith("./predicted_masks/office0_00.json ")
    assert lines[10].startswith("./predicted_masks/office0_10.json ")


def test_class_list_and_benchmark_are_stored(tmp_path):
    predictions.write(tmp_path, "s", make_preds(2), "scannet200")
    meta = json.loads((tmp_path / "s.classes.json").read_text())
    assert meta == {"benchmark": "scannet200", "class_names": list(NAMES)}


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="2 masks, 1 classes, 2 scores"):
        InstancePredictions(np.ones((2, 4), bool), np.array([0]), np.array([0.1, 0.2]), NAMES)


@pytest.mark.parametrize("mask, counts", [
    ([0, 0, 0, 0], ""),
    ([1, 1, 1, 1], "1 4"),
    ([1], "1 1"),
    ([0], ""),
    ([1, 0, 0, 1], "1 1 4 1"),
    ([0, 1, 1, 0, 1], "2 2 5 1"),
])
def test_rle_edge_cases(mask, counts):
    rle = rle_encode(np.array(mask))
    assert rle == {"length": len(mask), "counts": counts}
    np.testing.assert_array_equal(rle_decode(rle), mask)


def test_rle_accepts_bool_and_is_json_serialisable():
    mask = np.random.default_rng(1).random(500) < 0.5
    rle = json.loads(json.dumps(rle_encode(mask)))
    np.testing.assert_array_equal(rle_decode(rle).astype(bool), mask)
