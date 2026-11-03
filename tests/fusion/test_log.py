"""Fusion decisions log (ovo.entities.fusion.log): rows, file lifecycle, and the real pipeline."""

import csv
from pathlib import Path

import pytest

from fakes import random_scene
from ovo import layout
from ovo.entities.fusion import FusionCfg, FusionLog, FusionPipeline, FusionReport
from ovo.entities.fusion.gates import GATES
from ovo.entities.fusion.log import COLUMNS, decision_row
from ovo.entities.fusion.pipeline import Decision

FLOAT_FIELDS = ("cos_sim", "overlap")


def report(*decisions: Decision) -> FusionReport:
    return FusionReport(list(decisions), [], 0.0)


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def optional_float(cell: str):
    return None if cell == "" else float(cell)


# decision_row


def test_columns_contain_every_gate():
    assert set(GATES) <= set(COLUMNS)
    assert len(COLUMNS) == len(set(COLUMNS))


def test_gate_veto_leaves_later_steps_empty():
    d = Decision(4, 9, reason="centroid", gate_values={"centroid": 2.41})
    row = decision_row(120, d)
    assert row["frame_id"] == 120 and (row["first"], row["second"]) == (4, 9)
    assert row["accepted"] is False and row["reason"] == "centroid"
    assert row["centroid"] == 2.41
    assert row["aabb"] is None
    assert row["cos_sim"] is None and row["overlap"] is None and row["mode"] is None


def test_cos_sim_rejection():
    d = Decision(4, 15, reason="cos_sim", cos_sim=0.62, gate_values={"centroid": 0.3})
    row = decision_row(7, d)
    assert row["accepted"] is False and row["reason"] == "cos_sim"
    assert row["cos_sim"] == 0.62 and row["centroid"] == 0.3
    assert row["overlap"] is None and row["mode"] is None and row["aabb"] is None


def test_overlap_rejection():
    d = Decision(4, 15, reason="overlap", cos_sim=0.85, overlap=0.1)
    row = decision_row(7, d)
    assert row["accepted"] is False and row["reason"] == "overlap"
    assert row["cos_sim"] == 0.85 and row["overlap"] == 0.1 and row["mode"] is None
    assert all(row[gate] is None for gate in GATES)


@pytest.mark.parametrize("mode", ["strong", "weak", "both"])
def test_acceptance(mode):
    d = Decision(4, 9, accepted=True, reason="overlap", mode=mode, cos_sim=0.93, overlap=0.71)
    row = decision_row(120, d)
    assert row["accepted"] is True and row["mode"] == mode
    assert row["cos_sim"] == 0.93 and row["overlap"] == 0.71


def test_unknown_gate_value_raises():
    with pytest.raises(ValueError, match="height"):
        decision_row(0, Decision(1, 2, reason="height", gate_values={"height": 1.0}))


def test_row_keys_are_the_columns():
    assert tuple(decision_row(0, Decision(1, 2))) == COLUMNS


# FusionLog


def test_old_file_removed_even_if_nothing_is_written(tmp_path):
    path = tmp_path / "fusion" / "fusion_decisions.csv"
    path.parent.mkdir()
    path.write_text("stale\n")
    log = FusionLog(path)
    assert not path.exists()
    log.write(3, report())
    assert not path.exists()


def test_no_file_when_every_report_is_empty(tmp_path):
    path = tmp_path / "fusion" / "fusion_decisions.csv"
    log = FusionLog(path)
    for frame_id in range(3):
        log.write(frame_id, report())
    assert not path.exists()
    assert not path.parent.exists()


def test_parent_dir_created(tmp_path):
    path = tmp_path / "scene" / "fusion" / "fusion_decisions.csv"
    FusionLog(path).write(0, report(Decision(1, 2)))
    assert path.is_file()


def test_header_once_and_rows_appended_in_order(tmp_path):
    path = tmp_path / "fusion_decisions.csv"
    log = FusionLog(path)
    log.write(10, report(Decision(1, 2), Decision(1, 3)))
    log.write(20, report())
    log.write(30, report(Decision(5, 6)))
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert sum(line == ",".join(COLUMNS) for line in lines) == 1
    rows = read(path)
    assert [(r["frame_id"], r["first"], r["second"]) for r in rows] == [
        ("10", "1", "2"), ("10", "1", "3"), ("30", "5", "6")]


def test_none_written_as_empty_cell(tmp_path):
    path = tmp_path / "fusion_decisions.csv"
    FusionLog(path).write(0, report(Decision(1, 2, reason="centroid", gate_values={"centroid": 2.0})))
    (row,) = read(path)
    assert row["mode"] == row["cos_sim"] == row["overlap"] == row["aabb"] == ""


def test_floats_round_trip_exactly(tmp_path):
    path = tmp_path / "fusion_decisions.csv"
    values = [0.1 + 0.2, 1 / 3, 0.8100000000000001, 1e-17, float("inf")]
    log = FusionLog(path)
    log.write(0, report(*[
        Decision(i, i + 1, cos_sim=v, overlap=v / 7, gate_values={"centroid": v * 3})
        for i, v in enumerate(values)]))
    for row, v in zip(read(path), values):
        assert float(row["cos_sim"]) == v
        assert float(row["overlap"]) == v / 7
        assert float(row["centroid"]) == v * 3


# With the real pipeline

CONFIGS = {
    "default": {},
    "gates": {"gates": [{"centroid": {"max_dist": 0.15}}, {"aabb": {"max_dist": 0.02}}],
              "overlap": {"method": "point"}},
    "chained": {"apply": "chained"},
}


def assert_row_matches(row: dict, frame_id: int, d: Decision):
    assert int(row["frame_id"]) == frame_id
    assert (int(row["first"]), int(row["second"])) == (d.first, d.second)
    assert row["accepted"] == str(d.accepted)
    assert row["reason"] == d.reason
    assert (row["mode"] or None) == d.mode
    for name in FLOAT_FIELDS:
        assert optional_float(row[name]) == getattr(d, name)
    for gate in GATES:
        assert optional_float(row[gate]) == d.gate_values.get(gate)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", range(8))
def test_pipeline_decisions_round_trip(tmp_path, config, seed):
    instances, pcds = random_scene(seed)
    fusion = FusionPipeline(FusionCfg.from_dict(CONFIGS[config])).run(instances, pcds)
    path = layout.fusion_decisions(tmp_path)
    FusionLog(path).write(seed, fusion)
    if not fusion.decisions:
        assert not path.exists()
        return
    rows = read(path)
    assert len(rows) == len(fusion.decisions)
    for row, d in zip(rows, fusion.decisions):
        assert_row_matches(row, seed, d)


def test_pipeline_scenes_cover_every_outcome():
    """The round trip above is only meaningful if the scenes produce every kind of row."""
    reasons, modes = set(), set()
    for config in CONFIGS.values():
        pipeline = FusionPipeline(FusionCfg.from_dict(config))
        for seed in range(8):
            for d in pipeline.run(*random_scene(seed)).decisions:
                reasons.add(d.reason)
                modes.add(d.mode)
    assert {"centroid", "aabb", "cos_sim", "overlap"} <= reasons
    assert {"strong", "weak", "both", None} <= modes


# Layout


def test_fusion_decisions_path(tmp_path):
    assert layout.fusion_decisions(tmp_path / "office0") == tmp_path / "office0" / "fusion" / "fusion_decisions.csv"
    assert layout.fusion_decisions("scene") == Path("scene/fusion/fusion_decisions.csv")
