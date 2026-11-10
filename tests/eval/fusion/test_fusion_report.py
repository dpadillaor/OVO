"""ovo.eval.fusion.report and ``python -m ovo eval report`` on the fake experiment of
test_fusion_run.py.

Fusion of office0 at frame 100: 1~2 accepted (TP), 1~3 rejected on cos_sim (TN), 3~4 on overlap
(TN), 1~5 on cos_sim but unscored (5 has no mask). The step times are written by hand, with values
that add up exactly in binary.
"""

import csv
import shutil

import pytest
import torch

from ovo import layout
from ovo.__main__ import main
from ovo.entities.fusion.timing import StepTime, TimingLog
from ovo.eval.fusion import report
from ovo.eval.fusion.run import evaluate_experiment
from test_fusion_run import HEADER, SCENE, _replay, experiment  # noqa: F401 (fixture)

ROOM = "room0"
ROOM_100 = "100,1,2,False,overlap,,0.93,0.30,,\n100,3,4,False,overlap,,0.90,0.05,,\n"  # 1~2 FN, 3~4 TN
COLS = list(report.COLUMNS)


def times(extract=1.0, cos=(0.25, 0.5), overlap=(0.125, 0.25), apply=0.5, gates=None) -> dict:
    steps = {"extract": StepTime(extract)}
    steps.update({g: StepTime(0.0, 0.0) for g in gates or []})
    steps.update({"cos_sim": StepTime(*cos), "overlap": StepTime(*overlap), "apply": StepTime(apply)})
    return steps


def write_times(scene_dir, by_frame: dict) -> None:
    log = TimingLog(layout.fusion_timings(scene_dir))
    for frame, steps in by_frame.items():
        log.write(frame, steps)


def add_frame_50(experiment):
    """A second checkpointed fusion of office0, at frame 50 (1~3 rejected on cos_sim, TN)."""
    scene_dir = experiment.scene(SCENE)
    old = torch.load(layout.pre_fusion_checkpoint(scene_dir, 100), weights_only=False)
    torch.save({**old, "frame_id": 50}, layout.pre_fusion_checkpoint(scene_dir, 50))


def add_room(experiment):
    """room0: office0's map and GT, with a fusion at 100 that rejects both pairs on overlap."""
    office, room = experiment.scene(SCENE), experiment.scene(ROOM)
    gt = layout.instance_gt_dir("Replica")
    shutil.copy(gt / f"{SCENE}.txt", gt / f"{ROOM}.txt")
    shutil.copy(layout.replica_mesh(SCENE), layout.replica_mesh(ROOM))
    layout.fusion_dir(room).mkdir(parents=True)
    shutil.copy(layout.pre_fusion_checkpoint(office, 100), layout.pre_fusion_checkpoint(room, 100))
    layout.fusion_decisions(room).write_text(HEADER + ROOM_100)


def key(rows):
    return {(r["scene"], r["frame_id"], r["step"]): r for r in rows}


def counts(row):
    return tuple(row.get(k) for k in ("in", "reject", "FN", "TN"))


@pytest.fixture
def evaluated(experiment):  # noqa: F811
    evaluate_experiment("Replica", "exp")
    write_times(experiment.scene(SCENE), {100: times()})
    return experiment


# ---------------------------------------------------------------- one fusion

def test_rows_of_one_fusion(evaluated):
    rows = report.report_experiment("Replica", "exp")
    assert [(r["scene"], r["frame_id"], r["step"]) for r in rows] == [
        (SCENE, 100, "extract"), (SCENE, 100, "cos_sim"), (SCENE, 100, "overlap"), (SCENE, 100, "apply"),
        ("TOTAL", None, "extract"), ("TOTAL", None, "cos_sim"), ("TOTAL", None, "overlap"), ("TOTAL", None, "apply")]
    by = key(rows)
    cos, ovl = by[(SCENE, 100, "cos_sim")], by[(SCENE, 100, "overlap")]
    # cos_sim: all 4 judged pairs in, 1~3 and 1~5 stopped; only 1~3 (TN) is scored
    assert counts(cos) == (4, 2, 0, 1)
    assert counts(ovl) == (2, 1, 0, 1)
    assert (cos["prepare_s"], cos["pairs_s"], cos["ms_pair"]) == (0.25, 0.5, 125.0)
    assert (ovl["prepare_s"], ovl["pairs_s"], ovl["ms_pair"]) == (0.125, 0.25, 125.0)
    for step in ("extract", "apply"):
        row = by[(SCENE, 100, step)]
        assert row["prepare_s"] == {"extract": 1.0, "apply": 0.5}[step]
        assert all(row.get(k) is None for k in ("in", "reject", "FN", "TN", "pairs_s", "ms_pair"))
    for step in ("extract", "cos_sim", "overlap", "apply"):  # one fusion: TOTAL is that fusion
        total = by[("TOTAL", None, step)]
        assert {k: total.get(k) for k in COLS[2:]} == {k: by[(SCENE, 100, step)].get(k) for k in COLS[2:]}


def test_unscored_pairs_cost_time_but_have_no_verdict(evaluated):
    """1~5 reaches cos_sim and is stopped there, but 5 has no mask: in/reject count it, FN/TN not."""
    import json
    summary = json.loads(layout.fusion_eval_summary(evaluated.scene(SCENE), 100).read_text())
    cascade = {c["step"]: c for c in summary["cascade"]}
    cos = key(report.report_experiment("Replica", "exp"))[(SCENE, 100, "cos_sim")]
    assert cos["in"] == cascade["cos_sim"]["in"] + 1 == 4
    assert cos["reject"] == cascade["cos_sim"]["reject"] + 1 == 2
    assert cos["FN"] + cos["TN"] == cascade["cos_sim"]["reject"] == 1


def test_a_step_no_pair_reached_has_no_ms_pair(experiment):  # noqa: F811
    layout.fusion_decisions(experiment.scene(SCENE)).write_text(HEADER + "100,1,3,False,cos_sim,,0.40,,,\n")
    evaluate_experiment("Replica", "exp")
    write_times(experiment.scene(SCENE), {100: times(overlap=(0.125, 0.0))})
    by = key(report.report_experiment("Replica", "exp"))
    assert counts(by[(SCENE, 100, "overlap")]) == (0, 0, 0, 0)
    assert by[(SCENE, 100, "overlap")]["ms_pair"] is None and by[("TOTAL", None, "overlap")]["ms_pair"] is None


# ---------------------------------------------------------------- several fusions

@pytest.fixture
def two_scenes_three_fusions(experiment):  # noqa: F811
    add_frame_50(experiment)
    add_room(experiment)
    evaluate_experiment("Replica", "exp", [SCENE], frame=50)
    evaluate_experiment("Replica", "exp")  # office0 at 100 (its last), room0 at 100
    write_times(experiment.scene(SCENE), {50: times(2.0, (0.5, 0.5), (0.25, 0.0), 0.0), 100: times()})
    write_times(experiment.scene(ROOM), {100: times(4.0, (1.0, 0.0), (0.5, 0.75), 0.25)})
    return experiment


def test_total_sums_every_fusion_and_takes_ms_pair_from_the_sums(two_scenes_three_fusions):
    rows = report.report_experiment("Replica", "exp")
    assert [(r["scene"], r["frame_id"]) for r in rows[::4]] == [(SCENE, 50), (SCENE, 100), (ROOM, 100), ("TOTAL", None)]
    by = key(rows)
    assert counts(by[(SCENE, 50, "cos_sim")]) == (1, 1, 0, 1)
    assert counts(by[(ROOM, 100, "cos_sim")]) == (2, 0, 0, 0)
    assert counts(by[(ROOM, 100, "overlap")]) == (2, 2, 1, 1)

    cos, ovl = by[("TOTAL", None, "cos_sim")], by[("TOTAL", None, "overlap")]
    assert counts(cos) == (1 + 4 + 2, 1 + 2 + 0, 0, 1 + 1 + 0)
    assert counts(ovl) == (0 + 2 + 2, 0 + 1 + 2, 1, 0 + 1 + 1)
    assert (cos["prepare_s"], cos["pairs_s"]) == (0.5 + 0.25 + 1.0, 0.5 + 0.5 + 0.0)
    assert cos["ms_pair"] == pytest.approx(1000 * 1.0 / 7)
    means = [by[(s, f, "cos_sim")]["ms_pair"] for s, f in ((SCENE, 50), (SCENE, 100), (ROOM, 100))]
    assert cos["ms_pair"] != pytest.approx(sum(means) / 3)  # not the mean of the per-fusion values
    assert ovl["ms_pair"] == pytest.approx(1000 * 1.0 / 4)
    assert by[("TOTAL", None, "extract")]["prepare_s"] == 7.0
    assert by[("TOTAL", None, "apply")]["prepare_s"] == 0.75
    assert by[("TOTAL", None, "apply")].get("pairs_s") is None


def test_frame_picks_one_fusion(two_scenes_three_fusions):
    rows = report.report_experiment("Replica", "exp", [SCENE], frame=50)
    assert {(r["scene"], r["frame_id"]) for r in rows} == {(SCENE, 50), ("TOTAL", None)}


def test_scenes_restricts_the_report(two_scenes_three_fusions):
    rows = report.report_experiment("Replica", "exp", [ROOM])
    assert {r["scene"] for r in rows} == {ROOM, "TOTAL"}


def test_default_scenes_are_those_with_an_evaluated_fusion(evaluated):
    add_room(evaluated)  # decisions and a checkpoint, but never evaluated
    (evaluated.root / "instance_pred").mkdir()
    assert {r["scene"] for r in report.report_experiment("Replica", "exp")} == {SCENE, "TOTAL"}


def test_a_replay_is_reported_like_an_experiment(experiment):  # noqa: F811
    _, replay_dir = _replay(experiment, "same")
    evaluate_experiment("Replica", "exp/replays/same")
    write_times(replay_dir, {100: times()})
    rows = report.report_experiment("Replica", "exp/replays/same")
    assert counts(key(rows)[(SCENE, 100, "cos_sim")]) == (4, 2, 0, 1)
    assert not layout.fusion_timings(experiment.scene(SCENE)).exists()


def test_gates_are_steps_of_the_report(experiment):  # noqa: F811
    path = layout.pre_fusion_checkpoint(experiment.scene(SCENE), 100)
    ckpt = torch.load(path, weights_only=False)
    ckpt["config"]["semantic"]["instance_fusion"]["gates"] = [{"centroid": {"max_dist": 9e9}}]
    torch.save(ckpt, path)
    evaluate_experiment("Replica", "exp")
    write_times(experiment.scene(SCENE), {100: times(gates=["centroid"])})
    rows = [r for r in report.report_experiment("Replica", "exp") if r["scene"] == SCENE]
    assert [r["step"] for r in rows] == ["extract", "centroid", "cos_sim", "overlap", "apply"]
    assert counts(rows[1]) == (4, 0, 0, 0) and counts(rows[2]) == (4, 2, 0, 1)


# ---------------------------------------------------------------- errors

def test_no_evaluated_fusion(experiment):  # noqa: F811
    with pytest.raises(ValueError, match="no evaluated fusion"):
        report.report_experiment("Replica", "exp")
    with pytest.raises(ValueError, match="no evaluated fusion"):
        report.report_experiment("Replica", "missing")


def test_missing_timings_file(experiment):  # noqa: F811
    evaluate_experiment("Replica", "exp")
    with pytest.raises(FileNotFoundError, match="no step times"):
        report.report_experiment("Replica", "exp")


def test_no_timings_for_the_frame(experiment):  # noqa: F811
    evaluate_experiment("Replica", "exp")
    write_times(experiment.scene(SCENE), {50: times()})
    with pytest.raises(ValueError, match="no step times for the fusion at frame 100"):
        report.report_experiment("Replica", "exp")


@pytest.mark.parametrize("steps", [
    times(gates=["centroid"]),                                    # another config's steps
    {k: v for k, v in times().items() if k != "apply"},           # one missing
    dict(reversed(list(times().items()))),                        # another order
])
def test_timed_steps_must_be_the_evaluated_ones(experiment, steps):  # noqa: F811
    evaluate_experiment("Replica", "exp")
    write_times(experiment.scene(SCENE), {100: steps})
    with pytest.raises(ValueError, match="do not belong to this fusion"):
        report.report_experiment("Replica", "exp")


@pytest.mark.parametrize("scenes", [None, [SCENE, ROOM]])
def test_frame_needs_exactly_one_scene(evaluated, scenes):
    with pytest.raises(ValueError, match="exactly one scene"):
        report.report_experiment("Replica", "exp", scenes, frame=100)


def test_frame_must_be_evaluated(evaluated):
    with pytest.raises(ValueError, match=r"frame 50 is not evaluated; evaluated: \[100\]"):
        report.report_experiment("Replica", "exp", [SCENE], frame=50)


# ---------------------------------------------------------------- output

def test_csv_has_every_decimal_and_the_table_is_rounded(experiment, tmp_path):  # noqa: F811
    evaluate_experiment("Replica", "exp")
    write_times(experiment.scene(SCENE), {100: times(extract=0.123456789)})
    rows = report.report_experiment("Replica", "exp")
    out = tmp_path / "report.csv"
    report.write_csv(rows, out)

    with open(out, newline="") as f:
        lines = list(csv.reader(f))
    assert lines[0] == COLS == ["scene", "frame_id", "step", "in", "reject", "FN", "TN", "prepare_s", "pairs_s", "ms_pair"]
    assert lines[1] == [SCENE, "100", "extract", "", "", "", "", "0.123456789", "", ""]
    assert lines[2] == [SCENE, "100", "cos_sim", "4", "2", "0", "1", "0.25", "0.5", "125.0"]
    assert lines[5][:3] == ["TOTAL", "", "extract"]
    assert len(lines) == 1 + 8

    table = report.render(rows).splitlines()
    assert table[0].split() == COLS
    assert table[1].split() == [SCENE, "100", "extract", "0.1235"]
    assert table[2].split() == [SCENE, "100", "cos_sim", "4", "2", "0", "1", "0.2500", "0.5000", "125.000"]


def test_cli_prints_the_table_and_writes_the_csv(evaluated, capsys):
    main(["eval", "report", "--dataset", "Replica", "--experiment", "exp", "--out", "r.csv"])
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == COLS
    assert "Written to r.csv" in out
    with open("r.csv", newline="") as f:
        assert [r["step"] for r in csv.DictReader(f)] == ["extract", "cos_sim", "overlap", "apply"] * 2


def test_cli_without_out_writes_nothing(evaluated, capsys, tmp_path):
    main(["eval", "report", "--dataset", "Replica", "--experiment", "exp", "--scenes", SCENE, "--frame", "100"])
    out = capsys.readouterr().out
    assert "Written to" not in out
    assert len(out.splitlines()) == 1 + 8
    assert not list(tmp_path.glob("*.csv"))


@pytest.mark.parametrize("argv", [["eval", "report", "--experiment", "exp"], ["eval", "report", "--dataset", "Replica"],
                                  ["eval", "report", "--dataset", "Replica", "--experiment", "exp", "--frame", "x"]])
def test_cli_incomplete_command_lines_exit(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
