"""Fusion timing (ovo.entities.fusion.timing): preparing, the log, and where the pipeline puts
each step's time.

Timing values are never asserted beyond structure and inequalities, except under a fake clock
(time.perf_counter replaced by a counter the fake work advances), where attribution is exact.
"""

import csv

import pytest
import torch

from fakes import FakeInstance, feature, line, random_scene
from ovo.entities.fusion import FusionCfg, FusionPipeline, TimingLog
from ovo.entities.fusion import timing
from ovo.entities.fusion.timing import COLUMNS, StepTime, preparing

GATE_CONFIGS = {
    "none": [],
    "centroid": [{"centroid": {"max_dist": 0.3}}],
    "aabb": [{"aabb": {"max_dist": 0.05}}],
    "both": [{"aabb": {"max_dist": 0.05}}, {"centroid": {"max_dist": 0.3}}],
}


class Clock:
    """A fake time.perf_counter: time only passes when `advance` is called."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(timing.time, "perf_counter", c)  # the `time` module both timing and pipeline use
    return c


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- preparing

def test_preparing_adds_the_block_time_and_accumulates(clock):
    step = StepTime()
    with preparing(step, sync=False):
        clock.advance(2.0)
    assert step.prepare_s == 2.0
    clock.advance(100.0)  # outside any block: not counted
    with preparing(step, sync=False):
        clock.advance(0.5)
    assert step.prepare_s == 2.5
    assert step.pairs_s is None  # untouched


def test_preparing_with_a_real_clock_is_non_negative():
    step = StepTime()
    with preparing(step, sync=False):
        sum(range(1000))
    assert step.prepare_s >= 0.0


def test_sync_waits_for_the_gpu_before_and_after_each_block(monkeypatch, clock):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(clock.now))
    step = StepTime()
    with preparing(step, sync=True):
        assert len(calls) == 1  # before the clock starts
        clock.advance(1.0)
    assert calls == [0.0, 1.0]  # the second one before the clock stops
    with preparing(step, sync=True):
        clock.advance(1.0)
    assert len(calls) == 4
    assert step.prepare_s == 2.0


def test_no_sync_never_touches_the_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(1))
    with preparing(StepTime(), sync=False):
        pass
    assert calls == []


# ---------------------------------------------------------------- TimingLog

def test_log_removes_an_earlier_run_file(tmp_path):
    path = tmp_path / "fusion" / "fusion_timings.csv"
    path.parent.mkdir()
    path.write_text("stale\n")
    TimingLog(path)
    assert not path.exists()


def test_log_writes_the_header_once_and_one_row_per_step_and_fusion(tmp_path):
    path = tmp_path / "fusion" / "fusion_timings.csv"  # the directory does not exist yet
    log = TimingLog(path)
    log.write(10, {"extract": StepTime(1.5), "cos_sim": StepTime(0.25, 0.125), "apply": StepTime(0.5)})
    log.write(20, {"extract": StepTime(2.0), "cos_sim": StepTime(0.0, 0.0), "apply": StepTime(0.0)})

    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(COLUMNS) == "frame_id,step,prepare_s,pairs_s"
    assert lines.count(lines[0]) == 1
    assert lines[1:] == ["10,extract,1.5,", "10,cos_sim,0.25,0.125", "10,apply,0.5,",
                         "20,extract,2.0,", "20,cos_sim,0.0,0.0", "20,apply,0.0,"]


def test_log_writes_a_fusion_that_judged_no_pair(tmp_path):
    """Preparing was paid anyway: unlike the decisions log, the row is written."""
    path = tmp_path / "fusion_timings.csv"
    report = FusionPipeline(FusionCfg()).run([FakeInstance(1, feature(1.0, 0.0))], {1: line(0, 3)})
    assert report.decisions == []
    TimingLog(path).write(7, report.steps)
    assert [(r["frame_id"], r["step"], r["pairs_s"]) for r in read(path)] == [
        ("7", "cos_sim", "0.0"), ("7", "overlap", "0.0")]


# ---------------------------------------------------------------- the pipeline's steps

def run(instances, pcds, **cfg):
    return FusionPipeline(FusionCfg.from_dict(cfg)).run(instances, pcds)


@pytest.mark.parametrize("gates", GATE_CONFIGS)
@pytest.mark.parametrize("method", ["voxel", "point"])
def test_steps_are_the_gates_in_config_order_then_cos_sim_then_overlap(gates, method):
    expected = {"none": [], "centroid": ["centroid"], "aabb": ["aabb"], "both": ["aabb", "centroid"]}[gates]
    for seed in range(5):
        instances, pcds = random_scene(seed)
        pipe = FusionPipeline(FusionCfg.from_dict({"gates": GATE_CONFIGS[gates], "overlap": {"method": method}}))
        report = pipe.run(instances, pcds)
        assert list(report.steps) == pipe.steps == expected + ["cos_sim", "overlap"]
        for name, t in report.steps.items():
            assert t.prepare_s >= 0.0 and t.pairs_s >= 0.0, name
        spent = sum(t.prepare_s + t.pairs_s for t in report.steps.values())
        assert spent <= report.seconds + 1e-9  # disjoint parts of the run (1e-9: float rounding)


def test_a_step_no_pair_reaches_has_no_pair_time():
    same = feature(1.0, 0.0)
    instances = [FakeInstance(1, same), FakeInstance(2, same)]
    pcds = {1: line(0, 100), 2: line(0, 2)}  # same feature, 2 inside 1, centroids 2.45 m apart
    report = run(instances, pcds, gates=[{"centroid": {"max_dist": 1.0}}])
    assert [d.reason for d in report.decisions] == ["centroid"]
    assert report.steps["cos_sim"].pairs_s == 0.0
    assert report.steps["overlap"].pairs_s == 0.0


def test_no_candidate_pair_leaves_every_pair_time_at_zero():
    instances = [FakeInstance(1, feature(1.0, 0.0)), FakeInstance(2, feature(1.0, 0.0))]
    report = run(instances, {1: line(0, 3), 2: line(50, 53)})  # no shared voxel
    assert report.decisions == []
    assert all(t.pairs_s == 0.0 for t in report.steps.values())


def _scene():
    """The scene of test_pipeline.py: 1~2 accepted, 1~3 cos_sim, 1~4 overlap, 5 far away."""
    same, other = feature(1.0, 0.0), feature(0.0, 1.0)
    instances = [FakeInstance(1, same), FakeInstance(2, same), FakeInstance(3, other),
                 FakeInstance(4, same), FakeInstance(5, same)]
    pcds = {1: line(0, 10), 2: line(2, 6), 3: line(5, 9), 4: line(9, 30), 5: line(100, 110)}
    return instances, pcds


class Timed(FakeInstance):
    """Reading the descriptor costs 4 s of fake time."""

    def __init__(self, inst, clock):
        super().__init__(inst.id, None)
        self._feature, self._clock = inst.clip_feature, clock

    @property
    def clip_feature(self):
        self._clock.advance(4.0)
        return self._feature

    @clip_feature.setter
    def clip_feature(self, value):
        pass


def _costly(obj, name, clock, seconds):
    original = getattr(obj, name)

    def wrapped(*args, **kwargs):
        clock.advance(seconds)
        return original(*args, **kwargs)
    setattr(obj, name, wrapped)


def test_each_cost_is_attributed_to_its_step(monkeypatch, clock):
    """Under a fake clock each piece of work costs a distinct power of two, so every step's time
    is an exact count of the work it did, and nothing is lost or counted twice."""
    instances, pcds = _scene()
    instances = [Timed(inst, clock) for inst in instances]
    pipe = FusionPipeline(FusionCfg.from_dict({"overlap": {"method": "point"},
                                               "gates": [{"centroid": {"max_dist": 1.0}}]}))
    (gate,) = pipe.gates
    _costly(gate, "prepare", clock, 1.0)
    _costly(gate, "check", clock, 2.0)
    cos = torch.nn.functional.cosine_similarity
    monkeypatch.setattr(torch.nn.functional, "cosine_similarity",
                        lambda *a, **k: clock.advance(8.0) or cos(*a, **k))
    _costly(pipe.overlap, "prepare", clock, 16.0)
    _costly(pipe.overlap, "candidates", clock, 32.0)
    _costly(pipe.overlap, "measure", clock, 64.0)

    report = pipe.run(instances, pcds)

    judged = len(report.decisions)
    passed_gate = sum(d.reason != "centroid" for d in report.decisions)
    measured = sum(d.overlap is not None for d in report.decisions)
    assert judged > passed_gate > measured > 0  # every step stops some pair
    assert report.steps["centroid"].prepare_s == 1.0
    assert report.steps["centroid"].pairs_s == 2.0 * judged
    assert report.steps["cos_sim"].prepare_s == 4.0 * len(instances)
    assert report.steps["cos_sim"].pairs_s == 8.0 * passed_gate
    assert report.steps["overlap"].prepare_s == 16.0 + 32.0
    assert report.steps["overlap"].pairs_s == 64.0 * measured
    assert sum(t.prepare_s + t.pairs_s for t in report.steps.values()) == report.seconds


def test_pair_time_goes_only_to_the_steps_a_pair_reached(monkeypatch, clock):
    """Every pair is vetoed by the gate: the gate's check is the only per-pair work timed."""
    instances, pcds = _scene()
    pipe = FusionPipeline(FusionCfg.from_dict({"overlap": {"method": "point"},
                                               "gates": [{"centroid": {"max_dist": 0.0}}]}))
    _costly(pipe.gates[0], "check", clock, 2.0)
    report = pipe.run(instances, pcds)
    assert {d.reason for d in report.decisions} == {"centroid"}
    assert report.steps["centroid"].pairs_s == 2.0 * len(report.decisions) == 2.0 * 10  # point: all pairs
    assert report.steps["cos_sim"].pairs_s == 0.0
    assert report.steps["overlap"].pairs_s == 0.0
