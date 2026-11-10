"""ovo.eval.fusion.epochs: epoch boundaries from simulation.json, an instance's epoch, pair names."""

import json

import pytest

from ovo.eval.fusion import epochs


def _simulation(tmp_path, jumps, corrections):
    """simulation.json as SimulatedSLAM writes it (tests/slam/test_sim_slam.py)."""
    path = tmp_path / "simulation.json"
    path.write_text(json.dumps({
        "jumps": [{"frame": f, "at_frame": f, "axes": "camera", "translation": [0, 0, -0.1], "rotation": [0, 0, 0]}
                  for f in jumps],
        "corrections": [{"frame": f, "keyframes_moved": 3} for f in corrections],
    }))
    return path


def test_no_simulation_json_has_no_epochs(tmp_path):
    assert epochs.boundaries(tmp_path / "simulation.json") is None


def test_a_run_without_events_is_one_epoch(tmp_path):
    bounds = epochs.boundaries(_simulation(tmp_path, [], []))
    assert bounds == []
    assert [epochs.epoch(f, bounds) for f in (0, 500, 10_000)] == [0, 0, 0]


def test_jumps_and_corrections_are_merged_sorted_and_deduplicated(tmp_path):
    # a jump and a correction at frame 50 start one epoch, not two
    assert epochs.boundaries(_simulation(tmp_path, [70, 30, 50], [90, 50])) == [30, 50, 70, 90]


def test_corrections_alone_are_boundaries(tmp_path):
    assert epochs.boundaries(_simulation(tmp_path, [], [120, 40])) == [40, 120]


@pytest.mark.parametrize("frame, expected", [
    (0, 0), (29, 0),
    (30, 1),            # born at the jump's frame: already after it
    (49, 1), (50, 2), (69, 2), (70, 3), (10_000, 3),
])
def test_epoch_of_a_frame(frame, expected):
    assert epochs.epoch(frame, [30, 50, 70]) == expected


def test_epoch_without_boundaries_is_zero():
    assert epochs.epoch(123, []) == 0


@pytest.mark.parametrize("a, b, name", [
    (0, 1, "e0_e1"), (1, 0, "e0_e1"), (2, 2, "e2_e2"), (10, 2, "e2_e10"), (0, 12, "e0_e12"),
])
def test_pair_names_lowest_epoch_first(a, b, name):
    assert epochs.pair(a, b) == name
