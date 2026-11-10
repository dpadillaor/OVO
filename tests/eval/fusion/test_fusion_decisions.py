"""ovo.eval.fusion.decisions: rows of one frame, and the merges the apply policy made from them."""

import pytest

from ovo.eval.fusion import decisions

CSV = """frame_id,first,second,accepted,reason,mode,cos_sim,overlap,aabb,centroid
50,1,2,True,overlap,strong,0.93,0.71,,
120,2,3,True,overlap,weak,0.95,0.30,,
120,1,2,True,overlap,both,0.97,0.80,,
120,1,4,False,cos_sim,,0.62,,,
"""


def test_rows_of_one_frame_keep_their_cells():
    table = decisions.parse(CSV)
    assert table.columns[-2:] == ("aabb", "centroid")
    assert table.frames() == [50, 120]
    rows = table.at_frame(120)
    assert [(r.first, r.second, r.accepted, r.reason, r.mode) for r in rows] == [
        (2, 3, True, "overlap", "weak"), (1, 2, True, "overlap", "both"), (1, 4, False, "cos_sim", None)]
    assert rows[0].cells["overlap"] == "0.30"  # as written, not re-formatted


def test_greedy_survivor_can_be_absorbed_later():
    rows = decisions.parse(CSV).at_frame(120)
    merges = decisions.merges(rows, "greedy", [1, 2, 3, 4])
    assert merges == [(2, 3), (1, 2)]
    assert decisions.groups(merges) == {1: [1, 2, 3]}


def test_chained_survivor_is_the_earliest_instance():
    rows = decisions.parse(CSV).at_frame(120)
    merges = decisions.merges(rows, "chained", [3, 2, 1, 4])
    assert merges == [(3, 2), (3, 1)]
    assert decisions.groups(merges) == {3: [3, 2, 1]}


def test_a_pair_the_policy_would_skip_means_a_foreign_log():
    """Under greedy, 9 is absorbed by 4, so the pipeline never judges (7, 9)."""
    rows = decisions.parse(CSV.split("\n")[0] + "\n1,4,9,True,overlap,strong,,,,\n1,7,9,False,cos_sim,,,,,\n").rows
    with pytest.raises(ValueError, match="would not have been judged"):
        decisions.merges(rows, "greedy", [4, 7, 9])
    assert decisions.merges(rows[:1], "greedy", [4, 7, 9]) == [(4, 9)]


def test_an_instance_outside_the_map_means_a_foreign_log():
    rows = decisions.parse(CSV).at_frame(120)
    with pytest.raises(ValueError, match=r"\[4\], not in the pre-fusion map"):
        decisions.merges(rows, "chained", [1, 2, 3])


def test_bad_accepted_cell_raises():
    with pytest.raises(ValueError, match="accepted"):
        decisions.parse(CSV.replace("True", "yes", 1))
