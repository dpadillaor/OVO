"""Accept rule (ovo.entities.fusion.rule): semantic check, strong and weak branches, strict borders."""

import pytest

from ovo.entities.fusion.rule import AcceptCfg, accept, similar_enough

CFG = AcceptCfg()


def is_same(cos_sim, overlap, cfg=CFG):
    """The full decision as the pipeline takes it: semantics first, then the overlap branches."""
    return similar_enough(cos_sim, cfg) and accept(overlap, cos_sim, cfg) is not None


def test_defaults_are_the_historical_thresholds():
    assert CFG == AcceptCfg(min_cos=0.81, strong=0.5, weak=0.2, weak_min_cos=0.9)


@pytest.mark.parametrize("cos_sim, expected", [
    (0.80, False),
    (0.81, True),       # exactly min_cos passes: only strictly below is rejected
    (0.95, True),
    (-1.0, False),
])
def test_similar_enough(cos_sim, expected):
    assert similar_enough(cos_sim, CFG) is expected


@pytest.mark.parametrize("cos_sim, overlap, mode", [
    (0.85, 0.51, "strong"),
    (0.85, 0.5, None),      # overlap exactly strong does not accept
    (0.85, 0.3, None),      # weak overlap, but cos_sim not above weak_min_cos
    (0.9, 0.3, None),       # cos_sim exactly weak_min_cos does not take the weak branch
    (0.95, 0.3, "weak"),
    (0.95, 0.2, None),      # overlap exactly weak does not accept
    (0.95, 0.0, None),
    (0.95, 0.6, "both"),
    (0.9, 0.6, "strong"),
])
def test_accept_branches_and_borders(cos_sim, overlap, mode):
    assert accept(overlap, cos_sim, CFG) == mode


@pytest.mark.parametrize("cos_sim, overlap, expected", [
    (0.80, 1.0, False),     # below min_cos never fuses, whatever the overlap
    (0.81, 0.6, True),
    (0.81, 0.5, False),
    (0.95, 0.3, True),
    (0.95, 0.2, False),
])
def test_full_decision(cos_sim, overlap, expected):
    assert is_same(cos_sim, overlap) is expected


def test_custom_thresholds_are_used():
    loose = AcceptCfg(min_cos=0.7, strong=0.3, weak=0.1, weak_min_cos=0.8)
    assert is_same(0.75, 0.31, loose) and not is_same(0.75, 0.31, CFG)
    assert accept(0.15, 0.85, loose) == "weak" and accept(0.15, 0.85, CFG) is None
    assert not similar_enough(0.69, loose)
