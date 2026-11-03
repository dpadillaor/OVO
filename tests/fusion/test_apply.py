"""Apply policies (ovo.entities.fusion.apply): which pairs are still judged, and the merges.

Both policies are driven the way the pipeline drives them: for each candidate pair in order,
`on_accept` only if `should_judge` said yes and the pair is one the rule accepts.
"""

import pytest

from ovo.entities.fusion.apply import POLICIES, Chained, Greedy

ORDER = [1, 2, 3, 4, 5]


def drive(policy_cls, candidates, accepted, order=ORDER):
    """Run a policy over `candidates`; returns (pairs judged, merges)."""
    policy = policy_cls(order)
    judged = []
    for a, b in candidates:
        if not policy.should_judge(a, b):
            continue
        judged.append((a, b))
        if (a, b) in accepted:
            policy.on_accept(a, b)
    return judged, policy.merges()


def test_registry():
    assert POLICIES == {"greedy": Greedy, "chained": Chained}


# The same accepted pairs, applied by each policy: (candidates, accepted, greedy merges, chained merges)
SCENARIOS = {
    "single": ([(1, 2)], {(1, 2)}, [(1, 2)], [(1, 2)]),
    "chain, A~C rejected": (
        [(1, 2), (1, 3), (2, 3)], {(1, 2), (2, 3)},
        [(1, 2)],                       # greedy: 2 was absorbed, so (2, 3) is never judged
        [(1, 2), (1, 3)],               # chained: 3 joins 1 through 2
    ),
    "star": ([(1, 2), (1, 3), (1, 4)], {(1, 2), (1, 3), (1, 4)},
             [(1, 2), (1, 3), (1, 4)], [(1, 2), (1, 3), (1, 4)]),
    "late group joins earlier one": (
        [(1, 4), (2, 3), (3, 4)], {(2, 3), (3, 4)},
        [(2, 3)],                       # greedy: (3, 4) skipped, 3 is absorbed
        [(2, 3), (2, 4)],
    ),
    "two groups bridged": (
        [(1, 2), (3, 4), (2, 3), (4, 5)], {(1, 2), (3, 4), (2, 3), (4, 5)},
        [(1, 2), (3, 4)],
        [(1, 2), (1, 3), (1, 4), (1, 5)],   # survivor of the bridged group is the earliest, 1
    ),
    "nothing accepted": ([(1, 2), (2, 3)], set(), [], []),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_policies_on_the_same_accepted_pairs(name):
    candidates, accepted, greedy, chained = SCENARIOS[name]
    assert drive(Greedy, candidates, accepted)[1] == greedy
    assert drive(Chained, candidates, accepted)[1] == chained


# ---------------------------------------------------------------- Greedy

def test_greedy_skips_pairs_with_an_absorbed_instance():
    judged, merges = drive(Greedy, [(1, 2), (2, 3), (3, 2), (1, 3)], {(1, 2), (1, 3)})
    assert judged == [(1, 2), (1, 3)]
    assert merges == [(1, 2), (1, 3)]


def test_greedy_keeps_judging_the_survivor():
    policy = Greedy(ORDER)
    policy.on_accept(1, 2)
    assert policy.should_judge(1, 3)
    assert not policy.should_judge(1, 2) and not policy.should_judge(2, 3)


def test_greedy_merges_in_acceptance_order():
    _, merges = drive(Greedy, [(3, 4), (1, 2), (1, 5)], {(3, 4), (1, 2), (1, 5)})
    assert merges == [(3, 4), (1, 2), (1, 5)]


def test_greedy_has_no_chains():
    # 1~2 and 2~3 accepted, 1~3 rejected: 3 stays on its own
    _, merges = drive(Greedy, [(1, 2), (1, 3), (2, 3)], {(1, 2), (2, 3)})
    assert all(absorbed != 3 for _, absorbed in merges)


# ---------------------------------------------------------------- Chained

def test_chained_is_transitive_even_when_a_c_is_rejected():
    judged, merges = drive(Chained, [(1, 2), (1, 3), (2, 3)], {(1, 2), (2, 3)})
    assert (1, 3) in judged            # judged and rejected, then joined anyway through 2
    assert merges == [(1, 2), (1, 3)]


def test_chained_survivor_is_the_earliest_in_order():
    order = [7, 3, 9]                  # rank follows the order, not the ids
    policy = Chained(order)
    policy.on_accept(3, 9)
    policy.on_accept(7, 9)
    assert policy.merges() == [(7, 3), (7, 9)]


def test_chained_does_not_judge_pairs_within_a_group():
    policy = Chained(ORDER)
    policy.on_accept(1, 2)
    policy.on_accept(2, 3)
    assert not policy.should_judge(1, 3)
    assert not policy.should_judge(3, 1)
    assert policy.should_judge(3, 4)


def test_chained_merges_sorted_by_survivor_then_absorbed():
    policy = Chained(ORDER)
    for a, b in [(4, 5), (2, 3), (1, 4)]:
        policy.on_accept(a, b)
    assert policy.merges() == [(1, 4), (1, 5), (2, 3)]


def test_chained_long_chain_collapses_to_first():
    order = list(range(10))
    _, merges = drive(Chained, [(i, i + 1) for i in range(9)], {(i, i + 1) for i in range(9)}, order)
    assert merges == [(0, i) for i in range(1, 10)]
