"""Tests for the deferred merge policy (`semantic.deferred_fusion`)."""

from types import SimpleNamespace

from ovo.entities.fusion import collect_merge_pairs, group_merge_pairs


class _RecordingStrategy:
    """Accepts a fixed set of pairs and records every pair it was asked about."""

    def __init__(self, accept):
        self.accept = {frozenset(p) for p in accept}
        self.asked = []

    def candidate_pairs(self):
        return None  # no broadphase -> planner uses the brute-force loop

    def same_instance(self, i1, i2, pc1, pc2):
        self.asked.append((i1.id, i2.id))
        return frozenset((i1.id, i2.id)) in self.accept


def _instances(*ids):
    return [SimpleNamespace(id=i) for i in ids]


def _pcds(objects_list):
    return {ins.id: [None, None] for ins in objects_list}


class TestCollectMergePairs:
    def test_every_pair_is_judged(self):
        objs = _instances(1, 2, 3, 4)
        strategy = _RecordingStrategy(accept=[])
        collect_merge_pairs(objs, _pcds(objs), strategy)
        assert len(strategy.asked) == 6  # 4 choose 2

    def test_an_accepted_instance_keeps_being_compared(self):
        """The eager loop stops comparing an instance once it is merged; this one does not."""
        objs = _instances(1, 2, 3)
        strategy = _RecordingStrategy(accept=[(1, 2)])
        pairs = collect_merge_pairs(objs, _pcds(objs), strategy)
        assert pairs == [(1, 2)]
        assert (2, 3) in strategy.asked

    def test_pairs_come_out_in_list_order(self):
        objs = _instances(7, 3, 9)
        strategy = _RecordingStrategy(accept=[(3, 9), (7, 9)])
        assert collect_merge_pairs(objs, _pcds(objs), strategy) == [(7, 9), (3, 9)]


class TestGroupMergePairs:
    def test_no_pairs_no_groups(self):
        assert group_merge_pairs(_instances(1, 2, 3), []) == []

    def test_survivor_is_the_earliest_in_the_list(self):
        objs = _instances(5, 2, 8)  # id order and list order disagree on purpose
        assert group_merge_pairs(objs, [(5, 8)]) == [(5, [8])]
        assert group_merge_pairs(objs, [(2, 8)]) == [(2, [8])]
        # 5 comes first in the list, so it survives even though 2 is the smaller id
        assert group_merge_pairs(objs, [(2, 5)]) == [(5, [2])]

    def test_merges_are_transitive(self):
        """A-B and B-C put all three together even though A-C was never accepted."""
        objs = _instances(1, 2, 3)
        assert group_merge_pairs(objs, [(1, 2), (2, 3)]) == [(1, [2, 3])]

    def test_independent_groups_stay_apart(self):
        objs = _instances(1, 2, 3, 4)
        assert group_merge_pairs(objs, [(1, 2), (3, 4)]) == [(1, [2]), (3, [4])]

    def test_group_order_follows_the_survivors(self):
        objs = _instances(9, 4, 7, 1)
        groups = group_merge_pairs(objs, [(7, 1), (9, 4)])
        assert [g[0] for g in groups] == [9, 7]

    def test_result_does_not_depend_on_pair_order(self):
        objs = _instances(1, 2, 3, 4)
        a = group_merge_pairs(objs, [(1, 2), (2, 3), (3, 4)])
        b = group_merge_pairs(objs, [(3, 4), (2, 3), (1, 2)])
        assert a == b == [(1, [2, 3, 4])]
