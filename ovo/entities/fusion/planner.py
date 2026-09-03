"""Pair selection for classic fusion: which instances end up merged, and into whom.

Two policies share this module. The eager one lives in `ovo.py` because it mutates the map
while it decides. The deferred one is here in full: it judges every pair against untouched
geometry, then resolves the accepted pairs into groups that are applied afterwards.
"""

from typing import Dict, List, Sequence, Tuple


def collect_merge_pairs(objects_list: Sequence, obj_pcds: Dict[int, list], strategy) -> List[Tuple[int, int]]:
    """Judge every pair of instances without merging any of them.

    Unlike the eager loop this never skips a pair, so an instance already accepted for a merge
    is still compared against the rest, and every verdict is taken over the same precomputed
    geometry. Returns the accepted pairs as (earlier_id, later_id) in `objects_list` order.
    """
    candidates = strategy.candidate_pairs()
    if candidates is None:
        # Brute force: judge every pair.
        pairs: List[Tuple[int, int]] = []
        for i, instance1 in enumerate(objects_list):
            for instance2 in objects_list[i + 1:]:
                if strategy.same_instance(
                    instance1, instance2, obj_pcds[instance1.id], obj_pcds[instance2.id]
                ):
                    pairs.append((instance1.id, instance2.id))
        return pairs

    # Broadphase: judge only the candidate pairs, in objects_list order for a stable result.
    by_id = {inst.id: inst for inst in objects_list}
    rank = {inst.id: pos for pos, inst in enumerate(objects_list)}
    pairs = []
    for a, b in sorted(candidates, key=lambda p: (rank.get(p[0], len(rank)), rank.get(p[1], len(rank)))):
        if a not in by_id or b not in by_id:
            continue
        i1, i2 = (a, b) if rank[a] < rank[b] else (b, a)
        if strategy.same_instance(by_id[i1], by_id[i2], obj_pcds[i1], obj_pcds[i2]):
            pairs.append((i1, i2))
    return pairs


def group_merge_pairs(objects_list: Sequence, pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, List[int]]]:
    """Resolve accepted pairs into merge groups, one survivor each.

    Pairs are transitive: if A merges with B and B with C, the three end up together even
    though A and C were rejected. The survivor of a group is its earliest member in
    `objects_list`, the same instance the eager loop would have kept.

    Returns [(survivor_id, [absorbed_id, ...]), ...] ordered by survivor position.
    """
    rank = {instance.id: pos for pos, instance in enumerate(objects_list)}
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # keep the earliest instance as the root, so the survivor matches the eager order
        if rank.get(rb, len(rank)) < rank.get(ra, len(rank)):
            ra, rb = rb, ra
        parent[rb] = ra

    for a, b in pairs:
        union(a, b)

    groups: Dict[int, List[int]] = {}
    for ins_id in parent:
        root = find(ins_id)
        if ins_id != root:
            groups.setdefault(root, []).append(ins_id)

    return [
        (survivor, sorted(absorbed, key=lambda x: rank.get(x, len(rank))))
        for survivor, absorbed in sorted(groups.items(), key=lambda kv: rank.get(kv[0], len(rank)))
    ]
