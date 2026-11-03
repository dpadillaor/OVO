"""Apply policy: which pairs are still worth judging, and how accepted pairs become merges.

The pipeline asks `should_judge(a, b)` before measuring anything for a pair, so a pair whose
outcome can no longer change the result costs nothing. Accepted pairs are reported with
`on_accept(a, b)`; `merges()` returns (survivor, absorbed) in the order to apply them.

    greedy      merge B into A unless either was already absorbed. No chains: once A absorbed
                B, every later pair with B is skipped.
    chained     fusions chain (union-find): A~B and B~C fuse A, B and C, even if (A, C) was
                rejected. Pairs already in the same group are skipped. The survivor is the
                earliest instance.
"""

Pair = tuple[int, int]


class Greedy:
    def __init__(self, order: list[int]):
        self._absorbed: set[int] = set()
        self._merges: list[Pair] = []

    def should_judge(self, a: int, b: int) -> bool:
        return a not in self._absorbed and b not in self._absorbed

    def on_accept(self, a: int, b: int) -> None:
        self._absorbed.add(b)
        self._merges.append((a, b))

    def merges(self) -> list[Pair]:
        return self._merges


class Chained:
    def __init__(self, order: list[int]):
        self._rank = {i: k for k, i in enumerate(order)}
        self._parent = {i: i for i in order}

    def _root(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def should_judge(self, a: int, b: int) -> bool:
        return self._root(a) != self._root(b)

    def on_accept(self, a: int, b: int) -> None:
        ra, rb = self._root(a), self._root(b)
        keep, drop = (ra, rb) if self._rank[ra] < self._rank[rb] else (rb, ra)
        self._parent[drop] = keep

    def merges(self) -> list[Pair]:
        merges = [(self._root(i), i) for i in self._parent if self._root(i) != i]
        return sorted(merges, key=lambda m: (self._rank[m[0]], self._rank[m[1]]))


POLICIES = {"greedy": Greedy, "chained": Chained}
