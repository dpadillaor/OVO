from typing import Tuple, Dict, List

from .criteria import Criterion

import time
import torch


class FusionStrategy:
    def __init__(self, criteria: List[Criterion]):
        self.criteria = criteria
        self._decisions: list = []
        self._criterion_times: Dict[str, float] = {}
        self._n_pairs_evaluated: int = 0
        self._n_pairs_sc: Dict[str, int] = {}

    def same_instance(
        self,
        instance1,
        instance2,
        points_centroid1: Tuple[torch.Tensor, torch.Tensor],
        points_centroid2: Tuple[torch.Tensor, torch.Tensor],
    ) -> bool:
        points1, centroid1 = points_centroid1
        points2, centroid2 = points_centroid2
        ctx: dict = {}
        self._n_pairs_evaluated += 1

        for criterion in self.criteria:
            t0 = time.time()
            verdict, decision = criterion.check(instance1, instance2, points1, centroid1, points2, centroid2, ctx)
            self._criterion_times[criterion.name] = self._criterion_times.get(criterion.name, 0.0) + (time.time() - t0)
            if verdict is not None:
                sc_key = f"sc_{criterion.name}"
                self._n_pairs_sc[sc_key] = self._n_pairs_sc.get(sc_key, 0) + 1
                if decision:
                    self._decisions.append(decision)
                return verdict

        return False

    def pop_decisions(self) -> list:
        decisions, self._decisions = self._decisions, []
        return decisions

    def pop_timings(self) -> Dict[str, float]:
        timings = {**self._criterion_times, "n_pairs_evaluated": self._n_pairs_evaluated, **self._n_pairs_sc}
        self._criterion_times = {}
        self._n_pairs_evaluated = 0
        self._n_pairs_sc = {}
        return timings
