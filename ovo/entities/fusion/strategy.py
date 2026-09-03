from typing import Tuple, Dict, List

from .criteria import Criterion

import time
import torch


class FusionStrategy:
    def __init__(self, criteria: List[Criterion], profile: bool = False):
        self.criteria = criteria
        self._profile = profile
        self._decisions: list = []
        self._criterion_times: Dict[str, float] = {}
        self._n_pairs_evaluated: int = 0
        self._n_pairs_sc: Dict[str, int] = {}

    def prepare(self, objects_list, obj_pcds) -> None:
        """Run each criterion's per-pass precompute once, before the pair loop.

        The precompute time folds into the criterion's own `t_crit_<name>`, so that timer is
        the criterion's FULL cost, precompute plus per-pair. A criterion with no precompute
        (cos_sim, cooccurrence) is unaffected. The GPU sync that makes the timing honest runs
        only when profiling: it stalls the pipeline, so production pays nothing for it.
        """
        sync = self._profile and torch.cuda.is_available()
        for criterion in self.criteria:
            if sync:
                torch.cuda.synchronize()
            t0 = time.time()
            criterion.prepare(objects_list, obj_pcds)
            if sync:
                torch.cuda.synchronize()
            self._criterion_times[criterion.name] = self._criterion_times.get(criterion.name, 0.0) + (time.time() - t0)

    _ACCEPTING = {"overlap", "overlap_old", "overlap_voxel"}

    def candidate_pairs(self):
        """Broadphase candidate pairs, or None to fall back to the O(N^2) loop.

        Only safe (lossless) when the voxel criterion is the SOLE accepting criterion: then a
        pair sharing no cell can never be accepted, so restricting to shared-cell pairs drops
        nothing. With any other accepting criterion (overlap_old accepts on point proximity, not
        cell co-occupancy) the shared-cell set could miss a real merge, so we return None.
        """
        accepters = [c for c in self.criteria if c.name in self._ACCEPTING]
        if len(accepters) == 1 and accepters[0].name == "overlap_voxel":
            return accepters[0].candidate_pairs()
        return None

    def same_instance(
        self,
        instance1,
        instance2,
        points1: torch.Tensor,
        points2: torch.Tensor,
    ) -> bool:
        ctx: dict = {}
        self._n_pairs_evaluated += 1

        for criterion in self.criteria:
            t0 = time.time()
            verdict, decision = criterion.check(instance1, instance2, points1, points2, ctx)
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
