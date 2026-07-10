"""Validador de sombra (R1): compara las features del batch vs las del agregador online.

El batch (`aggregator.pairs()`) es el ORÁCULO. En modo sombra el online calcula en paralelo y esto
comprueba que coinciden campo a campo: ints exactos, floats con `atol` (la media `persistence` puede
diferir en el último ULP por orden de suma). Un mismatch = bug de invalidación localizado en un
defender (típicamente un dirty-mark olvidado). Valida el PLUMBING, no el comportamiento vivo (§11.2 R6).
"""
from typing import Dict, List, Tuple

from .types import ComparisonReport, FeatureTolerance, InsId, Mismatch, PairFeatures


class ShadowValidator:
    def compare(
        self,
        batch: Dict[InsId, List[PairFeatures]],
        online: Dict[InsId, List[PairFeatures]],
        tol: FeatureTolerance,
        report_idx: int = 0,
    ) -> ComparisonReport:
        b = self._index(batch)
        o = self._index(online)
        mismatches: List[Mismatch] = []
        for key in set(b) | set(o):
            bp, op = b.get(key), o.get(key)
            if bp is None:
                mismatches.append(Mismatch(key[0], key[1], "*", 0.0, 0.0, "missing_batch"))
                continue
            if op is None:
                mismatches.append(Mismatch(key[0], key[1], "*", 0.0, 0.0, "missing_online"))
                continue
            for f in FeatureTolerance.INT_FIELDS:
                bv, ov = getattr(bp, f), getattr(op, f)
                if bv != ov:
                    mismatches.append(Mismatch(key[0], key[1], f, float(bv), float(ov)))
            for f in FeatureTolerance.FLOAT_FIELDS:
                bv, ov = getattr(bp, f), getattr(op, f)
                if abs(bv - ov) > tol.atol:
                    mismatches.append(Mismatch(key[0], key[1], f, float(bv), float(ov)))
        n_b = sum(len(v) for v in batch.values())
        n_o = sum(len(v) for v in online.values())
        return ComparisonReport(report_idx, n_b, n_o, tuple(mismatches))

    @staticmethod
    def _index(feats: Dict[InsId, List[PairFeatures]]) -> Dict[Tuple[InsId, InsId], PairFeatures]:
        return {(pf.defender, pf.challenger): pf for pairs in feats.values() for pf in pairs}
