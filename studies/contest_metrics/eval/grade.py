"""Grader puro: califica una decisión del contest contra el GT por punto.

Un solo primitivo unifica merge y split (lo que fusion_metrics no podía):
toda decisión reasigna un conjunto de puntos S de un defender a un challenger C.
  - g(S) = GT dominante del conjunto afectado.       (merge: S = todo el defender; split: S = split_points)
  - g(C) = GT dominante del challenger.
  - belongs = g(S) == g(C)   (el conjunto de verdad pertenece a donde va)
  - moved   = la decisión reasignó (MERGE o SPLIT)

Confusión (positivo = reasignar):
  moved & belongs      -> TP  (reasignación correcta)
  moved & ¬belongs     -> FP  (robo: se llevó puntos que no eran de C)
  ¬moved & belongs     -> FN  (perdida: había un move bueno y no se hizo)
  ¬moved & ¬belongs    -> TN  (separación correcta)
No calificable (SKIP): sin GT medible (void) o sin challenger.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Grade(str, Enum):
    TP = "TP"      # reasignación correcta
    FP = "FP"      # over-merge / robo
    FN = "FN"      # move bueno omitido
    TN = "TN"      # separación correcta
    SKIP = "SKIP"  # no calificable (void / sin challenger)


@dataclass
class GradedDecision:
    """Una decisión del contest puntuada contra GT."""

    defender: int
    challenger: int | None
    decision: str            # MERGE_CONTAINMENT | SPLIT | NO_ACTION | DEFER_TO_FUSION
    reason: str              # rama/gate que la produjo (para atribución por gate)
    moved: bool
    gt_affected: int | None  # g(S)
    gt_challenger: int | None  # g(C)
    belongs: bool | None
    grade: Grade


def grade_move(moved: bool, gt_affected: int | None, gt_challenger: int | None) -> tuple[bool | None, Grade]:
    """(belongs, grade) de un move. SKIP si falta GT en cualquiera de los dos lados."""
    if gt_affected is None or gt_challenger is None:
        return None, Grade.SKIP
    belongs = gt_affected == gt_challenger
    if moved:
        return belongs, Grade.TP if belongs else Grade.FP
    return belongs, Grade.FN if belongs else Grade.TN


def confusion(graded: list[GradedDecision]) -> dict[str, int]:
    """Cuenta TP/FP/FN/TN/SKIP sobre una lista de decisiones puntuadas."""
    out = {g.value: 0 for g in Grade}
    for d in graded:
        out[d.grade.value] += 1
    return out


def rates(counts: dict[str, int]) -> dict[str, float]:
    """Precision/recall/f1 tratando MERGE/SPLIT (positivo=reasignar) como el clasificador."""
    tp, fp, fn = counts.get("TP", 0), counts.get("FP", 0), counts.get("FN", 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}
