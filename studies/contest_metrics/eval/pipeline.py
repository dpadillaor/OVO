"""Orquestación: califica las decisiones del contest de una escena (o varias) contra GT.

Reutiliza ContestProbe (veredictos en vivo bajo cualquier config -> contrafactual sin re-run)
+ SceneGT (GT por punto) + grade (primitivo point-set). Dos vistas:

  - decisions: veredicto por defender (lo que el contest DECIDIÓ) -> confusión por gate.
  - missed:    barrido de pares firmes cuyo trozo pertenece al challenger y NO se transfirió
               (oportunidades perdidas; el FN que el nivel-veredicto no ve por ser winner-take-all).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..query.engine import ContestProbe
from .gt import load_scene_gt
from .grade import GradedDecision, grade_move

MOVE_DECISIONS = {"MERGE_CONTAINMENT", "SPLIT"}

SCENE_GROUPS = {
    "tuning": ["office0", "office3", "office4", "room2"],
    "held": ["office1", "office2", "room0", "room1"],
    "all": ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"],
}


@dataclass
class MissedTransfer:
    """Un trozo firme que pertenece (GT) al challenger pero no se transfirió."""
    defender: int
    challenger: int
    gt_object: int
    firm_points: int


@dataclass
class SceneEval:
    scene: str
    decisions: list[GradedDecision] = field(default_factory=list)
    missed: list[MissedTransfer] = field(default_factory=list)
    error: str | None = None


def _rows_by_instance(map_state):
    """Devuelve rows(instance) -> filas de sus puntos, con caché."""
    ins = map_state.points_ins_ids.numpy()
    cache: dict[int, np.ndarray] = {}

    def rows(instance):
        if instance not in cache:
            cache[instance] = np.where(ins == instance)[0]
        return cache[instance]

    return rows


def grade_scene(exp: str, scene: str, overrides: dict | None = None) -> SceneEval:
    """Califica una escena. Nunca lanza por datos que falten: lo reporta en .error."""
    try:
        probe = ContestProbe.from_experiment(exp, scene, overrides=overrides or {})
    except (FileNotFoundError, ValueError) as e:
        return SceneEval(scene, error=str(e))

    gt = load_scene_gt(scene, probe._map)
    rows = _rows_by_instance(probe._map)

    decisions: list[GradedDecision] = []
    moved_pairs: set[tuple[int, int]] = set()
    for defender in probe._by_def:
      for v in probe.classify_all(defender):
        decision, challenger = v.decision.name, v.challenger
        moved = decision in MOVE_DECISIONS
        if moved and challenger is not None:
            moved_pairs.add((defender, challenger))

        # conjunto afectado S según la decisión
        if decision == "MERGE_CONTAINMENT":
            gt_s = gt.dominant_over_rows(rows(defender))            # toda la instancia
        elif decision == "SPLIT":
            gt_s = gt.dominant_over_ids(v.split_points or [])       # subconjunto transferido
        else:                                                       # NO_ACTION / DEFER
            pf = probe.directed(defender, challenger) if challenger is not None else None
            gt_s = gt.dominant_over_ids(pf.split_points) if (pf and pf.split_points) else None

        gt_c = gt.dominant_over_rows(rows(challenger)) if challenger is not None else None
        belongs, grade = grade_move(moved, gt_s, gt_c)
        decisions.append(GradedDecision(defender, challenger, decision, v.reason or "",
                                        moved, gt_s, gt_c, belongs, grade))

    missed = _scan_missed(probe, gt, rows, moved_pairs)
    return SceneEval(scene, decisions=decisions, missed=missed)


def _scan_missed(probe, gt, rows, moved_pairs) -> list[MissedTransfer]:
    """Pares firmes cuyo trozo pertenece (GT) al challenger, distinto del defender, y no se movió."""
    out: list[MissedTransfer] = []
    for f in probe._pairs:
        if (f.defender, f.challenger) in moved_pairs:
            continue
        gt_chunk = gt.dominant_over_ids(f.split_points)
        gt_ch = gt.dominant_over_rows(rows(f.challenger))
        gt_def = gt.dominant_over_rows(rows(f.defender))
        if gt_chunk is not None and gt_ch is not None and gt_chunk == gt_ch and gt_ch != gt_def:
            out.append(MissedTransfer(f.defender, f.challenger, gt_ch, f.firm_points))
    out.sort(key=lambda m: -m.firm_points)
    return out


def grade_scenes(exp: str, scenes: list[str], overrides: dict | None = None) -> list[SceneEval]:
    return [grade_scene(exp, s, overrides) for s in scenes]
