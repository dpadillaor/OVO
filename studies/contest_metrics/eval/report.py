"""Formateo puro de resultados de eval: confusión, precisión/recall, por-gate, missed. Sin I/O."""

from __future__ import annotations

from collections import defaultdict

from .grade import Grade, confusion, rates
from .pipeline import SceneEval


def gate_key(reason: str) -> str:
    """Colapsa el reason a un gate canónico (sin números): 'parcial + descriptor igual (sim=0.91) -> merge'
    -> 'parcial + descriptor igual'. Para agrupar variantes del mismo gate."""
    return reason.split("(")[0].split("->")[0].strip() or reason or "(sin reason)"


def _confusion_line(counts: dict) -> str:
    r = rates(counts)
    return (f"TP={counts['TP']:>3} FP={counts['FP']:>3} FN={counts['FN']:>3} "
            f"TN={counts['TN']:>3} SKIP={counts['SKIP']:>3}  "
            f"prec={r['precision']:.3f} rec={r['recall']:.3f} f1={r['f1']:.3f}")


def format_scene(se: SceneEval) -> str:
    if se.error:
        return f"[{se.scene}] ERROR: {se.error}"
    counts = confusion(se.decisions)
    lines = [f"[{se.scene}]  {len(se.decisions)} decisiones · {_confusion_line(counts)}",
             f"           missed transfers (trozo del challenger no movido): {len(se.missed)}"]
    for m in se.missed[:5]:
        lines.append(f"             def {m.defender} -> ch {m.challenger}  objeto GT {m.gt_object}  ({m.firm_points} pts firmes)")
    return "\n".join(lines)


def _pooled_gates(evals: list[SceneEval]) -> list[tuple[str, dict]]:
    """Confusión por gate, agregada sobre todas las escenas."""
    by_gate: dict[str, list] = defaultdict(list)
    for se in evals:
        for d in se.decisions:
            by_gate[gate_key(d.reason)].append(d)
    rows = [(g, confusion(ds)) for g, ds in by_gate.items()]
    rows.sort(key=lambda gc: -(gc[1]["FP"] * 100 + gc[1]["TP"]))  # gates con FP arriba
    return rows


def format_summary(evals: list[SceneEval], title: str = "") -> str:
    ok = [se for se in evals if not se.error]
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    # por escena
    for se in evals:
        lines.append(format_scene(se))
        lines.append("")
    # pooled confusion
    pooled = confusion([d for se in ok for d in se.decisions])
    total_missed = sum(len(se.missed) for se in ok)
    lines.append("-" * 72)
    lines.append(f"POOLED ({len(ok)} escenas)  {_confusion_line(pooled)}")
    lines.append(f"missed transfers totales: {total_missed}")
    # por gate
    lines.append(f"\n{'gate':<42} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'SKIP':>4}")
    for gate, cc in _pooled_gates(ok):
        lines.append(f"{gate[:40]:<42} {cc['TP']:>3} {cc['FP']:>3} {cc['FN']:>3} {cc['TN']:>3} {cc['SKIP']:>4}")
    return "\n".join(lines)


def to_payload(evals: list[SceneEval]) -> dict:
    """Estructura JSON-serializable para --json (per-scene + pooled + por-gate)."""
    ok = [se for se in evals if not se.error]
    return {
        "scenes": [
            {
                "scene": se.scene,
                "error": se.error,
                "confusion": confusion(se.decisions) if not se.error else None,
                "rates": rates(confusion(se.decisions)) if not se.error else None,
                "missed": [vars(m) for m in se.missed],
            }
            for se in evals
        ],
        "pooled": {
            "confusion": confusion([d for se in ok for d in se.decisions]),
            "missed_total": sum(len(se.missed) for se in ok),
            "by_gate": [{"gate": g, **cc} for g, cc in _pooled_gates(ok)],
        },
    }
