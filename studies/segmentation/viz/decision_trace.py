"""Lente trace: la tabla de auditoría de la poda, todas las máscaras y su veredicto.

Texto puro a partir del DecisionBreakdown. Ordena por score descendente (orden del NMS).
"""
from __future__ import annotations

from studies.segmentation.core.nms_decision import DecisionBreakdown


def to_markdown(breakdown: DecisionBreakdown) -> str:
    """DecisionBreakdown -> tabla markdown (id, área, score, estado, motivo)."""
    bd = breakdown
    head = (
        f"# Decisión NMS (OVO)  ·  iou_thr={bd.iou_thr}  score_thr={bd.score_thr}  inner_thr={bd.inner_thr}\n\n"
        f"kept={len(bd.kept)}  removed={len(bd.removed)}  total={len(bd.verdicts)}\n\n"
        "| id | área | score | estado | motivo |\n"
        "|---:|-----:|------:|:------:|:-------|\n"
    )
    rows = []
    for v in sorted(bd.verdicts, key=lambda x: x.score, reverse=True):
        if v.kept:
            estado, motivo = "KEPT", ""
        else:
            estado = "removed"
            motivo = "; ".join(
                (f"{k.kind}={k.value:.3f}" + (f" vs {k.killer_index}" if k.killer_index is not None else ""))
                for k in v.kills
            )
        rows.append(f"| {v.index} | {v.area} | {v.score:.3f} | {estado} | {motivo} |")
    return head + "\n".join(rows) + "\n"


def render(breakdown: DecisionBreakdown, out_path: str) -> None:
    """Escribe la tabla de auditoría en out_path (markdown)."""
    with open(out_path, "w") as f:
        f.write(to_markdown(breakdown))
