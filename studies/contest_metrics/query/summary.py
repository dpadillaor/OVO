"""Resumen agregado del contest sobre un experimento: cuánto actuó y cómo cambió el mapa.

Cuenta, por escena y en total:
  - veredictos aplicados (MERGE_CONTAINMENT / SPLIT / NO_ACTION) desde contest_verdicts.csv,
  - nº de instancias finales (ids distintos en instance_pred/<scene>.txt),
y, si se pasa un baseline, el nº de instancias baseline vs contest y su delta.

Sin torch: solo lee csv y el pred por vértice. Complementa a `history` (timeline por par).
"""
from __future__ import annotations

import pathlib
from collections import Counter

from ..common.resolve import resolve_exp_path
from .verdicts import load_verdicts

_DECISIONS = ("MERGE_CONTAINMENT", "SPLIT", "NO_ACTION")


def _verdicts_csv(exp_path: pathlib.Path, scene: str) -> pathlib.Path:
    return exp_path / scene / "fusion" / "contest" / "contest_verdicts.csv"


def _instance_pred(exp_path: pathlib.Path, scene: str) -> pathlib.Path:
    return exp_path / "instance_pred" / f"{scene}.txt"


def count_verdicts(exp_path: pathlib.Path, scene: str) -> Counter:
    """{MERGE_CONTAINMENT, SPLIT, NO_ACTION} sobre el histórico del run (0 si no hay csv)."""
    rows = load_verdicts(_verdicts_csv(exp_path, scene))
    return Counter(r.get("decision") for r in rows)


def count_instances(exp_path: pathlib.Path, scene: str) -> int | None:
    """Nº de instancias en el mapa final: una línea por instancia en instance_pred/<scene>.txt."""
    p = _instance_pred(exp_path, scene)
    if not p.exists():
        return None
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def summarize(exp: str, scenes: list[str], baseline: str | None = None) -> dict:
    """Por escena + totales: veredictos, instancias finales, y (si baseline) instancias base vs contest."""
    exp_path = resolve_exp_path(exp)
    base_path = resolve_exp_path(baseline) if baseline else None
    rows = []
    for s in scenes:
        v = count_verdicts(exp_path, s)
        row = {
            "scene": s,
            "merge": v.get("MERGE_CONTAINMENT", 0),
            "split": v.get("SPLIT", 0),
            "no_action": v.get("NO_ACTION", 0),
            "ins_contest": count_instances(exp_path, s),
            "ins_base": count_instances(base_path, s) if base_path else None,
        }
        rows.append(row)
    tot = {k: sum(r[k] for r in rows if isinstance(r[k], int)) for k in
           ("merge", "split", "no_action", "ins_contest", "ins_base")}
    tot["scene"] = "TOTAL"
    if base_path is None:
        tot["ins_base"] = None
    return {"exp": exp_path.name, "baseline": base_path.name if base_path else None,
            "rows": rows, "total": tot}


def format_summary(data: dict) -> str:
    has_base = data["baseline"] is not None
    head = f"contest summary · {data['exp']}"
    if has_base:
        head += f"   (baseline: {data['baseline']})"
    cols = f"  {'scene':8s} {'MERGE':>6s} {'SPLIT':>6s} {'NO_ACT':>7s} {'#ins':>6s}"
    if has_base:
        cols += f" {'#base':>6s} {'Δins':>6s}"
    lines = [head, cols, "  " + "-" * (len(cols) - 2)]

    def fmt(r):
        ic = r["ins_contest"]
        line = (f"  {r['scene']:8s} {r['merge']:>6} {r['split']:>6} {r['no_action']:>7} "
                f"{('' if ic is None else ic):>6}")
        if has_base:
            ib = r["ins_base"]
            d = "" if (ic is None or ib is None) else f"{ic - ib:+d}"
            line += f" {('' if ib is None else ib):>6} {d:>6}"
        return line

    for r in data["rows"]:
        lines.append(fmt(r))
    lines.append("  " + "-" * (len(cols) - 2))
    lines.append(fmt(data["total"]))
    return "\n".join(lines)
