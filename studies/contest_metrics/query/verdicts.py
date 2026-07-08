"""Histórico de veredictos: lee contest_verdicts.csv (lo que el contest DECIDIÓ, report a report).

Sin torch. Es el "qué pasó" a lo largo del run, complemento del "qué pasa ahora" del engine.
"""

from __future__ import annotations

import csv
import pathlib

_NUM = {"report", "defender", "challenger", "firm_points", "total_grabs",
        "containment", "reverse_containment", "persistence", "focus", "exclusivity",
        "sim", "seam_angle", "de_ch", "de_def"}
_INT = {"report", "defender", "challenger", "firm_points", "total_grabs"}


def _coerce(key: str, val: str):
    if val is None or val == "":
        return None
    if key in _INT:
        return int(float(val))
    if key in _NUM:
        return float(val)
    return val


def load_verdicts(path: str | pathlib.Path) -> list[dict]:
    """Filas del csv como dicts tipados. [] si el fichero no existe o está vacío."""
    p = pathlib.Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        return [{k: _coerce(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def filter_rows(rows: list[dict], *, defender: int | None = None, challenger: int | None = None,
                decision: str | None = None, report: int | None = None,
                pair: tuple[int, int] | None = None) -> list[dict]:
    """Filtra por cualquier combinación. `pair=(a,b)` casa ambos sentidos."""
    out = rows
    if pair is not None:
        a, b = pair
        out = [r for r in out if {r.get("defender"), r.get("challenger")} == {a, b}
               or (r.get("defender") in (a, b) and r.get("challenger") is None)]
    if defender is not None:
        out = [r for r in out if r.get("defender") == defender]
    if challenger is not None:
        out = [r for r in out if r.get("challenger") == challenger]
    if decision is not None:
        out = [r for r in out if r.get("decision") == decision.upper()]
    if report is not None:
        out = [r for r in out if r.get("report") == report]
    return out


def format_history(rows: list[dict]) -> str:
    """Timeline de decisiones (una fila por report)."""
    if not rows:
        return "(sin histórico: verdicts.csv vacío o inexistente)"
    head = f"  {'rep':>4} {'def':>5} {'chall':>5} {'decision':<18} {'contain':>8} {'persist':>8}  reason"
    lines = [head]
    for r in sorted(rows, key=lambda x: (x.get("report") or 0, x.get("defender") or 0)):
        ch = r.get("challenger")
        cont = r.get("containment")
        pers = r.get("persistence")
        lines.append(
            f"  {r.get('report'):>4} {r.get('defender'):>5} "
            f"{('' if ch is None else ch):>5} {r.get('decision', ''):<18} "
            f"{('' if cont is None else f'{cont:.3f}'):>8} "
            f"{('' if pers is None else f'{pers:.3f}'):>8}  {r.get('reason', '')}"
        )
    return "\n".join(lines)
