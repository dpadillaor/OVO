"""Pure: join per-criterion fusion timing with the by_criterion cascade into terminal tables."""
from __future__ import annotations

from dataclasses import dataclass

from core.timing.reader import CriterionTiming


@dataclass
class CriterionRow:
    """A criterion's timing (count/time) joined with its reject verdict counts (from by_criterion)."""

    name: str
    count: int | None
    time_s: float | None
    mean_s: float | None
    rej_total: int | None
    FN: int | None
    TN: int | None


def build_rows(timings: list[CriterionTiming], by_criterion: list[dict]) -> list[CriterionRow]:
    """One row per criterion seen in timing or in the by_criterion cascade."""
    timing_by_name = {t.name: t for t in timings}
    cascade_by_name = {c["criterion"]: c for c in by_criterion}

    rows = []
    for name in [*timing_by_name, *(c for c in cascade_by_name if c not in timing_by_name)]:
        t = timing_by_name.get(name)
        g = cascade_by_name.get(name)
        rows.append(CriterionRow(
            name=name,
            count=t.count if t else None,
            time_s=t.time_s if t else None,
            mean_s=t.mean_s if t else None,
            rej_total=g["reject"] if g else None,
            FN=g["FN"] if g else None,
            TN=g["TN"] if g else None,
        ))
    return rows


_HEADERS = ("criterion", "count", "total_s", "average_ms_pair", "rej_total", "FN", "TN")


def _cell(v, fmt: str = "") -> str:
    """Format a value, blank for None."""
    return "" if v is None else (format(v, fmt) if fmt else str(v))


def render_report(rows: list[CriterionRow], footer: dict | None = None) -> str:
    """Aligned table: per-criterion timing + reject verdicts; ACCEPTED summary footer.

    ``footer`` (optional) = {"total", "TP", "FP"} for accepted merges.
    """
    if not rows:
        return "No criteria found (no sc_*/t_crit_* logs and no fusion_eval_summary.json)."

    table_rows = [
        (r.name, _cell(r.count), _cell(r.time_s, ".4f"),
         _cell(None if r.mean_s is None else r.mean_s * 1e3, ".3f"),
         _cell(r.rej_total), _cell(r.FN), _cell(r.TN))
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in table_rows)) for i, h in enumerate(_HEADERS)]

    def fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
                         for i, c in enumerate(cells))

    sep = "  ".join("-" * w for w in widths)
    lines = [fmt(_HEADERS), sep, *(fmt(r) for r in table_rows)]

    if footer and footer.get("total"):
        lines += ["", f"ACCEPTED: total={footer['total']}  "
                      f"TP={footer.get('TP')}  FP={footer.get('FP')}"]
    return "\n".join(lines)


_CASCADE_HEADERS = ("criterion", "eval", "pass", "reject", "TP", "TN", "FP", "FN",
                    "precision", "recall", "f1")


def render_cascade(by_criterion: list[dict]) -> str:
    """Aligned cascade table: each gate as a classifier on its incoming stream.

    ``by_criterion`` is the summary block from summarize(..., chain=...). The
    terminal gate's ``accept_modes`` render as indented A/B/AB sub-rows.
    """
    if not by_criterion:
        return "No cascade (by_criterion absent; re-run eval with a fusion chain)."

    def num(v, fmt="") -> str:
        return "" if v is None else (format(v, fmt) if fmt else str(v))

    body: list[tuple[str, ...]] = []
    for g in by_criterion:
        body.append((
            g["criterion"], num(g["eval"]), num(g["pass"]), num(g["reject"]),
            num(g["TP"]), num(g["TN"]), num(g["FP"]), num(g["FN"]),
            num(g.get("precision"), ".3f"), num(g.get("recall"), ".3f"), num(g.get("f1"), ".3f"),
        ))
        for mode, m in (g.get("accept_modes") or {}).items():
            body.append((
                f"  {mode}", "", num(m["pass"]), "",
                num(m["TP"]), "", num(m["FP"]), "", "", "", "",
            ))

    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(_CASCADE_HEADERS)]

    def fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
                         for i, c in enumerate(cells))

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(_CASCADE_HEADERS), sep, *(fmt(r) for r in body)])
