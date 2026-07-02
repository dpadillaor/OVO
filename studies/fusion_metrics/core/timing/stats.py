"""Pure aggregation + terminal rendering for fusion criterion timings (no disk, no print)."""
from __future__ import annotations

from core.timing.reader import CriterionTiming


_HEADERS = ("criterion", "count", "total_s", "average_ms_pair")


def render_table(timings: list[CriterionTiming]) -> str:
    """Format criterion timings as an aligned table (count, total s, mean ms). Caller prints it."""
    if not timings:
        return "No active fusion criteria found (no sc_*/t_crit_* logs)."

    rows = [
        (t.name, str(t.count), f"{t.time_s:.4f}", f"{t.mean_s * 1e3:.3f}")
        for t in timings
    ]
    total_count = sum(t.count for t in timings)
    total_time = sum(t.time_s for t in timings)
    rows.append(("TOTAL", str(total_count), f"{total_time:.4f}", ""))

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(_HEADERS)]

    def fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
                         for i, c in enumerate(cells))

    sep = "  ".join("-" * w for w in widths)
    lines = [fmt(_HEADERS), sep, *(fmt(r) for r in rows[:-1]), sep, fmt(rows[-1])]
    return "\n".join(lines)
