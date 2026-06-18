"""Write the two evaluation outputs: enriched per-pair CSV + summary JSON."""
from __future__ import annotations

import csv
import json
import pathlib

from core.merge_decision_eval import EvaluatedPair

_EXTRA_COLS = ["same_object", "verdict"]


def write_pairs_csv(
    rows: list[dict[str, str]],
    pairs: list[EvaluatedPair],
    out_path: str | pathlib.Path,
) -> None:
    """Duplicate the original rows, appending same_object + verdict columns.

    ``rows`` and ``pairs`` must be aligned by index (same order, same length).
    """
    if len(rows) != len(pairs):
        raise ValueError(f"rows ({len(rows)}) and pairs ({len(pairs)}) misaligned.")

    fieldnames = list(rows[0].keys()) + _EXTRA_COLS if rows else _EXTRA_COLS
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row, pair in zip(rows, pairs):
            writer.writerow(
                {**row, "same_object": pair.same_object, "verdict": pair.verdict.value}
            )


def _pretty_json(obj, level: int = 1, indent: int = 2) -> str:
    """JSON with nested structure indented but scalar-only dicts/lists inline."""
    pad, close = " " * (indent * level), " " * (indent * (level - 1))
    nested = (dict, list)
    if isinstance(obj, dict):
        if all(not isinstance(v, nested) for v in obj.values()):
            return json.dumps(obj)  # leaf dict -> one line
        items = [f"{pad}{json.dumps(k)}: {_pretty_json(v, level + 1, indent)}"
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + close + "}"
    if isinstance(obj, list):
        if all(not isinstance(v, nested) for v in obj):
            return json.dumps(obj)  # scalar list -> one line
        items = [f"{pad}{_pretty_json(v, level + 1, indent)}" for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + close + "]"
    return json.dumps(obj)


def write_summary_json(summary: dict, out_path: str | pathlib.Path) -> None:
    """Write the summary as JSON: structure indented, scalar leaves inline."""
    with open(out_path, "w") as f:
        f.write(_pretty_json(summary) + "\n")
