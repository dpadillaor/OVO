"""Per-criterion cascade view: each gate scored as a classifier on its incoming stream.

The fusion chain short-circuits: a rejected pair dies at exactly one gate (its
``reason``); an accepted pair survived every gate. Given the chain order we can
replay the funnel -- what enters each gate, what it passes, what it rejects --
and score each gate with positive = "let the pair pass".
"""
from __future__ import annotations

from collections import defaultdict


def _rates(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    """precision, recall, f1 (positive = pass); None when a denominator is 0."""
    precision = round(tp / (tp + fp), 4) if tp + fp else None
    recall = round(tp / (tp + fn), 4) if tp + fn else None
    f1 = round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else None
    return precision, recall, f1


def cascade_by_criterion(pairs: list, chain: list[str]) -> list[dict]:
    """One row per gate (chain order), positive = pass. Terminal gate carries accept_modes.

    A pair leaves the cascade at gate ``chain.index(reason)`` if rejected, or after
    the last gate if accepted. Pairs whose reject reason is absent from ``chain``
    (chain/reason mismatch) drop out silently -- gate 0's ``eval`` then falls below
    the scored-pair total, which is the signal that the chain order is wrong.
    """
    n = len(chain)
    index = {name: i for i, name in enumerate(chain)}

    def gate_of(p) -> int:
        if p.decision.merged:
            return n  # survived all gates
        return index.get(p.decision.reason, -1)  # -1 = reason not in chain

    gpairs = [(gate_of(p), p) for p in pairs]

    rows: list[dict] = []
    for i, name in enumerate(chain):
        entering = [p for g, p in gpairs if g >= i]
        rejected = [p for g, p in gpairs if g == i]
        passed = [p for g, p in gpairs if g > i]

        tp = sum(1 for p in passed if p.same_object)
        fp = len(passed) - tp
        fn = sum(1 for p in rejected if p.same_object)
        tn = len(rejected) - fn
        precision, recall, f1 = _rates(tp, fp, fn)

        row = {
            "criterion": name,
            "eval": len(entering), "pass": len(passed), "reject": len(rejected),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
        if i == n - 1:  # terminal gate is the one that accepts -> split pass by OR branch
            modes: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "TP": 0, "FP": 0})
            for p in passed:
                m = p.decision.accept_mode or "?"
                modes[m]["pass"] += 1
                modes[m]["TP" if p.same_object else "FP"] += 1
            row["accept_modes"] = {k: modes[k] for k in sorted(modes)}
        rows.append(row)
    return rows
