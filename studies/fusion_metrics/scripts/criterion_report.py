"""Print a per-criterion fusion report: timing (count/time/mean) + verdict counts (total/FN/TN).

Joins the timing logs with the class-agnostic verdict breakdown already computed by
eval_fusion_decisions.py (read from fusion_eval_summary.json). Run that eval first if the
verdict columns come up blank.

Usage:
    python studies/fusion_metrics/scripts/criterion_report.py \
        --exp_path data/output/Replica/<run> --scene office0
"""
from __future__ import annotations

import sys
import pathlib
import argparse

# Make the package root (studies/fusion_metrics) importable as core.*
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.timing.reader import load_criterion_timings  # noqa: E402
from core.agnostic_impact.eval_summary import load_verdicts  # noqa: E402
from core.report import build_rows, render_report, render_cascade  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Run dir containing {scene}/logger/ and fusion_eval_summary.json")
    parser.add_argument("--scene", type=str, required=True, help="Scene name, e.g. office0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exp_path = pathlib.Path(args.exp_path)
    timings = load_criterion_timings(exp_path, args.scene)
    verdicts = load_verdicts(exp_path, args.scene)
    cascade = verdicts.get("by_criterion", [])
    counts = verdicts.get("counts", {})
    footer = {"total": verdicts.get("totals", {}).get("accepted"),
              "TP": counts.get("TP"), "FP": counts.get("FP")}
    print(render_report(build_rows(timings, cascade), footer))

    if cascade:
        print("\nCascade (each gate as a classifier, positive = pass):")
        print(render_cascade(cascade))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
