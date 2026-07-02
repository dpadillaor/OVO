"""Print per-criterion fusion timing (count, total time, mean) for a run/scene.

Usage:
    python studies/fusion_metrics/scripts/criterion_timing.py \
        --exp_path data/output/Replica/<run> --scene office0

Use --criteria to specify the criterion chain order matching the experiment config.
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
from core.timing.stats import render_table  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Run dir containing {scene}/logger/")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name, e.g. office0")
    parser.add_argument("--criteria", nargs="+", default=None,
                        help="Criterion names (default: uses CRITERIA from reader.py)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timings = load_criterion_timings(pathlib.Path(args.exp_path), args.scene,
                                     criteria=args.criteria)
    print(render_table(timings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
