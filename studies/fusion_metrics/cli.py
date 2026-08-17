"""Fusion-metrics analysis CLI: eval | report | timing (run from studies/fusion_metrics).

    python -m cli eval   --exp_path <run> [--scene office0]   # score decisions, write artifacts
    python -m cli report --exp_path <run> --scene office0     # per-criterion timing + verdicts
    python -m cli timing --exp_path <run> --scene office0     # per-criterion timing only

"""
from __future__ import annotations

import sys
import pathlib
import argparse

# Package root (studies/fusion_metrics) — this file lives at it — for core.*
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.pipeline import evaluate_experiment, DEFAULT_MESH_ROOT, DEFAULT_GT_ROOT
from core.timing.reader import load_criterion_timings
from core.timing.stats import render_table
from core.agnostic_impact.eval_summary import load_verdicts
from core.report import build_rows, render_report, render_cascade


def cmd_eval(args: argparse.Namespace) -> int:
    return evaluate_experiment(
        args.exp_path, scene=args.scene, ckpt=args.ckpt,
        mesh_root=args.mesh_root, gt_root=args.gt_root, out_dir=args.out_dir,
    )


def cmd_report(args: argparse.Namespace) -> int:
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


def cmd_timing(args: argparse.Namespace) -> int:
    timings = load_criterion_timings(pathlib.Path(args.exp_path), args.scene,
                                     criteria=args.criteria)
    print(render_table(timings))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli",
        description="Fusion-metrics analysis: score a run's fusion decisions and report on them.",
    )
    sub = parser.add_subparsers(dest="command", metavar="{eval,report,timing}")

    p_eval = sub.add_parser("eval", help="Score fusion decisions vs GT; write eval artifacts")
    p_eval.add_argument("--exp_path", required=True,
                        help="Run dir containing {scene}/fusion_decisions.csv")
    p_eval.add_argument("--scene", default=None,
                        help="Single scene. Omit to scan & evaluate every scene in the run.")
    p_eval.add_argument("--ckpt", default=None,
                        help="pre_fusion.ckpt path (default: mirror under data/checkpoints)")
    p_eval.add_argument("--mesh_root", default=str(DEFAULT_MESH_ROOT),
                        help="Dir holding {scene}_mesh.ply")
    p_eval.add_argument("--gt_root", default=str(DEFAULT_GT_ROOT),
                        help="Dir holding instance ground-truth {scene}.txt")
    p_eval.add_argument("--out_dir", default=None,
                        help="Where to write outputs (default: {scene}/fusion/fusion_LC/)")
    p_eval.set_defaults(func=cmd_eval)

    p_report = sub.add_parser("report", help="Per-criterion timing + verdict counts (run eval first)")
    p_report.add_argument("--exp_path", required=True,
                          help="Run dir with {scene}/logger/ and {scene}/fusion/fusion_LC/fusion_eval_summary.json")
    p_report.add_argument("--scene", required=True, help="Scene name, e.g. office0")
    p_report.set_defaults(func=cmd_report)

    p_timing = sub.add_parser("timing", help="Per-criterion timing only")
    p_timing.add_argument("--exp_path", required=True, help="Run dir containing {scene}/logger/")
    p_timing.add_argument("--scene", required=True, help="Scene name, e.g. office0")
    p_timing.add_argument("--criteria", nargs="+", default=None,
                          help="Criterion names (default: uses CRITERIA from reader.py)")
    p_timing.set_defaults(func=cmd_timing)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
