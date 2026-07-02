"""CLI: scene / experiment / compare dashboards as standalone HTML files.

Usage:
  python -m studies.fusion_metrics.viz scene   --exp <id> --scene <name>  [--out <.html>] [--mode objects|all]
  python -m studies.fusion_metrics.viz exp     --exp <id>                  [--out <.html>] [--mode objects|all]
  python -m studies.fusion_metrics.viz compare --exps <id1> <id2> ...      [--out <.html>] [--mode objects|all]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from .loader import OUTPUT_ROOT, load_scene, load_experiment, load_experiments
from .dashboards import scene_dashboard, experiment_summary, compare_dashboard


def _resolve_exp_path(exp_id: str) -> pathlib.Path:
    """Accept either a full path or a short experiment ID (look up under OUTPUT_ROOT)."""
    p = pathlib.Path(exp_id)
    if p.is_dir():
        return p.resolve()
    candidate = OUTPUT_ROOT / exp_id
    if candidate.is_dir():
        return candidate
    # Try a glob match
    matches = sorted(OUTPUT_ROOT.glob(f"{exp_id}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous ID '{exp_id}' matches {len(matches)} experiments:",
              file=sys.stderr)
        for m in matches[:10]:
            print(f"  {m.name}", file=sys.stderr)
        sys.exit(1)
    raise FileNotFoundError(f"Experiment '{exp_id}' not found under {OUTPUT_ROOT}")


def cmd_scene(args: argparse.Namespace) -> None:
    exp_path = _resolve_exp_path(args.exp)
    data = load_scene(exp_path, args.scene)
    fig = scene_dashboard(data, args.mode)
    out = args.out or str(exp_path / "_viz" / f"{args.scene}_dashboard.html")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)
    print(f"Saved: {out}")


def cmd_exp(args: argparse.Namespace) -> None:
    exp_path = _resolve_exp_path(args.exp)
    scenes = load_experiment(exp_path)
    fig = experiment_summary(scenes, args.mode)
    out = args.out or str(exp_path / "_viz" / "experiment_summary.html")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)
    print(f"Saved: {out}")


def cmd_compare(args: argparse.Namespace) -> None:
    exp_paths = [_resolve_exp_path(e) for e in args.exps]
    experiments = load_experiments(exp_paths)
    fig = compare_dashboard(experiments, args.mode)
    # Default output: first experiment's _viz dir
    first = exp_paths[0]
    out = args.out or str(first / "_viz" / "compare.html")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion evaluation visualizations")
    sub = parser.add_subparsers(dest="command")

    p_scene = sub.add_parser("scene", help="Single-scene dashboard")
    p_scene.add_argument("--exp", required=True, help="Experiment ID or path")
    p_scene.add_argument("--scene", required=True, help="Scene name (e.g. office0)")
    p_scene.add_argument("--out", help="Output HTML path")
    p_scene.add_argument("--mode", default="objects", choices=["objects", "all"],
                         help="AP evaluation mode (default: objects)")

    p_exp = sub.add_parser("exp", help="Experiment summary (all scenes)")
    p_exp.add_argument("--exp", required=True, help="Experiment ID or path")
    p_exp.add_argument("--out", help="Output HTML path")
    p_exp.add_argument("--mode", default="objects", choices=["objects", "all"])

    p_cmp = sub.add_parser("compare", help="Cross-experiment comparison")
    p_cmp.add_argument("--exps", nargs="+", required=True, help="Experiment IDs or paths")
    p_cmp.add_argument("--out", help="Output HTML path")
    p_cmp.add_argument("--mode", default="objects", choices=["objects", "all"])

    args = parser.parse_args()
    if args.command == "scene":
        cmd_scene(args)
    elif args.command == "exp":
        cmd_exp(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
