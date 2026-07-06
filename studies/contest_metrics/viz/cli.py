"""CLI: per-signal contest Tier2 figures into <exp>/<scene>/fusion/contest/figures/.

Usage:
  python -m studies.contest_metrics.viz scene --exp <id> --scene <name> [--ext svg] [--out-dir DIR] [--no-derived]
  python -m studies.contest_metrics.viz exp   --exp <id>                 [--ext svg] [--no-derived]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from .loader import OUTPUT_ROOT, load_scene, discover_scenes
from .figures_mpl import save_scene_figures


def _resolve_exp_path(exp_id: str) -> pathlib.Path:
    """Accept a full path, a short experiment ID, or a unique prefix under OUTPUT_ROOT."""
    p = pathlib.Path(exp_id)
    if p.is_dir():
        return p.resolve()
    candidate = OUTPUT_ROOT / exp_id
    if candidate.is_dir():
        return candidate
    matches = sorted(OUTPUT_ROOT.glob(f"{exp_id}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous ID '{exp_id}' matches {len(matches)} experiments:", file=sys.stderr)
        for m in matches[:10]:
            print(f"  {m.name}", file=sys.stderr)
        sys.exit(1)
    raise FileNotFoundError(f"Experiment '{exp_id}' not found under {OUTPUT_ROOT}")


def _figures_dir(exp_path: pathlib.Path, scene: str) -> pathlib.Path:
    return exp_path / scene / "fusion" / "contest" / "figures"


def _run_scene(exp_path: pathlib.Path, scene: str, args: argparse.Namespace) -> None:
    data = load_scene(exp_path, scene)
    out_dir = pathlib.Path(args.out_dir) if getattr(args, "out_dir", None) else _figures_dir(exp_path, scene)
    saved = save_scene_figures(data, out_dir, ext=args.ext, derived=not args.no_derived,
                               derivative=args.derivative)
    print(f"[{scene}] {len(saved)} figures -> {out_dir}")
    for p in saved:
        print(f"  {p.name}")


def cmd_scene(args: argparse.Namespace) -> None:
    exp_path = _resolve_exp_path(args.exp)
    _run_scene(exp_path, args.scene, args)


def cmd_exp(args: argparse.Namespace) -> None:
    exp_path = _resolve_exp_path(args.exp)
    scenes = discover_scenes(exp_path)
    if not scenes:
        print(f"No scenes with contest logs in {exp_path}", file=sys.stderr)
        sys.exit(1)
    for scene in scenes:
        _run_scene(exp_path, scene, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Contest Tier2 telemetry figures")
    sub = parser.add_subparsers(dest="command")

    p_scene = sub.add_parser("scene", help="Per-signal figures for one scene")
    p_scene.add_argument("--exp", required=True, help="Experiment ID or path")
    p_scene.add_argument("--scene", required=True, help="Scene name (e.g. office0)")
    p_scene.add_argument("--ext", default="svg", help="Figure format (svg/pdf/png)")
    p_scene.add_argument("--out-dir", help="Override output dir (default: <exp>/<scene>/fusion/contest/figures)")
    p_scene.add_argument("--no-derived", action="store_true", help="Skip derived rate figures")
    p_scene.add_argument("--derivative", action="store_true", help="Also write a signal+Δ/KF figure per signal")

    p_exp = sub.add_parser("exp", help="Per-signal figures for every scene in an experiment")
    p_exp.add_argument("--exp", required=True, help="Experiment ID or path")
    p_exp.add_argument("--ext", default="svg", help="Figure format (svg/pdf/png)")
    p_exp.add_argument("--no-derived", action="store_true", help="Skip derived rate figures")
    p_exp.add_argument("--derivative", action="store_true", help="Also write a signal+Δ/KF figure per signal")

    args = parser.parse_args()
    if args.command == "scene":
        cmd_scene(args)
    elif args.command == "exp":
        cmd_exp(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
