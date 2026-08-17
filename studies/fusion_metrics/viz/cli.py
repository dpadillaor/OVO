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

from .loader import (OUTPUT_ROOT, load_scene, load_experiment, load_experiments,
                     pool_by_criterion, pool_by_criterion_by_epoch, pool_verdicts,
                     pool_counts_by_epoch)
from .dashboards import scene_dashboard, experiment_summary, compare_dashboard
from .charts import (gate_sankey, confusion_donut, confusion_donut_by_epoch,
                     confusion_donut_from_counts)
from .figures_mpl import (save_pr_trajectory, save_pr_trajectory_by_epoch,
                          save_pr_trajectory_epochs_separate, save_cost_scatter,
                          save_pr_trajectory_pooled, save_cost_scatter_pooled)

_EPOCH_ORDER = {"predrift_predrift": 0, "predrift_postdrift": 1, "postdrift_postdrift": 2}


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


def _write_fig(fig, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".html":
        fig.write_html(out)
    else:
        fig.write_image(out)  # SVG/PNG/PDF via kaleido
    print(f"Saved: {out}")


def cmd_sankey(args: argparse.Namespace) -> None:
    exp_path = _resolve_exp_path(args.exp)

    # No --scene: experiment-level pooled cascade (micro-average over all scenes).
    if not args.scene:
        scenes = load_experiment(exp_path)
        figures = exp_path / "fusion_LC" / "figures"

        if args.by_epoch:
            pooled = pool_by_criterion_by_epoch(scenes)
            epochs = sorted(pooled.items(), key=lambda kv: _EPOCH_ORDER.get(kv[0], 9))
            if not epochs:
                print(f"No by_epoch data to pool for {args.exp}", file=sys.stderr)
                return
            ext = pathlib.Path(args.out).suffix if args.out else ".pdf"
            for ep, bc in epochs:
                fig = gate_sankey(bc, title=f"Fusion gate cascade · pooled · {ep} "
                                            f"({len(scenes)} scenes)")
                _write_fig(fig, figures / f"fusion_gate_cascade_pooled_{ep}{ext}")
            return

        by_criterion = pool_by_criterion(scenes)
        if not by_criterion:
            print(f"No by_criterion data to pool for {args.exp}", file=sys.stderr)
            return
        fig = gate_sankey(by_criterion,
                          title=f"Fusion gate cascade · pooled ({len(scenes)} scenes)")
        _write_fig(fig, pathlib.Path(args.out or figures / "fusion_gate_cascade_pooled.pdf"))
        return

    data = load_scene(exp_path, args.scene)
    figures = exp_path / args.scene / "fusion" / "fusion_LC" / "figures"

    if args.by_epoch:
        epochs = sorted(data.by_epoch.items(), key=lambda kv: _EPOCH_ORDER.get(kv[0], 9))
        if not epochs:
            print(f"No by_epoch data for {args.exp}/{args.scene}", file=sys.stderr)
            return
        ext = pathlib.Path(args.out).suffix if args.out else ".pdf"
        for ep, d in epochs:
            fig = gate_sankey(d.get("by_criterion", []), title=f"Fusion gate cascade · {ep}")
            _write_fig(fig, figures / f"fusion_gate_cascade_{ep}{ext}")
        return

    fig = gate_sankey(data.by_criterion)
    _write_fig(fig, pathlib.Path(args.out or figures / "fusion_gate_cascade.pdf"))


def cmd_confusion(args: argparse.Namespace) -> None:
    exp_path = _resolve_exp_path(args.exp)

    # No --scene: experiment-level pooled donut (micro-average over all scenes).
    if not args.scene:
        scenes = load_experiment(exp_path)
        figures = exp_path / "fusion_LC" / "figures"
        n = len(scenes)
        if args.by_epoch:
            fig = confusion_donut_by_epoch(pool_counts_by_epoch(scenes))
            out = pathlib.Path(args.out or figures / "fusion_confusion_donut_pooled_by_epoch.pdf")
        else:
            fig = confusion_donut_from_counts(
                pool_verdicts(scenes), title=f"Confusion split · pooled ({n} scenes)")
            out = pathlib.Path(args.out or figures / "fusion_confusion_donut_pooled.pdf")
        _write_fig(fig, out)
        return

    data = load_scene(exp_path, args.scene)
    figures = exp_path / args.scene / "fusion" / "fusion_LC" / "figures"
    if args.by_epoch:
        fig = confusion_donut_by_epoch(data.by_epoch)
        out = pathlib.Path(args.out or figures / "fusion_confusion_donut_by_epoch.pdf")
    else:
        fig = confusion_donut(data)
        out = pathlib.Path(args.out or figures / "fusion_confusion_donut.pdf")
    _write_fig(fig, out)


def cmd_cost(args: argparse.Namespace) -> None:
    exp_path = _resolve_exp_path(args.exp)
    default_name = {"decomp": "gate_cost_scatter", "unit-total": "gate_cost_unit_total",
                    "attribution": "gate_cost_attribution",
                    "total-pairs": "gate_cost_total_pairs"}[args.view]
    if args.scale == "auto":
        log = args.view != "total-pairs"  # total-pairs reads better linear; rest log
    else:
        log = args.scale == "log"

    # No --scene: experiment-level pooled cost (additive totals, micro-average unit cost).
    if not args.scene:
        scenes = load_experiment(exp_path)
        out = args.out or str(exp_path / "fusion_LC" / "figures" / f"{default_name}_pooled.pdf")
        saved = save_cost_scatter_pooled(exp_path, scenes, out, view=args.view, log=log)
        print(f"Saved: {saved}")
        return

    data = load_scene(exp_path, args.scene)
    out = args.out or str(exp_path / args.scene / "fusion" / "fusion_LC" / "figures" / f"{default_name}.pdf")
    saved = save_cost_scatter(data, exp_path, out, view=args.view, log=log)
    print(f"Saved: {saved}")


def cmd_trajectory(args: argparse.Namespace) -> None:
    exp_path = _resolve_exp_path(args.exp)
    ext = pathlib.Path(args.out).suffix if args.out else ".pdf"

    # scene-level
    if args.scene:
        data = load_scene(exp_path, args.scene)
        figures = exp_path / args.scene / "fusion" / "fusion_LC" / "figures"
        if args.by_epoch:
            # combined (all epochs, one plane) + one separate figure per epoch
            combined = save_pr_trajectory_by_epoch(data, args.out or figures / f"pr_trajectory_by_epoch{ext}")
            separates = save_pr_trajectory_epochs_separate(data, figures, ext=ext)
            for p in (combined, *separates):
                print(f"Saved: {p}")
        else:
            saved = save_pr_trajectory([data], args.out or figures / f"pr_trajectory{ext}")
            print(f"Saved: {saved}")
        return

    # experiment-level: bold pooled micro-average line + faint per-scene context.
    scenes = load_experiment(exp_path)
    figures = exp_path / "fusion_LC" / "figures"
    n = len(scenes)
    if args.by_epoch:
        pooled = pool_by_criterion_by_epoch(scenes)
        epochs = sorted(pooled.items(), key=lambda kv: _EPOCH_ORDER.get(kv[0], 9))
        if not epochs:
            print(f"No by_epoch data to pool for {args.exp}", file=sys.stderr)
            return
        for ep, pooled_bc in epochs:
            scene_series = [(s.scene, s.by_epoch[ep].get("by_criterion", []))
                            for s in scenes if ep in s.by_epoch]
            saved = save_pr_trajectory_pooled(
                pooled_bc, scene_series, figures / f"pr_trajectory_pooled_{ep}{ext}",
                title=f"P-R · pooled · {ep} ({n} scenes)", pooled_label=ep)
            print(f"Saved: {saved}")
        return

    pooled_bc = pool_by_criterion(scenes)
    scene_series = [(s.scene, s.by_criterion) for s in scenes]
    saved = save_pr_trajectory_pooled(
        pooled_bc, scene_series, args.out or figures / f"pr_trajectory_pooled{ext}",
        title=f"P-R trajectory · pooled ({n} scenes)")
    print(f"Saved: {saved}")


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

    p_sankey = sub.add_parser("sankey", help="Gate cascade Sankey (scene, or pooled over the experiment if --scene omitted)")
    p_sankey.add_argument("--exp", required=True, help="Experiment ID or path")
    p_sankey.add_argument("--scene", help="Scene name (e.g. office0). Omit for experiment-level pooled cascade.")
    p_sankey.add_argument("--out", help="Output path (.svg/.pdf/.png/.html)")
    p_sankey.add_argument("--by-epoch", action="store_true",
                          help="One Sankey per drift epoch (separate files)")

    p_cost = sub.add_parser("cost", help="Gate cost decomposition (scene, or pooled over the experiment if --scene omitted)")
    p_cost.add_argument("--exp", required=True, help="Experiment ID or path")
    p_cost.add_argument("--scene", help="Scene name (e.g. office0). Omit for experiment-level pooled cost.")
    p_cost.add_argument("--out", help="Output path (.svg/.pdf/.png)")
    p_cost.add_argument("--view", default="decomp",
                        choices=["decomp", "unit-total", "attribution", "total-pairs"],
                        help="decomp = pairs×unit (bubble=total); unit-total = ms/pair vs total s; "
                             "attribution = total-time bars per gate")
    p_cost.add_argument("--scale", default="auto", choices=["auto", "log", "linear"],
                        help="axis scale for scatter views (auto: total-pairs linear, rest log)")

    p_conf = sub.add_parser("confusion", help="Confusion-split donut (scene, or pooled over the experiment if --scene omitted)")
    p_conf.add_argument("--exp", required=True, help="Experiment ID or path")
    p_conf.add_argument("--scene", help="Scene name (e.g. office0). Omit for experiment-level pooled donut.")
    p_conf.add_argument("--out", help="Output path (.svg/.pdf/.png/.html)")
    p_conf.add_argument("--by-epoch", action="store_true",
                        help="Small-multiples: one donut per drift epoch")

    p_traj = sub.add_parser("trajectory", help="P-R trajectory, one polyline per scene")
    p_traj.add_argument("--exp", required=True, help="Experiment ID or path")
    p_traj.add_argument("--scene", help="Scene name (required with --by-epoch)")
    p_traj.add_argument("--out", help="Output path (.svg/.pdf/.png, matplotlib)")
    p_traj.add_argument("--by-epoch", action="store_true",
                        help="One polyline per drift epoch for a single scene")

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
    elif args.command == "sankey":
        cmd_sankey(args)
    elif args.command == "cost":
        cmd_cost(args)
    elif args.command == "confusion":
        cmd_confusion(args)
    elif args.command == "trajectory":
        cmd_trajectory(args)
    elif args.command == "exp":
        cmd_exp(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
