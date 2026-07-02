"""Interactive debugger for predicted-instance pairs on one scene.

Loads the PRE-FUSION instances from the checkpoint once (the state fusion saw),
projects them onto the GT mesh, opens a live Open3D window, then reads pairs from
the terminal. For each pair you type, the canvas updates and the terminal reports
the ground-truth verdict (same object or not) and, if you supply a decision, the
confusion-matrix outcome (TP/FP/FN/TN).

Instances are selected by their real map ``obj_id``.

Colors: red = A, blue = B; GT footprint green if A and B share their dominant GT
instance, otherwise yellow (GT of A) and orange (GT of B).

Example:
    python -m scripts.debug_merge_decisions --exp_path <experiment_dir> --scene office0 --z_max 1.5
"""

from __future__ import annotations

import sys
import time
import queue
import pathlib
import argparse
import threading

# Make the package root (studies/fusion_metrics) importable as core.* / viz.*
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Repo root, used to anchor the default data paths.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

from core.agnostic_impact.loaders import (
    SceneData,
    MapData,
    load_pre_fusion_scene,
    load_pre_fusion_map,
    obj_id_to_column,
)
from core.agnostic_impact.merge_decision_eval import classify_merge_decision
from core.agnostic_impact.fusion_agnostic_impact import instances_for_gt
from viz.debug_viz import (
    build_pair_cloud,
    build_instance_overlay_cloud,
    build_gt_inspection_cloud,
    LiveViewer,
)

# Possible confusion-matrix outcome.
VERDICT_GLOSS = {
    "TP": "correct merge",
    "FP": "over-merge (fused two different objects)",
    "FN": "over-split (kept one object apart)",
    "TN": "correct separation",
}

DEFAULT_MESH_ROOT = REPO_ROOT / "data" / "input" / "Datasets" / "Replica"
DEFAULT_GT_ROOT = REPO_ROOT / "data" / "input" / "Datasets" / "Replica" / "instance_gt"
PROMPT = "pair> "


def _default_ckpt(exp_path: pathlib.Path, scene: str) -> pathlib.Path:
    """Mirror data/output -> data/checkpoints to find the pre-fusion checkpoint."""
    try:
        rel = exp_path.relative_to(REPO_ROOT / "data" / "output")
    except ValueError:
        rel = pathlib.Path(exp_path.name)
    return REPO_ROOT / "data" / "checkpoints" / rel / scene / "pre_fusion.ckpt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Run dir (used to locate the pre-fusion checkpoint)")
    parser.add_argument("--scene", type=str, default="office0")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pre_fusion.ckpt path (default: mirror under data/checkpoints)")
    parser.add_argument("--mesh_root", type=str, default=str(DEFAULT_MESH_ROOT),
                        help="Dir holding {scene}_mesh.ply")
    parser.add_argument("--gt_root", type=str, default=str(DEFAULT_GT_ROOT),
                        help="Dir holding instance ground-truth {scene}.txt")
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--z_max", type=float, default=None,
                        help="Clip rendered cloud to z <= z_max (drop ceiling). "
                             "For Replica office0 try ~1.5")
    return parser.parse_args(argv)


def _report(obj_a: int, obj_b: int, decision: str | None, summary: dict) -> None:
    print(f"  A obj_id {obj_a} (col {summary['index_a']}): {summary['verts_a']} verts"
          f"  dominant GT {summary['dominant_gt_a']}")
    print(f"  B obj_id {obj_b} (col {summary['index_b']}): {summary['verts_b']} verts"
          f"  dominant GT {summary['dominant_gt_b']}")
    gt = "SAME" if summary["same_object"] else "DIFFERENT"
    print(f"  GT verdict: {gt} object")
    if decision is not None:
        verdict = classify_merge_decision(summary["same_object"], decision == "merge")
        print(f"  decision {decision.upper()} => {verdict.value} ({VERDICT_GLOSS[verdict]})")


def _show_instance(
    obj_id: int,
    *,
    scene: SceneData,
    map_data: MapData,
    matched,
    viewer: LiveViewer,
    z_max: float | None,
) -> None:
    """Overlay one instance's original OVO points on the GT mesh."""
    pcd, n_pts = build_instance_overlay_cloud(
        scene.xyz, map_data.xyz, map_data.obj_ids, obj_id, z_max=z_max
    )
    if n_pts == 0:
        print(f"  obj_id {obj_id} has no points in the pre-fusion map.")
        return
    viewer.show(pcd)
    projected = "yes" if obj_id in set(matched.tolist()) else "NO (lost in GT projection)"
    print(f"  obj_id {obj_id}: {n_pts} original points  | projects to GT: {projected}")


def _show_pair(
    tokens: list[str],
    *,
    scene: SceneData,
    matched,
    viewer: LiveViewer,
    z_max: float | None,
) -> None:
    """Render and score a pair: <obj_id_a> <obj_id_b> [merge|separate]."""
    try:
        obj_a, obj_b = int(tokens[0]), int(tokens[1])
    except ValueError:
        print(f"  invalid ids: {tokens[0]!r} {tokens[1]!r}")
        return
    decision = None
    if len(tokens) >= 3:
        if tokens[2] not in ("merge", "separate"):
            print(f"  decision must be 'merge' or 'separate', got {tokens[2]!r}")
            return
        decision = tokens[2]

    try:
        col_a = obj_id_to_column(matched, obj_a)
        col_b = obj_id_to_column(matched, obj_b)
    except ValueError as e:
        print(f"  {e}")
        return

    pcd, summary = build_pair_cloud(scene, col_a, col_b, z_max=z_max)
    viewer.show(pcd)
    _report(obj_a, obj_b, decision, summary)


def _resolve_gt_id(typed: int, present: set[int]) -> int:
    """Accept a full GT id, or a unique instance suffix (id % 1000)."""
    if typed in present:
        return typed
    matches = sorted(i for i in present if i % 1000 == typed)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"suffix {typed} is ambiguous: {matches}; use the full id")
    raise ValueError(f"GT id {typed} not present.")


def _show_gt(
    tokens: list[str],
    *,
    scene: SceneData,
    matched,
    viewer: LiveViewer,
    z_max: float | None,
) -> None:
    """Inspect one GT instance: matched prediction (green) + spurious fragments."""
    try:
        typed = int(tokens[1])
    except (IndexError, ValueError):
        print(f"  usage: gt <gt_id|instance_suffix>")
        return
    try:
        gt_id = _resolve_gt_id(typed, set(scene.gt_ids[scene.gt_ids > 0].tolist()))
    except ValueError as e:
        print(f"  {e}")
        return
    try:
        matched_col, spurious_cols = instances_for_gt(scene.pred_masks, scene.gt_ids, gt_id)
    except ValueError as e:
        print(f"  {e}")
        return

    pcd = build_gt_inspection_cloud(scene, gt_id, matched_col, spurious_cols, z_max=z_max)
    viewer.show(pcd)
    won = matched[matched_col] if matched_col is not None else None
    spurious_ids = [int(matched[c]) for c in spurious_cols]
    print(f"  GT {gt_id}: matched pred obj_id {won}  |  {len(spurious_cols)} spurious "
          f"fragments obj_ids {spurious_ids}")


def _list_gts(scene: SceneData) -> None:
    """List GT instances one per line: ``<id>  <name>``, object vs background."""
    from ovo.utils.replica_ins import valid_ins_class_ids, ins_class_names

    cls_to_name = dict(zip(valid_ins_class_ids, ins_class_names))
    ids = sorted(set(scene.gt_ids[scene.gt_ids > 0].tolist()))
    objs = [i for i in ids if i // 1000 in cls_to_name]
    bg = [i for i in ids if i // 1000 not in cls_to_name]

    print(f"  {len(objs)} object instances (type the id):")
    for i in objs:
        print(f"    {i:>6}  {cls_to_name[i // 1000]}")
    print(f"  {len(bg)} background (class not in valid_ins):")
    for i in bg:
        print(f"    {i:>6}  class {i // 1000}")


def _print_help() -> None:
    """List every REPL command."""
    print("\nCommands:")
    print("  <obj_id_a> <obj_id_b> [merge|separate]   score a pair")
    print("  show <obj_id>                            overlay its original points on GT")
    print("  gt <gt_id>                               inspect a GT: matched (green) + spurious")
    print("  gts                                      list GT instance ids (object vs background)")
    print("  missing                                  list instances lost in GT projection")
    print("  help                                     show this list")
    print("  q                                        quit")


def _handle_command(
    line: str,
    *,
    scene: SceneData,
    map_data: MapData,
    matched,
    missing: list[int],
    viewer: LiveViewer,
    z_max: float | None,
) -> None:
    """Parse and execute one REPL command. Errors are reported, not raised."""
    tokens = line.split()
    head = tokens[0].lower()

    if head in ("help", "h", "?"):
        _print_help()
        return
    if head == "missing":
        print(f"  {len(missing)} instances lost in GT projection: {missing}")
        return
    if head == "gts":
        _list_gts(scene)
        return
    if head == "gt":
        _show_gt(tokens, scene=scene, matched=matched, viewer=viewer, z_max=z_max)
        return
    if head == "show":
        if len(tokens) < 2:
            print("  usage: show <obj_id>")
            return
        try:
            obj_id = int(tokens[1])
        except ValueError:
            print(f"  invalid id: {tokens[1]!r}")
            return
        _show_instance(obj_id, scene=scene, map_data=map_data, matched=matched,
                       viewer=viewer, z_max=z_max)
        return

    if len(tokens) < 2:
        print("  usage: <obj_id_a> <obj_id_b> [merge|separate] | show <id> | missing")
        return
    _show_pair(tokens, scene=scene, matched=matched, viewer=viewer, z_max=z_max)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exp_path = pathlib.Path(args.exp_path).resolve()
    ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else _default_ckpt(exp_path, args.scene)

    print("Loading pre-fusion instances from checkpoint ...")
    scene, matched, n_pre = load_pre_fusion_scene(
        ckpt_path, args.scene, mesh_root=args.mesh_root, gt_root=args.gt_root
    )
    map_data = load_pre_fusion_map(ckpt_path)
    missing = sorted(set(map_data.instance_ids.tolist()) - set(matched.tolist()))
    print(f"Loaded {scene.n_points} vertices, {n_pre} pre-fusion instances "
          f"({scene.n_pred_instances} projected onto the GT mesh, {len(missing)} lost).")

    viewer = LiveViewer(point_size=args.point_size,
                        window_name=f"{args.scene} pair debug")

    # Read terminal lines in a background thread so the render loop stays live.
    commands: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        for raw in sys.stdin:
            commands.put(raw.strip())
        commands.put(None)  # EOF -> quit

    threading.Thread(target=_reader, daemon=True).start()

    _print_help()
    print(PROMPT, end="", flush=True)

    running = True
    while running:
        try:
            line = commands.get_nowait()
        except queue.Empty:
            if not viewer.tick():  # window closed by the user
                break
            time.sleep(0.02)
            continue

        if line is None or line.lower() in ("q", "quit", "exit"):
            running = False
            continue
        if line:
            _handle_command(line, scene=scene, map_data=map_data, matched=matched,
                            missing=missing, viewer=viewer, z_max=args.z_max)
        print(PROMPT, end="", flush=True)

    viewer.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
