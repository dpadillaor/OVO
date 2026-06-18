"""Batch-evaluate a run's fusion decisions against the GT.

Reads ``{exp_path}/{scene}/fusion_decisions.csv`` (the machine's merge/split
calls) and the matching pre-fusion checkpoint (the instances as fusion saw
them), scores every pair as TP/FP/FN/TN, and writes two outputs next to the
CSV: ``fusion_decisions_eval.csv`` (rows + verdict) and
``fusion_eval_summary.json`` (counts, rates, by-group breakdown).

Example:
    python -m scripts.eval_fusion_decisions \
        --exp_path data/output/Replica/20260614_GT_CLIP_opt-aggressive-10_c6050 \
        --scene office0
"""
from __future__ import annotations

import sys
import json
import pathlib
import argparse
from collections import defaultdict

# Make the package root (studies/fusion_metrics) importable as core.*
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Repo root, used to anchor the default data paths.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

from core.loaders import (
    load_fusion_decisions,
    parse_fusion_decision,
    load_pre_fusion_scene,
    load_ovo_map,
)
from core.merge_decision_eval import evaluate_decision, summarize
from core.fusion_agnostic_impact import fusion_impact, per_instance_stats
from core.writers import write_pairs_csv, write_summary_json, write_instance_stats_csv

DEFAULT_MESH_ROOT = REPO_ROOT / "data" / "input" / "Datasets" / "Replica"
DEFAULT_GT_ROOT = DEFAULT_MESH_ROOT / "instance_gt"


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
                        help="Run dir containing {scene}/fusion_decisions.csv")
    parser.add_argument("--scene", type=str, default="office0")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pre_fusion.ckpt path (default: mirror under data/checkpoints)")
    parser.add_argument("--mesh_root", type=str, default=str(DEFAULT_MESH_ROOT),
                        help="Dir holding {scene}_mesh.ply")
    parser.add_argument("--gt_root", type=str, default=str(DEFAULT_GT_ROOT),
                        help="Dir holding instance ground-truth {scene}.txt")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write outputs (default: next to the CSV)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exp_path = pathlib.Path(args.exp_path).resolve()
    if not exp_path.is_dir():
        raise FileNotFoundError(f"Experiment dir not found: {exp_path}")
    scene_dir = exp_path / args.scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(
            f"Scene '{args.scene}' not found in experiment: {scene_dir}"
        )
    csv_path = scene_dir / "fusion_decisions.csv"
    ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else _default_ckpt(exp_path, args.scene)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else scene_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, frame_id = load_fusion_decisions(csv_path)
    print(f"Loaded {len(rows)} fusion decisions (frame {frame_id}).")

    scene_data, col_obj_ids, n_pre = load_pre_fusion_scene(
        ckpt_path, args.scene, mesh_root=args.mesh_root, gt_root=args.gt_root
    )
    print(f"{n_pre} pre-fusion instances "
          f"({scene_data.n_pred_instances} projected onto the GT mesh).")

    # True post-fusion instance count from the final OVO map (not derived).
    n_post = len(load_ovo_map(exp_path, args.scene).instance_ids)
    print(f"{n_post} post-fusion instances (final OVO map).")

    # Score each decision; drop pairs whose instances did not survive projection
    # to the GT mesh (pruned / unmatched obj_ids), tracking each skipped object.
    colset = set(col_obj_ids.tolist())
    eval_rows: list[dict[str, str]] = []
    pairs = []
    decisions = []
    merges_scored = merges_skipped = 0
    skipped_objects: dict[int, dict[str, int]] = defaultdict(lambda: {"pairs": 0, "merges": 0})
    for row in rows:
        decision = parse_fusion_decision(row)
        decisions.append(decision)
        try:
            pairs.append(evaluate_decision(scene_data, col_obj_ids, decision))
            eval_rows.append(row)
            merges_scored += decision.merged
        except ValueError:
            merges_skipped += decision.merged
            for i in (decision.i1, decision.i2):
                if i not in colset:
                    skipped_objects[i]["pairs"] += 1
                    skipped_objects[i]["merges"] += decision.merged
    pairs_skipped = len(rows) - len(pairs)
    if pairs_skipped:
        print(f"Skipped {pairs_skipped} pairs with obj_ids absent from the GT "
              f"projection: {sorted(skipped_objects)}")

    # Class-agnostic AP before vs after fusion (impact of the recorded merges),
    # both over all GT instances and over objects only (production's void handling).
    from ovo.utils.replica_ins import valid_ins_class_ids, ins_class_names
    cls_to_name = dict(zip(valid_ins_class_ids, ins_class_names))
    ap_args = (scene_data.pred_masks, col_obj_ids, decisions, scene_data.gt_ids)
    agnostic_impact = {
        "all": fusion_impact(*ap_args),
        "objects": fusion_impact(*ap_args, valid_classes=set(valid_ins_class_ids)),
    }

    summary = {
        "experiment": exp_path.name,
        "scene": args.scene,
        "frame_id": frame_id,
        "run": {
            "instances_pre": n_pre,
            "instances_post": n_post,
            "merges_applied": n_pre - n_post,
        },
        "evaluation": {
            "pairs_total": len(rows),
            "pairs_scored": len(pairs),
            "pairs_skipped": pairs_skipped,
            "merges_scored": merges_scored,
            "merges_skipped": merges_skipped,
            "skipped_objects": [
                {"obj_id": int(o), "pairs": d["pairs"], "merges": d["merges"]}
                for o, d in sorted(skipped_objects.items())
            ],
        },
        "verdicts": summarize(pairs),
        "agnostic_impact": agnostic_impact,
    }

    # Per-GT-instance breakdown: IoU/acc pre vs post + matched/spurious predictions.
    stats = per_instance_stats(scene_data.pred_masks, col_obj_ids, decisions, scene_data.gt_ids)

    out_csv = out_dir / "fusion_decisions_eval.csv"
    out_json = out_dir / "fusion_eval_summary.json"
    out_stats = out_dir / "fusion_instance_stats.csv"
    write_pairs_csv(eval_rows, pairs, out_csv)
    write_summary_json(summary, out_json)
    write_instance_stats_csv(stats, out_stats, cls_to_name)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_stats}")
    print(json.dumps(summary["verdicts"]["counts"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
