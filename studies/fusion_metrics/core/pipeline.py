"""Evaluation pipeline: score a run's real fusion merge/split decisions against GT.

Reads ``{exp_path}/{scene}/fusion_decisions.csv`` (the machine's calls) and the
matching pre-fusion checkpoint (the instances as fusion saw them), scores every
pair as TP/FP/FN/TN, and writes three artifacts to ``{exp_path}/{scene}/fusion/fusion_LC/``:
``fusion_decisions_eval.csv`` (rows + verdict), ``fusion_eval_summary.json``
(counts, rates, by-group breakdown) and ``fusion_instance_stats.csv``.

Pure orchestration — no argparse. The CLI (``python -m cli eval``) is the entry.
"""
from __future__ import annotations

import sys
import json
import pathlib
from collections import defaultdict

import yaml

from .agnostic_impact.loaders import (
    load_fusion_decisions,
    parse_fusion_decision,
    load_pre_fusion_scene,
    load_ovo_map,
    reproject_ids_to_gt,
    load_drift_inputs,
)
from .agnostic_impact.merge_decision_eval import (
    evaluate_decision,
    summarize,
    epoch_by_obj_id,
    summarize_by_epoch,
)
from .agnostic_impact.fusion_agnostic_impact import (
    fusion_impact,
    per_instance_stats,
    PRODUCTION_MIN_REGION_SIZE,
)
from .agnostic_impact.writers import write_pairs_csv, write_summary_json, write_instance_stats_csv

# Repo root (studies/fusion_metrics/core/pipeline.py -> OVO), anchors default data paths.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_MESH_ROOT = REPO_ROOT / "data" / "input" / "Datasets" / "Replica"
DEFAULT_GT_ROOT = DEFAULT_MESH_ROOT / "instance_gt"


def _own_ckpt(exp_path: pathlib.Path, scene: str) -> pathlib.Path:
    """The run's own checkpoint: mirror data/output -> data/checkpoints."""
    try:
        rel = exp_path.relative_to(REPO_ROOT / "data" / "output")
    except ValueError:
        rel = pathlib.Path(exp_path.name)
    return REPO_ROOT / "data" / "checkpoints" / rel / scene / "pre_fusion.ckpt"


def _config_ckpt(exp_path: pathlib.Path, scene: str) -> pathlib.Path | None:
    """Borrowed checkpoint: the run's config may restore another run's pre_fusion.

    A run launched from a shared checkpoint records it under
    ``restore_pre_fusion_checkpoint`` (``{scene}`` placeholder, repo-relative).
    """
    cfg_path = exp_path / scene / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        ref = (yaml.safe_load(f) or {}).get("restore_pre_fusion_checkpoint")
    if not ref:
        return None
    ref = str(ref).replace("{scene}", scene)
    p = pathlib.Path(ref)
    return p if p.is_absolute() else REPO_ROOT / p


# Canonical criterion order (mirror ovo/entities/fusion/factory.py build order).
# overlap / overlap_old are terminal (mutually exclusive in a real chain).
_CANONICAL_CRITERIA = ("cooccurrence", "centroid", "aabb", "cos_sim", "overlap", "overlap_old")


def _fusion_chain(exp_path: pathlib.Path, scene: str, observed_reasons: set[str]) -> list[str] | None:
    """Ordered criterion chain for the cascade view.

    Prefers the run's explicit ``fusion_criteria`` (config.yaml). Otherwise infers
    it from the reject reasons actually present in the CSV, ordered canonically --
    robust to configs that omit ``fusion_method``/``fusion_criteria`` and to the
    overlap-vs-overlap_old choice (the reason strings are the source of truth).
    A criterion that never rejected simply won't appear (its cascade row would be
    a trivial pass-through). None if nothing to build.
    """
    cfg_path = exp_path / scene / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            chain = (yaml.safe_load(f) or {}).get("fusion_criteria")
        if chain:
            return list(chain)
    inferred = [c for c in _CANONICAL_CRITERIA if c in observed_reasons]
    return inferred or None


def _resolve_ckpt(exp_path: pathlib.Path, scene: str) -> pathlib.Path:
    """Find the pre-fusion checkpoint: the run's own, else the one its config
    borrowed (shared across runs), else raise listing both tried paths."""
    own = _own_ckpt(exp_path, scene)
    if own.exists():
        return own
    borrowed = _config_ckpt(exp_path, scene)
    if borrowed is not None and borrowed.exists():
        print(f"Using checkpoint from config restore_pre_fusion_checkpoint: {borrowed}")
        return borrowed
    tried = f"\n  own:    {own}" + (f"\n  config: {borrowed}" if borrowed else "")
    raise FileNotFoundError(
        f"No pre-fusion checkpoint for '{scene}'. Tried:{tried}\n"
        f"Pass ckpt explicitly, or check the run saved one "
        f"(jump_drift_enabled AND save_pre_fusion_checkpoint)."
    )


def discover_scenes(exp_path: pathlib.Path) -> list[str]:
    """Scene subdirs = those holding a ``config.yaml`` (utility dirs don't)."""
    return sorted(
        d.name for d in exp_path.iterdir()
        if d.is_dir() and (d / "config.yaml").exists()
    )


def fusion_decisions_csv(scene_dir: pathlib.Path) -> pathlib.Path:
    """The run's fusion_decisions.csv. New runs write it under ``{scene}/fusion/``;
    older runs left it at the scene root — prefer the new spot, fall back to the old."""
    new = scene_dir / "fusion" / "fusion_decisions.csv"
    return new if new.exists() else scene_dir / "fusion_decisions.csv"


def check_scene(exp_path: pathlib.Path, scene: str, ckpt: str | None,
                mesh_root: str | pathlib.Path, gt_root: str | pathlib.Path) -> str | None:
    """Return ``None`` if the scene has everything to evaluate, else why not."""
    scene_dir = exp_path / scene
    if not scene_dir.is_dir():
        return "scene dir not found"
    if not fusion_decisions_csv(scene_dir).exists():
        return "no fusion_decisions.csv"
    if not (scene_dir / "ovo_map.ckpt").exists():
        return "no ovo_map.ckpt (post-fusion map)"
    if ckpt:
        ckpt_path = pathlib.Path(ckpt)
    else:
        try:
            ckpt_path = _resolve_ckpt(exp_path, scene)
        except FileNotFoundError:
            return "no pre_fusion.ckpt (own or config)"
    if not ckpt_path.exists():
        return f"checkpoint missing: {ckpt_path}"
    if not (pathlib.Path(mesh_root) / f"{scene}_mesh.ply").exists():
        return f"no GT mesh: {scene}_mesh.ply"
    if not (pathlib.Path(gt_root) / f"{scene}.txt").exists():
        return f"no GT labels: {scene}.txt"
    return None


def evaluate_scene(exp_path: pathlib.Path, scene: str, ckpt: str | None = None,
                   mesh_root: str | pathlib.Path = DEFAULT_MESH_ROOT,
                   gt_root: str | pathlib.Path = DEFAULT_GT_ROOT,
                   out_dir: str | pathlib.Path | None = None) -> int:
    scene_dir = exp_path / scene
    csv_path = fusion_decisions_csv(scene_dir)
    ckpt_path = pathlib.Path(ckpt) if ckpt else _resolve_ckpt(exp_path, scene)
    out_dir = pathlib.Path(out_dir) if out_dir else scene_dir / "fusion" / "fusion_LC"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, frame_id = load_fusion_decisions(csv_path)
    print(f"Loaded {len(rows)} fusion decisions (frame {frame_id}).")

    scene_data, col_obj_ids, n_pre = load_pre_fusion_scene(
        ckpt_path, scene, mesh_root=str(mesh_root), gt_root=str(gt_root)
    )
    print(f"{n_pre} pre-fusion instances "
          f"({scene_data.n_pred_instances} projected onto the GT mesh).")

    # True post-fusion instance count from the final OVO map (not derived).
    post_map = load_ovo_map(exp_path, scene)
    n_post = len(post_map.instance_ids)
    print(f"{n_post} post-fusion instances (final OVO map).")

    # Production reprojects the post-fusion map onto the GT mesh to evaluate; an
    # instance whose points reach no GT vertex is dropped there but present here.
    # Track the gap so our raw count and production's evaluated count reconcile.
    post_on_mesh = reproject_ids_to_gt(post_map.obj_ids, post_map.xyz, scene_data.xyz)
    post_unmatched = sorted(set(post_map.instance_ids.tolist()) - post_on_mesh)
    if post_unmatched:
        print(f"{len(post_unmatched)} post-fusion instances drop in GT projection: "
              f"{post_unmatched}")

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
        "objects": fusion_impact(
            *ap_args,
            valid_classes=set(valid_ins_class_ids),
            min_region_size=PRODUCTION_MIN_REGION_SIZE,
        ),
    }

    # Drift-epoch split: classify every instance (pre/post the jump) from the
    # checkpoint, then break the verdicts down by the pair's drift bucket.
    observed_reasons = {d.reason for d in decisions if not d.merged and d.reason}
    chain = _fusion_chain(exp_path, scene, observed_reasons)
    if chain is None:
        print("Warning: no fusion chain resolvable; by_criterion cascade skipped.")
    try:
        drift = load_drift_inputs(ckpt_path)
        epochs = epoch_by_obj_id(drift.created_at_frame, drift.jump_frame)
        verdicts = summarize(pairs, chain=chain)
        verdicts["by_epoch"] = summarize_by_epoch(pairs, epochs, chain=chain)
    except ValueError:
        print("Warning: drift-epoch classification unavailable (segment_every != map_every).")
        verdicts = summarize(pairs, chain=chain)
        verdicts["by_epoch"] = {}

    # Cascade sanity: gate-0 eval below scored total means chain/reason mismatch.
    cascade = verdicts.get("by_criterion")
    if cascade and cascade[0]["eval"] < len(pairs):
        print(f"Warning: cascade gate-0 eval {cascade[0]['eval']} < {len(pairs)} scored "
              f"pairs; some reject reasons are absent from the chain {chain}.")

    summary = {
        "experiment": exp_path.name,
        "scene": scene,
        "frame_id": frame_id,
        "run": {
            "instances_pre": n_pre,
            "instances_post": n_post,
            "merges_applied": n_pre - n_post,
            "instances_post_reprojected": len(post_on_mesh),
            "post_unmatched_gt": post_unmatched,
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
        "verdicts": verdicts,
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


def _write_index(
    exp_path: pathlib.Path,
    evaluated: list[str],
    skipped: dict[str, str],
    failed: dict[str, str],
) -> pathlib.Path:
    """Machine-readable per-scene status, so a batch driver knows what ran.

    Written to ``<exp_path>/fusion_eval_index.json``. ``skipped`` = missing
    inputs (reason), ``failed`` = raised during evaluation (error).
    """
    index_path = exp_path / "fusion_eval_index.json"
    index = {
        "experiment": exp_path.name,
        "counts": {
            "total": len(evaluated) + len(skipped) + len(failed),
            "evaluated": len(evaluated),
            "skipped": len(skipped),
            "failed": len(failed),
        },
        "evaluated": sorted(evaluated),
        "skipped": dict(sorted(skipped.items())),
        "failed": dict(sorted(failed.items())),
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    return index_path


def evaluate_experiment(exp_path: str | pathlib.Path, scene: str | None = None,
                        ckpt: str | None = None,
                        mesh_root: str | pathlib.Path = DEFAULT_MESH_ROOT,
                        gt_root: str | pathlib.Path = DEFAULT_GT_ROOT,
                        out_dir: str | pathlib.Path | None = None) -> int:
    """Evaluate one scene (``scene`` given) or scan & evaluate every ready scene.

    Returns 0 when everything requested was evaluated, 1 otherwise (a batch driver
    can branch on it; the per-scene detail lives in ``fusion_eval_index.json``).
    """
    exp_path = pathlib.Path(exp_path).resolve()
    if not exp_path.is_dir():
        raise FileNotFoundError(f"Experiment dir not found: {exp_path}")

    # One explicit scene -> just run it. No scene -> scan the whole experiment.
    if scene:
        return evaluate_scene(exp_path, scene, ckpt, mesh_root, gt_root, out_dir)

    scenes = discover_scenes(exp_path)
    if not scenes:
        print(f"No scenes (dirs with config.yaml) found in {exp_path}")
        _write_index(exp_path, [], {}, {})
        return 1
    print(f"Found {len(scenes)} scene(s): {', '.join(scenes)}")

    ready: list[str] = []
    skipped: dict[str, str] = {}
    for s in scenes:
        reason = check_scene(exp_path, s, ckpt, mesh_root, gt_root)
        if reason is None:
            ready.append(s)
            print(f"  [OK]   {s}")
        else:
            skipped[s] = reason
            print(f"  [SKIP] {s}: {reason}")

    evaluated: list[str] = []
    failed: dict[str, str] = {}
    if ready:
        print(f"\nEvaluating {len(ready)} scene(s): {', '.join(ready)}")
        for i, s in enumerate(ready, 1):
            print(f"\n=== [{i}/{len(ready)}] {s} ===")
            try:
                evaluate_scene(exp_path, s, ckpt, mesh_root, gt_root, out_dir)
                evaluated.append(s)
            except Exception as e:  # one bad scene must not abort the batch
                failed[s] = f"{type(e).__name__}: {e}"
                print(f"  [FAIL] {s}: {failed[s]}", file=sys.stderr)

    index_path = _write_index(exp_path, evaluated, skipped, failed)
    print(f"\nDone. {len(evaluated)} evaluated, {len(skipped)} skipped, "
          f"{len(failed)} failed (of {len(scenes)}). Index: {index_path}")
    if skipped:
        print(f"  skipped: {', '.join(f'{s} ({r})' for s, r in skipped.items())}")
    if failed:
        print(f"  failed:  {', '.join(failed)}")
    return 0 if len(evaluated) == len(scenes) else 1
