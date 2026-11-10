"""Evaluate one fusion step of a run (or of a replay) against the GT: were its decisions right, and
what did they do to the class-agnostic AP? Replica only: it needs the GT mesh.

The pre-fusion map comes from the step's checkpoint (`{scene}/fusion/pre_fusion_{frame}.ckpt`, or
for a replay the checkpoint in its `replayed_from`), the decisions from the scene's
fusion_decisions.csv. Output, next to them:

    {scene}/fusion/eval_{frame:05d}/
        decisions_eval.csv    the step's rows of fusion_decisions.csv + gt_first, gt_second,
                              same_object, verdict (empty when an instance has no mask),
                              epoch_first, epoch_second (empty without simulation.json)
        summary.json          instances, pairs, verdicts, cascade, by_epoch (verdicts and cascade
                              per epoch pair, ovo.eval.fusion.epochs; null without
                              simulation.json), agnostic AP pre/post/delta
        instance_stats.csv    one row per GT instance (ovo.eval.fusion.impact.instance_stats)

    python -m ovo eval fusion --dataset Replica --experiment my_exp [--scenes office0 [--frame 1990]]
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ovo import layout
from ovo.entities.fusion import FusionCfg
from ovo.eval import benchmarks
from ovo.eval.instances import nan_to_none
from ovo.replay import checkpoint
from . import decisions, epochs, impact, projection, verdicts

FORMAT = 2  # 2: by_epoch, epoch_first, epoch_second
NEW_COLUMNS = ("gt_first", "gt_second", "same_object", "verdict", "epoch_first", "epoch_second")
STATS_COLUMNS = ("gt_id", "class_name", "iou_pre", "iou_post", "coverage_pre", "coverage_post", "change",
                 "match_50", "pred_pre", "pred_post", "fragments_pre")


@dataclass(frozen=True)
class Job:
    """One scene's evaluation, every input found."""
    scene: str
    scene_dir: Path
    frame_id: int
    checkpoint: Path
    fusion: Optional[dict]  # instance_fusion of a replay; None: the checkpoint's own
    table: decisions.Table
    last_frame: int  # the run's last fusion with judged pairs (for a replay, the source run's)
    simulation: Path  # the run's simulation.json (for a replay, the source run's); may not exist


def evaluate_experiment(dataset: str, experiment: str, scenes: Optional[list[str]] = None,
                        frame: Optional[int] = None) -> list[Path]:
    """Evaluate the last checkpointed fusion of every scene with decisions (or of `scenes`), or the
    one at `frame` (one scene only). Returns the output folders."""
    if dataset.lower() != "replica":
        raise ValueError(f"fusion evaluation needs the GT mesh and supports Replica only, got {dataset!r}")
    exp = layout.ExperimentLayout.named(dataset, experiment)
    if scenes is None:
        found = exp.root.iterdir() if exp.root.is_dir() else []
        scenes = sorted(p.name for p in found if p.is_dir() and layout.fusion_decisions(p).is_file())
    if not scenes:
        raise ValueError(f"{exp.root}: no scene has fusion decisions")
    if frame is not None and len(scenes) != 1:
        raise ValueError("--frame needs exactly one scene: each scene fuses at its own frames")

    # Find every input first: a missing one fails before any scene is written.
    jobs = [_job(dataset, scene, exp.scene(scene), frame) for scene in scenes]
    benchmark = benchmarks.for_dataset(dataset)
    out = []
    for job in jobs:
        print(f"Evaluating the fusion of {job.scene} at frame {job.frame_id}")
        out.append(evaluate_scene(job, benchmark))
    return out


def _job(dataset: str, scene: str, scene_dir: Path, frame: Optional[int]) -> Job:
    csv_path = layout.fusion_decisions(scene_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{scene}: no fusion decisions at {csv_path}")
    for path in (layout.instance_gt_dir(dataset) / f"{scene}.txt", layout.replica_mesh(scene)):
        if not path.is_file():
            raise FileNotFoundError(f"{scene}: missing GT {path}")

    table = decisions.parse(csv_path.read_text())
    ckpts, fusion, source_dir = _checkpoints(scene_dir)
    source_csv = layout.fusion_decisions(source_dir) if source_dir else None
    if not ckpts:
        raise ValueError(f"{scene_dir}: no pre-fusion checkpoint (replay.checkpoints) to evaluate from")
    if frame is not None and frame not in ckpts:
        raise ValueError(f"{scene}: no checkpoint at frame {frame}; available: {sorted(ckpts)}")
    frame_id = frame if frame is not None else max(ckpts)
    run_frames = decisions.parse(source_csv.read_text()).frames() if source_csv else table.frames()
    return Job(scene, scene_dir, frame_id, ckpts[frame_id], fusion, table, max(run_frames, default=frame_id),
               layout.simulation(source_dir or scene_dir))


def _checkpoints(scene_dir: Path) -> tuple[dict[int, Path], Optional[dict], Optional[Path]]:
    """{frame: checkpoint} of a scene; for a replay (whose checkpoint is the source run's, saved
    with the source fusion) also the fusion config it ran with and the source run's scene folder,
    whose decisions say whether the replayed fusion was the run's last and whose simulation.json
    gives the epochs."""
    config_path = scene_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    source = (config or {}).get("replayed_from")
    if source is None:
        return checkpoint.available(scene_dir), None, None
    path = Path(source["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"{scene_dir}: replayed from {path}, which no longer exists")
    source_dir = path.parent.parent  # {scene}/fusion/pre_fusion_*.ckpt
    source_csv = layout.fusion_decisions(source_dir)
    if not source_csv.is_file():
        raise FileNotFoundError(f"{scene_dir}: replayed from {path}, but its run has no decisions at {source_csv}")
    fusion = config["semantic"]["instance_fusion"]
    FusionCfg.from_dict(fusion)  # an invalid fusion raises before any scene is written
    return {int(source["frame_id"]): path}, fusion, source_dir


def evaluate_scene(job: Job, benchmark: benchmarks.InstanceBenchmark) -> Path:
    ckpt = checkpoint.load(job.checkpoint)
    if ckpt["frame_id"] != job.frame_id:
        raise ValueError(f"{job.checkpoint}: saved at frame {ckpt['frame_id']}, expected {job.frame_id}")
    fusion = job.fusion if job.fusion is not None else ckpt["config"]["semantic"]["instance_fusion"]
    cfg = FusionCfg.from_dict(fusion)
    rows = job.table.at_frame(job.frame_id)

    xyz = ckpt["map"]["xyz"].numpy()
    obj_ids = ckpt["map"]["obj_ids"].reshape(-1).numpy()
    in_map = set(np.unique(obj_ids[obj_ids >= 0]).tolist())
    order = [i for i in ckpt["ovo"]["objects"] if i in in_map]  # the pipeline's instance order

    gt_ids = np.loadtxt(layout.instance_gt_dir("Replica") / f"{job.scene}.txt", dtype=np.int64, ndmin=1)
    vertices = mesh_vertices(layout.replica_mesh(job.scene))
    if len(vertices) != len(gt_ids):
        raise ValueError(f"{job.scene}: the mesh has {len(vertices)} vertices, the GT {len(gt_ids)} ids")

    masks, ids = projection.project(xyz, obj_ids, vertices)
    merges = decisions.merges(rows, cfg.apply, order)
    dominant = verdicts.dominant_gt(masks, ids, gt_ids)
    scored = [verdicts.score(r, dominant) for r in rows]
    judged = [s for s in scored if s is not None]
    steps = [g.name for g in cfg.gates] + ["cos_sim", "overlap"]
    born = _epochs(job, ckpt["ovo"]["objects"], {i for r in rows for i in (r.first, r.second)})

    pre = impact.side(masks, ids, gt_ids, benchmark.class_ids)
    post = impact.side(*impact.post_masks(masks, ids, decisions.groups(merges)), gt_ids, benchmark.class_ids)
    unscored_ids = sorted({i for r in rows for i in (r.first, r.second) if i not in dominant})
    summary = {
        "format": FORMAT,
        "scene": job.scene,
        "frame_id": job.frame_id,
        "last_fusion": job.frame_id == job.last_frame,
        "fusion": fusion,
        "instances": {"pre": len(in_map), "post": len(in_map) - len(merges), "merges": len(merges)},
        "pairs": {"judged": len(rows), "scored": len(judged), "unscored": len(rows) - len(judged),
                  "unscored_ids": unscored_ids},
        "verdicts": verdicts.counts(judged),
        "cascade": verdicts.cascade(judged, steps),
        "by_epoch": None if born is None else _by_epoch(rows, scored, born, steps),
        "agnostic_ap": impact.agnostic_ap(gt_ids, pre, post, benchmark.class_ids),
    }
    stats = impact.instance_stats(gt_ids, pre, post, benchmark.class_ids, benchmark.class_names)

    out = layout.fusion_eval_dir(job.scene_dir, job.frame_id)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(layout.fusion_eval_decisions(job.scene_dir, job.frame_id), job.table.columns + NEW_COLUMNS,
               [{**r.cells, **_gt_cells(r, s, dominant), **_epoch_cells(r, born)} for r, s in zip(rows, scored)])
    layout.fusion_eval_summary(job.scene_dir, job.frame_id).write_text(json.dumps(nan_to_none(summary), indent=2))
    _write_csv(layout.fusion_eval_instance_stats(job.scene_dir, job.frame_id), STATS_COLUMNS, stats)
    return out


def mesh_vertices(path: Path) -> np.ndarray:
    import open3d as o3d  # only needed here

    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def _gt_cells(row: decisions.Row, scored: Optional[verdicts.Scored], dominant: dict) -> dict[str, str]:
    cell = lambda i: "" if dominant.get(i) is None else str(dominant[i])
    return {
        "gt_first": cell(row.first),
        "gt_second": cell(row.second),
        "same_object": "" if scored is None else str(scored.same_object),
        "verdict": "" if scored is None else scored.verdict,
    }


def _epochs(job: Job, objects: dict, ids: set[int]) -> Optional[dict[int, int]]:
    """{instance id: epoch} of the judged instances; None without simulation.json."""
    bounds = epochs.boundaries(job.simulation)
    if bounds is None:
        return None
    unknown = sorted(i for i in ids if objects[i]["created_at_frame"] is None)
    if unknown:
        raise ValueError(f"{job.checkpoint}: instances {unknown} have no created_at_frame")
    return {i: epochs.epoch(objects[i]["created_at_frame"], bounds) for i in ids}


def _by_epoch(rows: list[decisions.Row], scored: list[Optional[verdicts.Scored]], born: dict[int, int],
              steps: list[str]) -> dict:
    """verdicts and cascade of the scored pairs of each epoch pair, in epoch order."""
    groups: dict[tuple[int, int], list[verdicts.Scored]] = {}
    for r, s in zip(rows, scored):
        if s is not None:
            groups.setdefault(tuple(sorted((born[r.first], born[r.second]))), []).append(s)
    return {epochs.pair(*key): {"verdicts": verdicts.counts(g), "cascade": verdicts.cascade(g, steps)}
            for key, g in sorted(groups.items())}


def _epoch_cells(row: decisions.Row, born: Optional[dict[int, int]]) -> dict[str, str]:
    if born is None:
        return {"epoch_first": "", "epoch_second": ""}
    return {"epoch_first": str(born[row.first]), "epoch_second": str(born[row.second])}


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
