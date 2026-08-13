"""Orquestación de los experimentos: compone core (medida/poda) + viz, y escribe a disco.

Capa de I/O: carga frames, carga/libera modelos, vuelca resultados. Sin argparse (eso es cli).
La lógica pura vive en core; el pintado en viz.
"""
from __future__ import annotations

import gc
import glob
import json
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from studies.segmentation.core.segmenters import SamConfig, SamSegmenter, Sam3Segmenter, Sam3PointPredictor
from studies.segmentation.core.nms_decision import scores_from_records, evaluate
from studies.segmentation.core.profiling import profile_frame
from studies.segmentation.core.timing_stats import aggregate
from studies.segmentation.viz import (
    pipeline_steps, masks_gallery, removed_masks, decision_trace, point_ambiguity,
    segmenter_compare, timing as viz_timing, scene_timing)
from ovo.utils.segment_utils import mask2segmap

RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_DATASET = "data/input/Datasets/Replica"
RESIZE = (1200, 680)
LENSES = ("pipeline", "masks", "removed", "trace")
MODELS = ("sam2", "sam3")


def _make_segmenter(model: str, cfg: SamConfig, device: str):
    """Fábrica del segmentador AMG por modelo."""
    return Sam3Segmenter(cfg, device=device) if model == "sam3" else SamSegmenter(cfg, device=device)


def _free(obj) -> None:
    """Libera un modelo de GPU: el pico de VRAM es global al proceso, hay que aislar entre modelos."""
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_frame(dataset_root: str, scene: str, frame: int) -> tuple[np.ndarray, str]:
    """Carga y prepara un frame de Replica: (RGB uint8 HxW, ruta)."""
    path = os.path.join(dataset_root, scene, "results", f"frame{frame:06d}.jpg")
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"No existe el frame: {path}")
    img = cv2.resize(bgr, RESIZE).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), path


def _scene_frames(dataset_root: str, scene: str, every: int, limit: int | None = None) -> list[int]:
    """Índices de frames de la escena, muestreados cada `every` (= segment_every de OVO)."""
    files = sorted(glob.glob(os.path.join(dataset_root, scene, "results", "frame*.jpg")))
    idxs = [int(os.path.basename(f)[5:-4]) for f in files][::every]
    return idxs[:limit] if limit else idxs


def _write_meta(out_dir: Path, model: str, cfg: SamConfig, thr: dict, frame_path: str, counts: dict) -> None:
    meta = {
        "variant": out_dir.name, "model": model,
        "scene": out_dir.parent.parent.name, "frame": int(out_dir.parent.name.lstrip("f")),
        "frame_path": frame_path, "resize": list(RESIZE),
        "amg_params": asdict(cfg), "ovo_nms": thr, "counts": counts,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def run_frame(scene: str, frame: int, model: str, variant: str, cfg: SamConfig, thr: dict,
              lenses: tuple[str, ...], dataset_root: str, device: str) -> Path:
    """Lentes del AMG sobre un frame -> results/{scene}/f{frame}/{variant}/."""
    image, frame_path = _load_frame(dataset_root, scene, frame)
    records = _make_segmenter(model, cfg, device).segment(image)
    masks, scores = scores_from_records(records)
    bd = evaluate(masks, scores, thr["iou_thr"], thr["score_thr"], thr["inner_thr"])

    out_dir = RESULTS / scene / f"f{frame:04d}" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    if "pipeline" in lenses:
        pipeline_steps.render(image, records, bd, str(out_dir / "pipeline.png"), raw_label=model.upper())
    if "masks" in lenses:
        masks_gallery.render(image, records, str(out_dir / "masks"), bd)
    if "removed" in lenses:
        removed_masks.render(image, records, bd, str(out_dir / "removed"))
    if "trace" in lenses:
        decision_trace.render(bd, str(out_dir / "trace.md"))

    _write_meta(out_dir, model, cfg, thr, frame_path,
                {"n_raw": len(records), "n_kept": len(bd.kept), "n_removed": len(bd.removed)})
    return out_dir


def _final_segmap(model: str, cfg: SamConfig, thr: dict, image: np.ndarray, device: str) -> np.ndarray:
    records = _make_segmenter(model, cfg, device).segment(image)
    masks, scores = scores_from_records(records)
    bd = evaluate(masks, scores, thr["iou_thr"], thr["score_thr"], thr["inner_thr"])
    _, binary_maps = mask2segmap([records[v.index] for v in bd.kept], image, sort=True)
    return binary_maps


def run_compare(scene: str, frame: int, cfg: SamConfig, thr: dict, dataset_root: str, device: str) -> Path:
    """Segmap final sam2 vs sam3 lado a lado -> compare/final.png."""
    image, _ = _load_frame(dataset_root, scene, frame)
    named = {m: _final_segmap(m, cfg, thr, image, device) for m in MODELS}
    out_dir = RESULTS / scene / f"f{frame:04d}" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    segmenter_compare.render(image, named, str(out_dir / "final.png"))
    return out_dir / "final.png"


def run_timing(scene: str, frame: int, cfg: SamConfig, reps: int, dataset_root: str, device: str) -> Path:
    """Coste sam2 vs sam3 en un frame (reps veces) -> compare/timing.{png,json}."""
    image, _ = _load_frame(dataset_root, scene, frame)
    out_dir = RESULTS / scene / f"f{frame:04d}" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, blob = {}, {}
    for model in MODELS:
        seg = _make_segmenter(model, cfg, device)
        profiles = [profile_frame(seg, image, warmup=1 if r == 0 else 0) for r in range(reps)]
        stats[model] = aggregate(profiles)
        blob[model] = asdict(stats[model])
        _free(seg)
    viz_timing.render(stats, str(out_dir / "timing.png"), title=f"{scene}/f{frame} coste AMG (reps={reps})")
    (out_dir / "timing.json").write_text(json.dumps(blob, indent=2))
    return out_dir / "timing.png"


def run_scene(scene: str, cfg: SamConfig, every: int, dataset_root: str, device: str,
              limit: int | None = None) -> Path:
    """Coste sam2 vs sam3 a lo largo de una escena (cada `every` frames) -> scene/timing_scene.{png,json}.

    Justo: carga cada modelo UNA vez, warmup en el primer frame, recorre la escena, libera, y el otro.
    """
    frames = _scene_frames(dataset_root, scene, every, limit)
    series, blob = {}, {}
    for model in MODELS:
        seg = _make_segmenter(model, cfg, device)
        profs = []
        for i, fidx in enumerate(frames):
            image, _ = _load_frame(dataset_root, scene, fidx)
            profs.append(profile_frame(seg, image, warmup=1 if i == 0 else 0))  # 1er frame = warmup
        series[model] = profs
        blob[model] = {"stats": asdict(aggregate(profs)),
                       "per_frame": [{"frame": f, "total_ms": p.total_ms, "n_masks": p.n_masks}
                                     for f, p in zip(frames, profs)]}
        _free(seg)

    out_dir = RESULTS / scene / "scene"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_timing.render(frames, series, str(out_dir / "timing_scene.png"),
                        title=f"{scene}: coste por frame (cada {every}, n={len(frames)})")
    (out_dir / "timing_scene.json").write_text(json.dumps(blob, indent=2))
    return out_dir / "timing_scene.png"


def run_point(scene: str, frame: int, model: str, cfg: SamConfig, xy: tuple[int, int],
              dataset_root: str, device: str) -> Path:
    """Pincha un punto (sam2/sam3/both) -> point/pt_X_Y/{model}.png + compare.png."""
    image, _ = _load_frame(dataset_root, scene, frame)
    out_dir = RESULTS / scene / f"f{frame:04d}" / "point" / f"pt_{xy[0]:04d}_{xy[1]:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(MODELS) if model == "both" else [model]
    named = {}
    for m in models:
        predictor = Sam3PointPredictor(device=device) if m == "sam3" else SamSegmenter(cfg, device=device)
        named[m] = predictor.predict_point(image, xy)
        point_ambiguity.render(image, named[m], str(out_dir / f"{m}.png"))
    if len(named) > 1:
        point_ambiguity.render_compare(image, named, str(out_dir / "compare.png"))
        return out_dir / "compare.png"
    return out_dir / f"{model}.png"
