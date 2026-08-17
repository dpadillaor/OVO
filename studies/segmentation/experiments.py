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
    segmenter_compare, timing as viz_timing, scene_timing, prompt_grid)
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


def _kept_segmap(records: list[dict], thr: dict, image: np.ndarray) -> np.ndarray:
    """Máscaras crudas -> capas finales (binary_maps) tras la poda de OVO."""
    masks, scores = scores_from_records(records)
    bd = evaluate(masks, scores, thr["iou_thr"], thr["score_thr"], thr["inner_thr"])
    _, binary_maps = mask2segmap([records[v.index] for v in bd.kept], image, sort=True)
    return binary_maps


def run_compare(scene: str, frame: int, cfg: SamConfig, thr: dict, dataset_root: str, device: str) -> Path:
    """Segmap final sam2 vs sam3 lado a lado -> compare/final.png."""
    image, _ = _load_frame(dataset_root, scene, frame)
    named = {m: _kept_segmap(_make_segmenter(m, cfg, device).segment(image), thr, image) for m in MODELS}
    out_dir = RESULTS / scene / f"f{frame:04d}" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    segmenter_compare.render(image, named, str(out_dir / "final.png"))
    return out_dir / "final.png"


def run_segmap(scene: str, frame: int, model: str, variant: str, cfg: SamConfig, thr: dict,
               dataset_root: str, device: str, dim: float = 0.3, alpha: float = 0.3,
               bg: str = "frame", raw: bool = False, style: str = "fill",
               number: bool = False) -> Path:
    """Segmap de un modelo, imagen sola -> segmap/{variant}.png.

    raw=False: máscaras finales (tras el mask_nms de OVO). raw=True: crudas del AMG, sin poda OVO.
    style: fill (relleno), contour (solo bordes, revela solapes), heat (nº de máscaras por píxel).
    """
    image, _ = _load_frame(dataset_root, scene, frame)
    records = _make_segmenter(model, cfg, device).segment(image)
    if raw:
        _, binary_maps = mask2segmap(records, image, sort=True)
    else:
        binary_maps = _kept_segmap(records, thr, image)
    out_dir = RESULTS / scene / f"f{frame:04d}" / "segmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{variant}.png"
    if style in ("removed", "pairs"):
        masks, scores = scores_from_records(records)
        bd = evaluate(masks, scores, thr["iou_thr"], thr["score_thr"], thr["inner_thr"])
        if style == "pairs":
            pairs = []
            for v in bd.removed:
                killer = v.kills[0].killer_index if v.kills else None
                kil = records[killer]["segmentation"] if killer is not None else None
                pairs.append((records[v.index]["segmentation"], kil))
            rgb = segmenter_compare.removed_pairs(image, pairs, dim)
        else:
            kept = np.stack([records[v.index]["segmentation"] for v in bd.kept]) if bd.kept else np.empty((0,) + image.shape[:2])
            rem = np.stack([records[v.index]["segmentation"] for v in bd.removed]) if bd.removed else np.empty((0,) + image.shape[:2])
            rgb = segmenter_compare.removed_overlay(image, kept, rem, dim)
    elif style == "contour":
        rgb = segmenter_compare.contours(image, binary_maps, dim)
    elif style == "heat":
        rgb = segmenter_compare.heat(image, binary_maps, dim)
    else:
        rgb = segmenter_compare.colored(image, binary_maps, dim, alpha, bg)
    if number:
        rgb = segmenter_compare.number_masks(rgb, binary_maps)
    _save_rgb(rgb, out)
    return out


def run_prompts(scene: str, frame: int, model: str, cfg: SamConfig,
                dataset_root: str, device: str, dim: float = 0.75) -> Path:
    """Frame con toda la rejilla de puntos que el AMG pincha como prompts -> segmap/prompts.png."""
    image, _ = _load_frame(dataset_root, scene, frame)
    seg = _make_segmenter(model, cfg, device)
    grid = np.concatenate(seg._amg.point_grids, axis=0)  # rejilla normalizada [0,1] (todas las capas)
    out_dir = RESULTS / scene / f"f{frame:04d}" / "segmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "prompts.png"
    prompt_grid.render(image, grid, str(out), dim=dim)
    return out


def _label(img_rgb: np.ndarray, text: str) -> np.ndarray:
    """Escribe una etiqueta arriba-izquierda sobre una imagen RGB."""
    out = img_rgb.copy()
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_timing(scene: str, frame: int, cfg: SamConfig, reps: int, dataset_root: str, device: str) -> Path:
    """Coste sam2 vs sam3 en un frame (reps veces) -> compare/timing.{png,json}."""
    image, _ = _load_frame(dataset_root, scene, frame)
    out_dir = RESULTS / scene / f"f{frame:04d}" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, blob = {}, {}
    for model in MODELS:
        seg = _make_segmenter(model, cfg, device)
        profiles = [profile_frame(seg, image, warmup=1 if r == 0 else 0)[0] for r in range(reps)]
        stats[model] = aggregate(profiles)
        blob[model] = asdict(stats[model])
        _free(seg)
    viz_timing.render(stats, str(out_dir / "timing.png"), title=f"{scene}/f{frame} coste AMG (reps={reps})")
    (out_dir / "timing.json").write_text(json.dumps(blob, indent=2))
    return out_dir / "timing.png"


def run_scene(scene: str, cfg: SamConfig, thr: dict, every: int, dataset_root: str, device: str,
              limit: int | None = None, save_frames: bool = False) -> Path:
    """Coste sam2 vs sam3 a lo largo de una escena (cada `every` frames) -> scene/timing_scene.{png,json}.

    Justo: carga cada modelo UNA vez, warmup en el primer frame, recorre la escena, libera, y el otro.
    Con save_frames, cada modelo guarda su segmap por frame en su pasada; al final se juntan lado a lado.
    """
    frames = _scene_frames(dataset_root, scene, every, limit)
    out_dir = RESULTS / scene / "scene"
    out_dir.mkdir(parents=True, exist_ok=True)

    series, coverage, blob = {}, {}, {}
    for model in MODELS:
        seg = _make_segmenter(model, cfg, device)
        profs, covs = [], []
        for i, fidx in enumerate(frames):
            image, _ = _load_frame(dataset_root, scene, fidx)
            prof, records = profile_frame(seg, image, warmup=1 if i == 0 else 0)  # 1er frame = warmup
            profs.append(prof)
            binary_maps = _kept_segmap(records, thr, image)  # máscaras finales (tras poda OVO)
            covs.append(_coverage(binary_maps, image.shape[:2]))
            if save_frames:  # guardar el segmap coloreado (fuera de la medida, no contamina total_ms)
                _save_rgb(segmenter_compare.colored(image, binary_maps),
                          out_dir / "frames" / model / f"f{fidx:06d}.png")
        series[model], coverage[model] = profs, covs
        blob[model] = {"stats": asdict(aggregate(profs)),
                       "per_frame": [{"frame": f, "total_ms": p.total_ms, "n_masks": p.n_masks,
                                      "coverage": c}
                                     for f, p, c in zip(frames, profs, covs)]}
        _free(seg)

    scene_timing.render(frames, series, str(out_dir / "timing_scene.png"), coverage=coverage,
                        title=f"{scene}: coste, máscaras y cobertura por frame (cada {every}, n={len(frames)})")
    (out_dir / "timing_scene.json").write_text(json.dumps(blob, indent=2))
    if save_frames:
        _join_frames(scene, frames, out_dir, dataset_root)
    return out_dir / "timing_scene.png"


def _coverage(binary_maps: np.ndarray, shape: tuple[int, int]) -> float:
    """Fracción de píxeles cubiertos por al menos una máscara final (unión / total)."""
    if len(binary_maps) == 0:
        return 0.0
    return float(np.any(binary_maps.astype(bool), axis=0).sum()) / (shape[0] * shape[1])


def _save_rgb(img_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def _join_frames(scene: str, frames: list[int], out_dir: Path, dataset_root: str) -> None:
    """Junta original | sam2 | sam3 por frame en scene/compare/ (cv2, barato, sin re-segmentar)."""
    comp_dir = out_dir / "compare"
    comp_dir.mkdir(parents=True, exist_ok=True)
    for fidx in frames:
        orig, _ = _load_frame(dataset_root, scene, fidx)
        cols = [_label(orig, "original")]
        for model in MODELS:
            p = out_dir / "frames" / model / f"f{fidx:06d}.png"
            seg = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            cols.append(_label(seg, model))
        row = np.hstack(cols)
        _save_rgb(row, comp_dir / f"f{fidx:06d}.png")


def run_point(scene: str, frame: int, model: str, cfg: SamConfig, xy: tuple[int, int],
              dataset_root: str, device: str) -> Path:
    """Pincha un punto (sam2/sam3/both) -> point/pt_X_Y/{model}.png + compare.png."""
    image, _ = _load_frame(dataset_root, scene, frame)
    out_dir = RESULTS / scene / f"f{frame:04d}" / "point" / f"pt_{xy[0]:04d}_{xy[1]:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(MODELS) if model == "both" else [model]
    named = {}
    for m in models:
        predictor = _make_segmenter(m, cfg, device)  # sam3 -> Sam3Segmenter (predictor interactivo, da stability)
        named[m] = predictor.predict_point(image, xy)
        point_ambiguity.render(image, named[m], str(out_dir / f"{m}.png"))
    if len(named) > 1:
        point_ambiguity.render_compare(image, named, str(out_dir / "compare.png"))
        return out_dir / "compare.png"
    return out_dir / f"{model}.png"
