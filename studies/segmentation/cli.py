"""Orquestador del estudio: frame -> segmenter -> NMS -> viz, con rutas y meta.json automáticos.

Capa de I/O: aquí viven argparse, la carga de frames y la escritura a disco.
La lógica pura (poda, decisión) vive en core; el pintado en viz.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from studies.segmentation.core.segmenters import SamConfig, SamSegmenter, Sam3Segmenter, Sam3PointPredictor
from studies.segmentation.core.nms_decision import scores_from_records, evaluate
from studies.segmentation.core.profiling import profile_frame
from studies.segmentation.core.timing_stats import aggregate
from studies.segmentation.viz import (
    pipeline_steps, masks_gallery, removed_masks, decision_trace, point_ambiguity, segmenter_compare, timing as viz_timing)
from ovo.utils.segment_utils import mask2segmap

_RESULTS = Path(__file__).resolve().parent / "results"
_DEFAULT_DATASET = "data/input/Datasets/Replica"
_RESIZE = (1200, 680)
_LENSES = ("pipeline", "masks", "removed", "trace")


def _load_frame(dataset_root: str, scene: str, frame: int) -> tuple[np.ndarray, str]:
    """Carga y prepara un frame de Replica: (RGB uint8 HxW, ruta)."""
    path = os.path.join(dataset_root, scene, "results", f"frame{frame:06d}.jpg")
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"No existe el frame: {path}")
    img = cv2.resize(bgr, _RESIZE).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), path


def _write_meta(out_dir: Path, model: str, cfg: SamConfig, thr: dict, frame_path: str, counts: dict) -> None:
    """Vuelca meta.json autodescriptivo del run."""
    meta = {
        "variant": out_dir.name,
        "model": model,  # sam2 | sam3 (con sam3 la maquinaria AMG es la de SAM2, el predictor es SAM3)
        "scene": out_dir.parent.parent.name,
        "frame": int(out_dir.parent.name.lstrip("f")),
        "frame_path": frame_path,
        "resize": list(_RESIZE),
        "amg_params": asdict(cfg),
        "ovo_nms": thr,
        "counts": counts,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def run(scene: str, frame: int, model: str, variant: str, cfg: SamConfig, thr: dict,
        lenses: tuple[str, ...], dataset_root: str, device: str) -> Path:
    """Ejecuta el estudio sobre un frame y escribe todo en results/{scene}/f{frame}/{variant}/."""
    image, frame_path = _load_frame(dataset_root, scene, frame)
    segmenter = Sam3Segmenter(cfg, device=device) if model == "sam3" else SamSegmenter(cfg, device=device)
    records = segmenter.segment(image)
    masks, scores = scores_from_records(records)
    breakdown = evaluate(masks, scores, thr["iou_thr"], thr["score_thr"], thr["inner_thr"])

    out_dir = _RESULTS / scene / f"f{frame:04d}" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if "pipeline" in lenses:
        pipeline_steps.render(image, records, breakdown, str(out_dir / "pipeline.png"), raw_label=model.upper())
    if "masks" in lenses:
        masks_gallery.render(image, records, str(out_dir / "masks"), breakdown)
    if "removed" in lenses:
        removed_masks.render(image, records, breakdown, str(out_dir / "removed"))
    if "trace" in lenses:
        decision_trace.render(breakdown, str(out_dir / "trace.md"))

    counts = {"n_raw": len(records), "n_kept": len(breakdown.kept), "n_removed": len(breakdown.removed)}
    _write_meta(out_dir, model, cfg, thr, frame_path, counts)
    return out_dir


def _final_segmap(model: str, cfg: SamConfig, thr: dict, image: np.ndarray, device: str) -> np.ndarray:
    """Corre un segmentador, poda con el NMS de OVO y devuelve las capas finales (binary_maps)."""
    segmenter = Sam3Segmenter(cfg, device=device) if model == "sam3" else SamSegmenter(cfg, device=device)
    records = segmenter.segment(image)
    masks, scores = scores_from_records(records)
    bd = evaluate(masks, scores, thr["iou_thr"], thr["score_thr"], thr["inner_thr"])
    kept = [records[v.index] for v in bd.kept]
    _, binary_maps = mask2segmap(kept, image, sort=True)
    return binary_maps


def run_compare(scene: str, frame: int, cfg: SamConfig, thr: dict,
                dataset_root: str, device: str) -> Path:
    """Compara el segmap FINAL de sam2_baseline vs sam3 en un frame, lado a lado."""
    image, _ = _load_frame(dataset_root, scene, frame)
    named = {m: _final_segmap(m, cfg, thr, image, device) for m in ("sam2", "sam3")}
    out_dir = _RESULTS / scene / f"f{frame:04d}" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final.png"
    segmenter_compare.render(image, named, str(out_path))
    return out_path


def run_timing(scene: str, frame: int, cfg: SamConfig, reps: int,
               dataset_root: str, device: str) -> Path:
    """Mide el coste (total/encoder/decode/VRAM) de sam2 vs sam3 en un frame, reps veces."""
    from dataclasses import asdict
    image, _ = _load_frame(dataset_root, scene, frame)
    out_dir = _RESULTS / scene / f"f{frame:04d}" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    import gc
    import torch
    stats, blob = {}, {}
    for model in ("sam2", "sam3"):
        seg = Sam3Segmenter(cfg, device=device) if model == "sam3" else SamSegmenter(cfg, device=device)
        profiles = [profile_frame(seg, image, warmup=1 if r == 0 else 0) for r in range(reps)]
        stats[model] = aggregate(profiles)
        blob[model] = asdict(stats[model])
        del seg  # liberar el modelo antes de medir el siguiente: el pico de VRAM es global al proceso
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    viz_timing.render(stats, str(out_dir / "timing.png"),
                      title=f"{scene}/f{frame} coste AMG (reps={reps})")
    (out_dir / "timing.json").write_text(json.dumps(blob, indent=2))
    return out_dir / "timing.png"


def _point_predict(model: str, cfg: SamConfig, image: np.ndarray, xy: tuple[int, int], device: str):
    predictor = Sam3PointPredictor(device=device) if model == "sam3" else SamSegmenter(cfg, device=device)
    return predictor.predict_point(image, xy)


def run_point(scene: str, frame: int, model: str, cfg: SamConfig, xy: tuple[int, int],
              dataset_root: str, device: str) -> Path:
    """Pincha un punto con sam2/sam3/both y guarda en point/pt_X_Y/ ({model}.png y compare.png)."""
    image, _ = _load_frame(dataset_root, scene, frame)
    out_dir = _RESULTS / scene / f"f{frame:04d}" / "point" / f"pt_{xy[0]:04d}_{xy[1]:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["sam2", "sam3"] if model == "both" else [model]
    named = {}
    for m in models:
        pm = _point_predict(m, cfg, image, xy, device)
        named[m] = pm
        point_ambiguity.render(image, pm, str(out_dir / f"{m}.png"))
    if len(named) > 1:
        point_ambiguity.render_compare(image, named, str(out_dir / "compare.png"))
        return out_dir / "compare.png"
    return out_dir / f"{model}.png"


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("scene")
    sp.add_argument("frame", type=int)
    sp.add_argument("--dataset-root", default=_DEFAULT_DATASET)
    sp.add_argument("--ckpt", default="data/input/sam_ckpts/")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--points-per-side", type=int, default=16)
    sp.add_argument("--crop-n-layers", type=int, default=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Estudio de segmentación SAM sobre Replica.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("frame", help="lentes del AMG: pipeline/masks/removed/trace")
    _add_common(pf)
    pf.add_argument("--model", default="sam2", choices=("sam2", "sam3"))
    pf.add_argument("--variant", default=None, help="etiqueta de la config (carpeta); por defecto según modelo")
    pf.add_argument("--lenses", nargs="+", default=list(_LENSES), choices=_LENSES)
    pf.add_argument("--iou-thr", type=float, default=0.8)
    pf.add_argument("--score-thr", type=float, default=0.7)
    pf.add_argument("--inner-thr", type=float, default=0.5)

    pp = sub.add_parser("point", help="pincha un punto -> 3 máscaras multimask")
    _add_common(pp)
    pp.add_argument("--xy", type=int, nargs=2, required=True, metavar=("X", "Y"))
    pp.add_argument("--model", default="sam2", choices=("sam2", "sam3", "both"))

    pc = sub.add_parser("compare", help="segmap final: sam2_baseline vs sam3, lado a lado")
    _add_common(pc)
    pc.add_argument("--iou-thr", type=float, default=0.8)
    pc.add_argument("--score-thr", type=float, default=0.7)
    pc.add_argument("--inner-thr", type=float, default=0.5)

    pt = sub.add_parser("timing", help="coste (total/encoder/decode/VRAM) sam2 vs sam3")
    _add_common(pt)
    pt.add_argument("--reps", type=int, default=5)

    args = p.parse_args()
    cfg = SamConfig(sam_ckpt_path=args.ckpt, points_per_side=args.points_per_side,
                    crop_n_layers=args.crop_n_layers)

    if args.cmd == "frame":
        thr = {"iou_thr": args.iou_thr, "score_thr": args.score_thr, "inner_thr": args.inner_thr}
        variant = args.variant or ("sam3" if args.model == "sam3" else "sam2_baseline")
        out = run(args.scene, args.frame, args.model, variant, cfg, thr,
                  tuple(args.lenses), args.dataset_root, args.device)
    elif args.cmd == "compare":
        thr = {"iou_thr": args.iou_thr, "score_thr": args.score_thr, "inner_thr": args.inner_thr}
        out = run_compare(args.scene, args.frame, cfg, thr, args.dataset_root, args.device)
    elif args.cmd == "timing":
        out = run_timing(args.scene, args.frame, cfg, args.reps, args.dataset_root, args.device)
    else:
        out = run_point(args.scene, args.frame, args.model, cfg, tuple(args.xy),
                        args.dataset_root, args.device)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
