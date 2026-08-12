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

from studies.segmentation.core.segmenters import SamConfig, SamSegmenter
from studies.segmentation.core.nms_decision import scores_from_records, evaluate
from studies.segmentation.viz import pipeline_steps, masks_gallery, removed_masks, decision_trace

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


def _write_meta(out_dir: Path, cfg: SamConfig, thr: dict, frame_path: str, counts: dict) -> None:
    """Vuelca meta.json autodescriptivo del run."""
    meta = {
        "variant": out_dir.name,
        "scene": out_dir.parent.parent.name,
        "frame": int(out_dir.parent.name.lstrip("f")),
        "frame_path": frame_path,
        "resize": list(_RESIZE),
        "segmenter": asdict(cfg),
        "ovo_nms": thr,
        "counts": counts,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def run(scene: str, frame: int, variant: str, cfg: SamConfig, thr: dict,
        lenses: tuple[str, ...], dataset_root: str, device: str) -> Path:
    """Ejecuta el estudio sobre un frame y escribe todo en results/{scene}/f{frame}/{variant}/."""
    image, frame_path = _load_frame(dataset_root, scene, frame)
    records = SamSegmenter(cfg, device=device).segment(image)
    masks, scores = scores_from_records(records)
    breakdown = evaluate(masks, scores, thr["iou_thr"], thr["score_thr"], thr["inner_thr"])

    out_dir = _RESULTS / scene / f"f{frame:04d}" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if "pipeline" in lenses:
        pipeline_steps.render(image, records, breakdown, str(out_dir / "pipeline.png"))
    if "masks" in lenses:
        masks_gallery.render(image, records, str(out_dir / "masks"), breakdown)
    if "removed" in lenses:
        removed_masks.render(image, records, breakdown, str(out_dir / "removed"))
    if "trace" in lenses:
        decision_trace.render(breakdown, str(out_dir / "trace.md"))

    counts = {"n_raw": len(records), "n_kept": len(breakdown.kept), "n_removed": len(breakdown.removed)}
    _write_meta(out_dir, cfg, thr, frame_path, counts)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Estudio de segmentación SAM sobre un frame de Replica.")
    p.add_argument("scene")
    p.add_argument("frame", type=int)
    p.add_argument("--variant", default="sam2_baseline", help="etiqueta de la config (carpeta)")
    p.add_argument("--lenses", nargs="+", default=list(_LENSES), choices=_LENSES)
    p.add_argument("--dataset-root", default=_DEFAULT_DATASET)
    p.add_argument("--ckpt", default="data/input/sam_ckpts/")
    p.add_argument("--device", default="cuda")
    # generación (SAM2)
    p.add_argument("--points-per-side", type=int, default=16)
    p.add_argument("--crop-n-layers", type=int, default=0)
    # poda (NMS externo OVO)
    p.add_argument("--iou-thr", type=float, default=0.8)
    p.add_argument("--score-thr", type=float, default=0.7)
    p.add_argument("--inner-thr", type=float, default=0.5)
    args = p.parse_args()

    cfg = SamConfig(
        sam_ckpt_path=args.ckpt,
        points_per_side=args.points_per_side,
        crop_n_layers=args.crop_n_layers,
    )
    thr = {"iou_thr": args.iou_thr, "score_thr": args.score_thr, "inner_thr": args.inner_thr}
    out = run(args.scene, args.frame, args.variant, cfg, thr,
              tuple(args.lenses), args.dataset_root, args.device)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
