"""CLI del estudio: solo parseo de argumentos + dispatch. La orquestación vive en experiments."""
from __future__ import annotations

import argparse

from studies.segmentation.core.segmenters import SamConfig
from studies.segmentation import experiments as exp


def _add_common(sp: argparse.ArgumentParser, with_frame: bool = True) -> None:
    sp.add_argument("scene")
    if with_frame:
        sp.add_argument("frame", type=int)
    sp.add_argument("--dataset-root", default=exp.DEFAULT_DATASET)
    sp.add_argument("--ckpt", default="data/input/sam_ckpts/")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--points-per-side", type=int, default=16)
    sp.add_argument("--crop-n-layers", type=int, default=0)


def _add_thresholds(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--iou-thr", type=float, default=0.8)
    sp.add_argument("--score-thr", type=float, default=0.7)
    sp.add_argument("--inner-thr", type=float, default=0.5)


def main() -> None:
    p = argparse.ArgumentParser(description="Estudio de segmentación SAM sobre Replica.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("frame", help="lentes del AMG: pipeline/masks/removed/trace")
    _add_common(pf)
    _add_thresholds(pf)
    pf.add_argument("--model", default="sam2", choices=("sam2", "sam3"))
    pf.add_argument("--variant", default=None, help="etiqueta de la config; por defecto según modelo")
    pf.add_argument("--lenses", nargs="+", default=list(exp.LENSES), choices=exp.LENSES)

    pp = sub.add_parser("point", help="pincha un punto -> 3 máscaras multimask")
    _add_common(pp)
    pp.add_argument("--xy", type=int, nargs=2, required=True, metavar=("X", "Y"))
    pp.add_argument("--model", default="sam2", choices=("sam2", "sam3", "both"))

    pm = sub.add_parser("segmap", help="segmap final de un modelo, imagen sola (figura)")
    _add_common(pm)
    _add_thresholds(pm)
    pm.add_argument("--model", default="sam2", choices=("sam2", "sam3"))
    pm.add_argument("--variant", default=None, help="nombre del png; por defecto = model")
    pm.add_argument("--dim", type=float, default=0.6, help="brillo del fondo (1.0 = original)")
    pm.add_argument("--alpha", type=float, default=0.4, help="opacidad del color de máscara")
    pm.add_argument("--bg", default="frame", choices=("frame", "flat"),
                    help="flat = gris liso donde no hay máscara (resalta huecos)")
    pm.add_argument("--raw", action="store_true", help="máscaras crudas del AMG, sin el mask_nms de OVO")
    pm.add_argument("--style", default="fill", choices=("fill", "contour", "heat", "removed", "pairs"),
                    help="contour=bordes; heat=nº máscaras/píxel; removed=descartadas; pairs=descartada->asesina")
    pm.add_argument("--number", action="store_true", help="numera cada máscara en su centroide (anti-solape)")

    pc = sub.add_parser("compare", help="segmap final: sam2 vs sam3, lado a lado")
    _add_common(pc)
    _add_thresholds(pc)

    pg = sub.add_parser("prompts", help="frame con la rejilla de puntos que el AMG pincha")
    _add_common(pg)
    pg.add_argument("--model", default="sam2", choices=("sam2", "sam3"))
    pg.add_argument("--dim", type=float, default=0.75, help="brillo del frame de fondo")

    pt = sub.add_parser("timing", help="coste sam2 vs sam3 en un frame (reps)")
    _add_common(pt)
    pt.add_argument("--reps", type=int, default=5)

    ps = sub.add_parser("scene", help="coste sam2 vs sam3 a lo largo de una escena")
    _add_common(ps, with_frame=False)
    _add_thresholds(ps)
    ps.add_argument("--every", type=int, default=10, help="muestreo de frames (= segment_every de OVO)")
    ps.add_argument("--limit", type=int, default=None, help="máx frames (para iterar rápido)")
    ps.add_argument("--save-frames", action="store_true", help="guardar segmap por frame (original|sam2|sam3)")

    args = p.parse_args()
    cfg = SamConfig(sam_ckpt_path=args.ckpt, points_per_side=args.points_per_side,
                    crop_n_layers=args.crop_n_layers)

    if args.cmd == "frame":
        thr = {"iou_thr": args.iou_thr, "score_thr": args.score_thr, "inner_thr": args.inner_thr}
        variant = args.variant or ("sam3" if args.model == "sam3" else "sam2_baseline")
        out = exp.run_frame(args.scene, args.frame, args.model, variant, cfg, thr,
                            tuple(args.lenses), args.dataset_root, args.device)
    elif args.cmd == "segmap":
        thr = {"iou_thr": args.iou_thr, "score_thr": args.score_thr, "inner_thr": args.inner_thr}
        variant = args.variant or args.model
        out = exp.run_segmap(args.scene, args.frame, args.model, variant, cfg, thr,
                             args.dataset_root, args.device, args.dim, args.alpha, args.bg,
                             args.raw, args.style, args.number)
    elif args.cmd == "compare":
        thr = {"iou_thr": args.iou_thr, "score_thr": args.score_thr, "inner_thr": args.inner_thr}
        out = exp.run_compare(args.scene, args.frame, cfg, thr, args.dataset_root, args.device)
    elif args.cmd == "prompts":
        out = exp.run_prompts(args.scene, args.frame, args.model, cfg,
                              args.dataset_root, args.device, args.dim)
    elif args.cmd == "timing":
        out = exp.run_timing(args.scene, args.frame, cfg, args.reps, args.dataset_root, args.device)
    elif args.cmd == "scene":
        thr = {"iou_thr": args.iou_thr, "score_thr": args.score_thr, "inner_thr": args.inner_thr}
        out = exp.run_scene(args.scene, cfg, thr, args.every, args.dataset_root, args.device,
                            args.limit, args.save_frames)
    else:
        out = exp.run_point(args.scene, args.frame, args.model, cfg, tuple(args.xy),
                            args.dataset_root, args.device)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
