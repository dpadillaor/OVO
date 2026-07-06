#!/usr/bin/env python3
"""Costura A-W + normales: mide y visualiza el "giro de normal" en la frontera de
los pares de dominancia, la señal geométrica que separa fragmento-real (transferir)
de objeto-en-contacto (no transferir) cuando la máscara 2D y el descriptor no pueden.

Para cada par de dominancia (loser A -> winner W) del CSV de veredictos:
  - chunk = puntos de A que W reclama (del store; dueño==A y W entre sus ganadores).
  - para cada punto del chunk, busca sus K vecinos en W (kd-tree). Filtra los que no
    tienen W pegado (vecino mas cercano > DMAX). El parche promedia las K normales de
    W (con alineacion de signo) -> normal de la SUPERFICIE de W, no del filo ruidoso.
  - angulo |cos| entre la normal del punto del chunk y la normal-parche de W.
    Bajo (~<10 grados) = superficie continua = mismo objeto (transferir).
    Alto (~>20 grados) = quiebre = objetos distintos pegados (no transferir).

Requiere un ckpt CON normales (pcd_normals) y dueños INTACTOS, es decir un run con
`contest_split_mode: partial` (la dominancia no se ejecuta -> el chunk sigue siendo
de A). Un run con split aplicado mueve el chunk a W y el chunk sale vacio.

Las normales por punto viven en el mapa (map_params["normals"]), estimadas del depth
en VanillaMapper.map via geometry_utils.depth_to_normals.

Uso:
    python Study_seg/seam_normals_viz.py [RUN_OFFICE_DIR] [--kw 15] [--dmax 0.05]
                                         [--out Study_seg/normales_parche.rrd]
Si no se pasa RUN_OFFICE_DIR, coge el ultimo run *contest-normals-partial-pia*.
Imprime la tabla de angulos por par y guarda un .rrd para inspeccion visual.

Entidades del rrd, por par `[tag]_dom_A_W_angNN`:
  A_loser (gris)        nube completa de A (perdedor)
  W_winner (azul)       nube completa de W (ganador)
  seam_chunk_A (rojo)   puntos de A en disputa = candidatos a transferir
  W_patch_usado (amaril)los puntos de W que el parche promedia (la banda, no el filo)
  normal_chunk (naranja)normal de cada punto del chunk
  normal_W_parche (cian)normal-parche de W en el punto emparejado
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import open3d as o3d
import rerun as rr
import torch

LABELS = {(89, 103): "BUENO", (18, 26): "MALO", (31, 22): "MALO",
          (131, 103): "MALO", (134, 103): "MALO"}


def contest_file(run_dir: str, name: str) -> str:
    """Locate a contest output. New runs write it under ``{scene}/fusion/``;
    older runs left it at the scene root — prefer the new spot, fall back to old."""
    new = f"{run_dir}/fusion/{name}"
    return new if os.path.exists(new) else f"{run_dir}/{name}"


def find_default_dir() -> str:
    cands = sorted(glob.glob("data/output/Replica/*contest-normals-partial-pia*/office3"))
    if not cands:
        raise SystemExit("No encuentro un run *contest-normals-partial-pia*. Pasa el dir a mano.")
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None, help="dir .../<scene> con ovo_map.ckpt y contest.json")
    ap.add_argument("--kw", type=int, default=15, help="vecinos de W para el parche (default 15)")
    ap.add_argument("--dmax", type=float, default=0.05, help="distancia maxima de costura en metros (default 0.05)")
    ap.add_argument("--out", default="Study_seg/normales_parche.rrd", help="ruta del .rrd de salida")
    args = ap.parse_args()

    run_dir = (args.run_dir or find_default_dir()).rstrip("/")
    print(f"run_dir: {run_dir}")

    ck = torch.load(f"{run_dir}/ovo_map.ckpt", map_location="cpu", weights_only=False)
    mp = ck["map_params"]
    if "normals" not in mp:
        raise SystemExit("El ckpt no tiene normales. Usa un run con la integracion de normales.")
    xyz = mp["xyz"].numpy().astype(np.float64)
    nrm = mp["normals"].numpy().astype(np.float64)
    ids = mp["ids"].ravel().numpy()
    obj = mp["obj_ids"].ravel().numpy()
    store = json.load(open(contest_file(run_dir, "contest.json")))["grabs"]
    id2row = {int(k): i for i, k in enumerate(ids.tolist())}

    rows = list(csv.DictReader(open(contest_file(run_dir, "contest_verdicts.csv"))))
    dom = [(int(r["defender"]), int(r["challenger"])) for r in rows
           if r["decision"] == "SPLIT" and "dominancia" in r["reason"]]

    def chunk_rows(A, W):
        return np.array([id2row[int(k)] for k, wins in store.items()
                         if int(k) in id2row and obj[id2row[int(k)]] == A and str(W) in wins], dtype=np.int64)

    rr.init("seam_normals")
    L = 0.04
    out = [f"{'pair':>10} {'tag':6} {'costura':>7} {'ang_parche':>10}"]
    for A, W in dom:
        idxW = np.where(obj == W)[0]
        ch = chunk_rows(A, W)
        if len(ch) == 0 or len(idxW) == 0:
            continue
        pcW = o3d.geometry.PointCloud()
        pcW.points = o3d.utility.Vector3dVector(xyz[idxW])
        kdW = o3d.geometry.KDTreeFlann(pcW)

        sA, sWc, nWp, angs, wpatch = [], [], [], [], set()
        for r in ch:
            pa, na = xyz[r], nrm[r]
            if np.linalg.norm(na) < 1e-6:
                continue
            k = min(args.kw, len(idxW))
            _, nn, _ = kdW.search_knn_vector_3d(pa, k)
            wlist = [idxW[j] for j in nn]
            if np.linalg.norm(pa - xyz[wlist[0]]) > args.dmax:
                continue
            nref = nrm[wlist[0]]
            if np.linalg.norm(nref) < 1e-6:
                continue
            acc = np.zeros(3)
            for wj in wlist:
                nv = nrm[wj]
                if np.linalg.norm(nv) < 1e-6:
                    continue
                if np.dot(nv, nref) < 0:
                    nv = -nv
                acc += nv
                wpatch.add(int(wj))
            npar = acc / (np.linalg.norm(acc) + 1e-9)
            angs.append(np.degrees(np.arccos(abs(np.clip(np.dot(na, npar), -1, 1)))))
            sA.append(r)
            sWc.append(wlist[0])
            nWp.append(npar)
        if not angs:
            continue

        tag = LABELS.get((A, W), "")
        out.append(f"{A:>4}->{W:<4} {tag:6} {len(sA):>7} {np.mean(angs):>9.1f}d")

        sA, sWc, nWp = np.array(sA), np.array(sWc), np.array(nWp)
        wp = np.array(sorted(wpatch))
        a = int(round(np.mean(angs)))
        base = f"{tag + '_' if tag else ''}dom_{A}_{W}_ang{a:02d}"
        sub = np.linspace(0, len(sA) - 1, min(200, len(sA))).astype(int)
        rr.log(f"{base}/A_loser", rr.Points3D(xyz[obj == A], colors=[160, 160, 160], radii=0.004), static=True)
        rr.log(f"{base}/W_winner", rr.Points3D(xyz[idxW], colors=[60, 110, 200], radii=0.004), static=True)
        rr.log(f"{base}/seam_chunk_A", rr.Points3D(xyz[sA], colors=[230, 60, 60], radii=0.006), static=True)
        rr.log(f"{base}/W_patch_usado", rr.Points3D(xyz[wp], colors=[255, 230, 0], radii=0.007), static=True)
        rr.log(f"{base}/normal_chunk", rr.Arrows3D(origins=xyz[sA[sub]], vectors=nrm[sA[sub]] * L, colors=[255, 140, 0]), static=True)
        rr.log(f"{base}/normal_W_parche", rr.Arrows3D(origins=xyz[sWc[sub]], vectors=nWp[sub] * L, colors=[0, 210, 210]), static=True)

    rr.save(args.out)
    print("\n".join(out))
    print(f"\nrrd -> {args.out}")


if __name__ == "__main__":
    main()
