"""Correlaciona la dinámica de la cámara (derivadas de la pose estimada) con la telemetría Tier2 del contest.

Fuentes (por escena de un experimento):
  - estimated_c2w.npy    : torch.save de {frame_id: c2w 4x4} (pose estimada por frame).
  - logger/contest/kf_frame_id.log : frame_id fuente de cada muestra Tier2 (ancla de alineación).
  - logger/contest/*.log : señales Tier2 (n_matched, n_robos, ...), fila-alineadas con kf_frame_id.

Señales de movimiento (derivadas sobre la trayectoria densa, en reloj de frame):
  v_lin  = ‖Δpos‖ / Δframe            (rapidez de traslación, módulo)
  v_ang  = ángulo geodésico / Δframe  (rapidez de giro, módulo, invariante)
  yaw/pitch/roll = componentes euler de ΔR / Δframe (solo "por ver"; acopladas, frame-dependientes)
  a_lin, a_ang = 2ª derivada (aceleración / tirón)

Uso:
  python -m studies.cam_motion.analyze --exp <ruta_o_id> --scene scene0011_00 [--fps 30] [--out <dir>]
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr

TELEMETRY = ["n_robos", "robo_rate", "orphan_rate", "n_orphans", "n_births", "n_matched", "n_pre_assign"]
MOTION = ["v_lin", "v_ang", "yaw", "pitch", "roll", "a_lin", "a_ang"]


def _resolve(exp: str) -> pathlib.Path:
    p = pathlib.Path(exp)
    if p.is_dir():
        return p.resolve()
    for root in (pathlib.Path("data/output/ScanNet"), pathlib.Path("data/output/Replica")):
        hits = sorted(root.glob(f"{exp}*"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"experimento no encontrado: {exp}")


def load_poses(scene_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (frames ordenados, posiciones (N,3), rotaciones (N,3,3)). Toda la trayectoria densa."""
    c2w = torch.load(scene_dir / "estimated_c2w.npy", weights_only=False)
    frames = np.array(sorted(c2w.keys()))
    mats = np.stack([np.asarray(c2w[int(f)], dtype=np.float64) for f in frames])
    return frames, mats[:, :3, 3], mats[:, :3, :3]


def camera_motion(frames: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> dict[int, dict]:
    """Derivadas por frame (backward diff, normalizadas por Δframe). Clave = frame de destino."""
    out: dict[int, dict] = {}
    for i in range(1, len(frames)):
        df = float(frames[i] - frames[i - 1]) or 1.0
        R_rel = rot[i] @ rot[i - 1].T
        r = Rotation.from_matrix(R_rel)
        yaw, pitch, roll = r.as_euler("yxz", degrees=True)  # y=yaw, x=pitch, z=roll
        out[int(frames[i])] = {
            "v_lin": float(np.linalg.norm(pos[i] - pos[i - 1]) / df),
            "v_ang": float(np.degrees(r.magnitude()) / df),
            "yaw": yaw / df, "pitch": pitch / df, "roll": roll / df,
        }
    return out


def _nearest(frame: int, keys: np.ndarray) -> int:
    return int(keys[np.abs(keys - frame).argmin()])


def load_telemetry(scene_dir: pathlib.Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """-> (kf_frame_id por muestra, {señal: serie}). Deriva robo_rate/orphan_rate al vuelo."""
    cdir = scene_dir / "logger" / "contest"
    kf_frame = np.loadtxt(cdir / "kf_frame_id.log", ndmin=1).astype(int)
    raw = {s: np.loadtxt(cdir / f"{s}.log", ndmin=1) for s in
           ["n_matched", "n_pre_assign", "n_used", "n_orphans", "n_births", "n_robos"]}
    n = len(kf_frame)
    raw = {k: v[:n] for k, v in raw.items()}  # por si algún log quedó 1 corto
    raw["robo_rate"] = raw["n_robos"] / np.maximum(raw["n_pre_assign"], 1)
    raw["orphan_rate"] = raw["n_orphans"] / np.maximum(raw["n_matched"], 1)
    return kf_frame[:min(len(v) for v in raw.values())], raw


def build_matrix(scene_dir: pathlib.Path):
    frames, pos, rot = load_poses(scene_dir)
    motion = camera_motion(frames, pos, rot)
    mkeys = np.array(sorted(motion.keys()))
    kf_frame, tele = load_telemetry(scene_dir)
    n = min(len(kf_frame), *(len(v) for v in tele.values()))
    kf_frame, tele = kf_frame[:n], {k: v[:n] for k, v in tele.items()}

    # velocidad en cada frame de telemetría (lookup exacto o vecino más próximo)
    mot = {m: np.array([motion[_nearest(f, mkeys)][m] for f in kf_frame]) for m in ["v_lin", "v_ang", "yaw", "pitch", "roll"]}
    mot["a_lin"] = np.gradient(mot["v_lin"])
    mot["a_ang"] = np.gradient(mot["v_ang"])
    return kf_frame, mot, tele


def correlate(mot: dict, tele: dict, lags=(0, 1)) -> list[dict]:
    rows = []
    for m in MOTION:
        x = np.abs(mot[m]) if m in ("yaw", "pitch", "roll") else mot[m]
        for t in TELEMETRY:
            y = tele[t]
            for lag in lags:
                a, b = (x[:-lag], y[lag:]) if lag > 0 else (x, y)
                if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
                    continue
                rows.append({"motion": m, "telemetry": t, "lag": lag,
                             "pearson": pearsonr(a, b)[0], "p_p": pearsonr(a, b)[1],
                             "spearman": spearmanr(a, b)[0], "p_s": spearmanr(a, b)[1]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    scene_dir = _resolve(args.exp) / args.scene
    kf_frame, mot, tele = build_matrix(scene_dir)
    rows = correlate(mot, tele)

    print(f"\n{scene_dir.parent.name} / {args.scene} — {len(kf_frame)} muestras Tier2 alineadas a pose\n")
    print(f"{'motion':7} {'telemetry':13} lag {'pearson':>8} {'p':>8} {'spearman':>9} {'p':>8}")
    for r in sorted(rows, key=lambda r: -abs(r["spearman"])):
        star = "*" if r["p_s"] < 0.05 else " "
        print(f"{r['motion']:7} {r['telemetry']:13} {r['lag']:>3} {r['pearson']:>8.3f} {r['p_p']:>8.3g} "
              f"{r['spearman']:>9.3f} {r['p_s']:>8.3g} {star}")

    out = pathlib.Path(args.out) if args.out else scene_dir / "fusion" / "contest" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    _plots(kf_frame, mot, tele, rows, out, args.fps)
    print(f"\nfiguras + csv -> {out}")


def _plots(kf_frame, mot, tele, rows, out: pathlib.Path, fps: float):
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(out / "cam_motion_corr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["motion", "telemetry", "lag", "pearson", "p_p", "spearman", "p_s"])
        w.writeheader()
        w.writerows(rows)

    # overlay: velocidad de cámara vs robos
    x = np.arange(len(kf_frame))
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(x, mot["v_ang"], color="tab:red", lw=1, label="v_ang (deg/frame)")
    ax1.plot(x, mot["v_lin"], color="tab:orange", lw=1, alpha=0.7, label="v_lin (m/frame)")
    ax1.set_ylabel("velocidad cámara"); ax1.set_xlabel("muestra Tier2 (KF segmentado)")
    ax2 = ax1.twinx()
    ax2.plot(x, tele["robo_rate"], color="tab:blue", lw=1, alpha=0.6, label="robo_rate")
    ax2.set_ylabel("robo_rate", color="tab:blue")
    ax1.legend(loc="upper left"); ax1.set_title("Dinámica de cámara vs robos")
    fig.tight_layout(); fig.savefig(out / "cam_motion_overlay.svg"); plt.close(fig)

    # heatmap spearman lag0
    lag0 = [r for r in rows if r["lag"] == 0]
    M = {(r["motion"], r["telemetry"]): r["spearman"] for r in lag0}
    grid = np.array([[M.get((m, t), np.nan) for t in TELEMETRY] for m in MOTION])
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TELEMETRY))); ax.set_xticklabels(TELEMETRY, rotation=45, ha="right")
    ax.set_yticks(range(len(MOTION))); ax.set_yticklabels(MOTION)
    for i in range(len(MOTION)):
        for j in range(len(TELEMETRY)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Spearman  movimiento ↔ telemetría  (lag 0)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "cam_motion_heatmap.svg"); plt.close(fig)


if __name__ == "__main__":
    main()
