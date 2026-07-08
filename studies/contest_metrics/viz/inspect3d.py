"""Visor 3D interactivo del contest: escena en gris + query por terminal de un par (def→ch).

Espejo de la herramienta de fusión (debug_merge_decisions) pero para SPLITS del contest.
Carga el substrato PRE-fusión (lo que el contest vio), abre una ventana Open3D no bloqueante
y lee pares del terminal. Para cada `def→ch` pinta las dos instancias y el TROZO disputado
(los split_points reales), y reporta las señales + el veredicto del discriminador.

Colores: gris contexto · rojo defender · azul challenger · amarillo el trozo (encima).
El corte de techo (--z-max) descarta lo alto para ver la habitación por dentro.
"""
from __future__ import annotations

import argparse
import queue
import re
import sys
import threading
import time

import numpy as np
import torch

from ..query.engine import ContestProbe

COLOR_CONTEXT = (0.62, 0.62, 0.62)   # gris
COLOR_DEF = (0.88, 0.15, 0.15)       # rojo: defender
COLOR_CH = (0.15, 0.35, 0.90)        # azul: challenger
COLOR_CHUNK = (0.10, 0.85, 0.20)     # verde: trozo disputado (split_points)
COLOR_HILITE = (0.90, 0.30, 0.85)    # magenta: instancia suelta (show)
CHUNK_REPS = 6                        # copias jitter para engordar el trozo
CHUNK_JITTER = 0.006                  # radio del jitter (m)
PROMPT = "contest> "


def _fatten(xyz: np.ndarray, color, reps: int = CHUNK_REPS, eps: float = CHUNK_JITTER):
    """Duplica cada punto con jitter pequeño -> el trozo se ve más grande/denso."""
    rng = np.random.default_rng(0)
    copies = [xyz] + [xyz + rng.uniform(-eps, eps, size=xyz.shape) for _ in range(reps - 1)]
    pts = np.concatenate(copies, axis=0)
    return pts, np.tile(np.array(color), (len(pts), 1))


class LiveViewer:
    """Ventana Open3D no bloqueante: show() cambia la nube, tick() bombea un frame."""

    def __init__(self, *, point_size: float = 3.0, window_name: str = "contest inspect") -> None:
        import open3d as o3d
        self._o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name)
        self.vis.get_render_option().point_size = point_size
        self._geoms: list = []

    def _pcd(self, xyz: np.ndarray, colors: np.ndarray):
        pcd = self._o3d.geometry.PointCloud()
        pcd.points = self._o3d.utility.Vector3dVector(xyz)
        pcd.colors = self._o3d.utility.Vector3dVector(colors)
        return pcd

    def show(self, xyz: np.ndarray, colors: np.ndarray,
             emph_xyz: np.ndarray | None = None, emph_color=None) -> None:
        """Base cloud + trozo enfatizado (duplicado+jitter -> se ve más grande)."""
        first = not self._geoms
        for g in self._geoms:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self._geoms = []
        geoms = [self._pcd(xyz, colors)]
        if emph_xyz is not None and len(emph_xyz):
            fat_xyz, fat_col = _fatten(emph_xyz, emph_color)
            geoms.append(self._pcd(fat_xyz, fat_col))
        for i, g in enumerate(geoms):
            self.vis.add_geometry(g, reset_bounding_box=(first and i == 0))
            self._geoms.append(g)

    def tick(self) -> bool:
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self) -> None:
        self.vis.destroy_window()


class Inspector:
    """Estado del substrato + render. Mantiene el corte de techo y pinta bajo demanda."""

    def __init__(self, probe: ContestProbe, z_max: float | None) -> None:
        self.p = probe
        self.xyz_full = probe._map.xyz.numpy().astype(np.float64)
        self.ins_full = probe._map.points_ins_ids.numpy()
        self.pids = probe._map.point_ids
        self.z_max = z_max
        # máscara del corte de techo: los puntos que se renderizan
        if z_max is not None:
            self.keep = self.xyz_full[:, 2] <= z_max
        else:
            self.keep = np.ones(len(self.xyz_full), dtype=bool)
        self.xyz = self.xyz_full[self.keep]
        self.ins = self.ins_full[self.keep]

    # ---- pintado --------------------------------------------------------
    def _base(self) -> np.ndarray:
        return np.tile(np.array(COLOR_CONTEXT), (len(self.xyz), 1))

    def scene(self) -> np.ndarray:
        return self._base()

    def _chunk_mask(self, split_points) -> np.ndarray:
        ids = torch.tensor([int(q) for q in (split_points or [])], dtype=self.pids.dtype)
        is_chunk_full = torch.isin(self.pids, ids).numpy() if ids.numel() else np.zeros(len(self.pids), bool)
        return is_chunk_full[self.keep]

    def pair_cloud(self, defender: int, challenger: int, split_points) -> np.ndarray:
        col = self._base()
        col[self.ins == defender] = COLOR_DEF
        col[self.ins == challenger] = COLOR_CH
        col[self._chunk_mask(split_points)] = COLOR_CHUNK   # encima
        return col

    def chunk_xyz(self, split_points) -> np.ndarray:
        """Coordenadas visibles del trozo (para engordarlo como capa aparte)."""
        return self.xyz[self._chunk_mask(split_points)]

    def instance_cloud(self, ins_id: int) -> np.ndarray:
        col = self._base()
        col[self.ins == ins_id] = COLOR_HILITE
        return col


# ---- REPL ---------------------------------------------------------------
_ARROW = re.compile(r"-?\d+")


def _parse_pair(line: str) -> tuple[int, int] | None:
    nums = _ARROW.findall(line)
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else None


def _report_pair(insp: Inspector, defender: int, challenger: int) -> object | None:
    """Imprime señales del par + veredicto del discriminador. Devuelve el PairFeatures (o None)."""
    f = insp.p.directed(defender, challenger)
    sizes = insp.p._map.sizes
    print(f"  def {defender} ({sizes.get(defender, 0)} pts) · ch {challenger} ({sizes.get(challenger, 0)} pts)")
    if f is None:
        print(f"  (sin disputa {defender}<-{challenger}: el challenger no le roba puntos firmes)")
        return None
    print(f"  trozo firm={f.firm_points}  cont={f.containment:.3f}  focus={f.focus:.2f}  "
          f"pers={f.persistence:.2f}  excl={f.exclusivity:.2f}")
    v = insp.p._verdict_toward(defender, challenger)
    if v is not None:
        print(f"  veredicto: {v.decision.name}  ·  {v.reason}")
    else:
        allv = insp.p.classify_all(defender)
        heads = ", ".join(f"{x.decision.name}->{x.challenger}" for x in allv) or "ninguno"
        print(f"  veredicto: no dirige a {challenger} (el defender resuelve: {heads})")
    return f


def _print_help() -> None:
    print("\nComandos:")
    print("  A B   |  A->B  |  A→B          pinta el par: def rojo, ch azul, TROZO amarillo")
    print("  show <id>                      resalta una instancia sola (magenta)")
    print("  pairs <def>                    lista los challengers de un defender + señales")
    print("  scene                          reset: toda la escena en gris")
    print("  help                           esta lista")
    print("  q                              salir")


def _cmd_pairs(insp: Inspector, defender: int) -> None:
    fs = sorted(insp.p._by_def.get(defender, []), key=lambda f: -f.firm_points)
    if not fs:
        print(f"  {defender} no tiene pares de disputa.")
        return
    print(f"  challengers de {defender}:")
    print(f"    {'ch':>5} {'firm':>6} {'cont':>5} {'focus':>5} {'pers':>5} {'excl':>5}")
    for f in fs:
        print(f"    {f.challenger:5d} {f.firm_points:6d} {f.containment:5.2f} "
              f"{f.focus:5.2f} {f.persistence:5.2f} {f.exclusivity:5.2f}")


def _handle(line: str, insp: Inspector, viewer: LiveViewer) -> None:
    tokens = line.split()
    head = tokens[0].lower()
    if head in ("help", "h", "?"):
        _print_help(); return
    if head == "scene":
        viewer.show(insp.xyz, insp.scene()); return
    if head == "show":
        if len(tokens) < 2 or not tokens[1].lstrip("-").isdigit():
            print("  uso: show <id>"); return
        i = int(tokens[1])
        n = int((insp.ins == i).sum())
        if n == 0:
            print(f"  instancia {i} sin puntos (bajo el corte o inexistente)."); return
        viewer.show(insp.xyz, insp.instance_cloud(i))
        print(f"  instancia {i}: {n} pts visibles"); return
    if head == "pairs":
        if len(tokens) < 2 or not tokens[1].lstrip("-").isdigit():
            print("  uso: pairs <def>"); return
        _cmd_pairs(insp, int(tokens[1])); return

    pair = _parse_pair(line)
    if pair is None:
        print("  no entiendo. escribe 'help'."); return
    defender, challenger = pair
    f = _report_pair(insp, defender, challenger)
    sp = f.split_points if f is not None else None
    viewer.show(insp.xyz, insp.pair_cloud(defender, challenger, sp),
                emph_xyz=insp.chunk_xyz(sp) if sp else None, emph_color=COLOR_CHUNK)


def run(exp: str, scene: str, z_max: float | None, point_size: float,
        overrides: dict | None = None) -> int:
    print(f"Cargando substrato {exp} / {scene} ...")
    probe = ContestProbe.from_experiment(exp, scene, overrides or {})
    insp = Inspector(probe, z_max)
    n_vis = int(insp.keep.sum())
    print(f"Cargados {len(insp.xyz_full)} puntos ({n_vis} bajo z<={z_max})."
          if z_max is not None else f"Cargados {len(insp.xyz_full)} puntos (sin corte).")
    print(f"{len(probe._pairs)} pares de disputa, {len(probe._by_def)} defenders.")

    viewer = LiveViewer(point_size=point_size, window_name=f"{scene} contest inspect")
    viewer.show(insp.xyz, insp.scene())

    commands: queue.Queue = queue.Queue()

    def _reader() -> None:
        for raw in sys.stdin:
            commands.put(raw.strip())
        commands.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    _print_help()
    print(PROMPT, end="", flush=True)

    running = True
    while running:
        try:
            line = commands.get_nowait()
        except queue.Empty:
            if not viewer.tick():
                break
            time.sleep(0.02)
            continue
        if line is None or line.lower() in ("q", "quit", "exit"):
            running = False
            continue
        if line:
            try:
                _handle(line, insp, viewer)
            except Exception as e:  # noqa: BLE001 — REPL: reporta, no crashea
                print(f"  error: {e}")
        print(PROMPT, end="", flush=True)

    viewer.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="contest_metrics inspect", description=__doc__)
    ap.add_argument("--exp", required=True, help="ID, ruta o prefijo del experimento")
    ap.add_argument("--scene", required=True, help="escena (p.ej. office4)")
    ap.add_argument("--z-max", dest="z_max", type=float, default=None,
                    help="corta el techo: renderiza z <= z_max (Replica office ~1.5)")
    ap.add_argument("--point-size", dest="point_size", type=float, default=3.0)
    args = ap.parse_args(argv)
    return run(args.exp, args.scene, args.z_max, args.point_size)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
