"""Resuelve un experimento+escena a sus artefactos de contest.

Corazón del asunto: analizar el contest exige el mapa TAL COMO LO VIO el contest,
no el mapa post-fusión. Ese mapa es `pre_fusion.ckpt`. Dos formas de llegar a él:

  - generador (jump-drift + save_pre_fusion_checkpoint): ESCRIBIÓ su pre_fusion.ckpt
    bajo data/checkpoints/<rel>/<scene>/.
  - consumidor (replay, restore_pre_fusion_checkpoint): NO lo escribe; su config
    apunta al ckpt de un generador.

El sustrato (ckpt) es el mismo se mire por donde se mire; los veredictos/umbrales
son por-run. Config dice qué DEBERÍA existir; el disco dice qué EXISTE.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from . import paths

try:
    import yaml
except ImportError:  # yaml solo hace falta para leer el config del run
    yaml = None


@dataclass
class ExpContext:
    """Todo lo que las capas de arriba necesitan para consultar un run de contest."""

    exp_id: str
    scene: str
    exp_path: pathlib.Path
    kind: str                              # "consumer" | "generator" | "none"
    config: dict
    ckpt_path: pathlib.Path | None         # pre_fusion.ckpt resuelto
    verdicts_path: pathlib.Path | None     # contest_verdicts.csv (histórico)
    contest_json_path: pathlib.Path | None
    ok: bool = False                       # ckpt resuelto Y existe en disco
    errors: list[str] = field(default_factory=list)


def resolve_exp_path(exp: str | pathlib.Path) -> pathlib.Path:
    """Acepta ruta completa, ID corto, o prefijo único bajo OUTPUT_ROOT."""
    p = pathlib.Path(exp)
    if p.is_dir():
        return p.resolve()
    candidate = paths.OUTPUT_ROOT / str(exp)
    if candidate.is_dir():
        return candidate
    matches = sorted(paths.OUTPUT_ROOT.glob(f"{exp}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches[:8])
        raise ValueError(f"ID '{exp}' ambiguo: {len(matches)} coinciden ({names}…)")
    raise FileNotFoundError(f"experimento '{exp}' no encontrado en {paths.OUTPUT_ROOT}")


def _load_config(scene_dir: pathlib.Path) -> dict:
    cfg_path = scene_dir / "config.yaml"
    if yaml is None or not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


def _first_existing(*candidates: pathlib.Path) -> pathlib.Path | None:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else None


def _resolve_ckpt(exp_path: pathlib.Path, scene: str, config: dict) -> tuple[str, pathlib.Path | None]:
    """(kind, ckpt_path) según el config del run. No comprueba existencia aún."""
    restore = config.get("restore_pre_fusion_checkpoint")
    if restore:
        p = pathlib.Path(restore)
        if not p.is_absolute():
            p = paths.REPO_ROOT / p
        return "consumer", p

    noise = config.get("noise", {}) or {}
    if noise.get("save_pre_fusion_checkpoint"):   # baseline limpio o jump-drift: ambos guardan ckpt
        scene_dir = exp_path / scene
        try:
            rel = scene_dir.relative_to(paths.REPO_ROOT / "data" / "output")
        except ValueError:
            rel = pathlib.Path(exp_path.name) / scene
        return "generator", paths.CHECKPOINT_ROOT / rel / "pre_fusion.ckpt"

    return "none", None


def resolve(exp: str | pathlib.Path, scene: str) -> ExpContext:
    """exp+scene -> ExpContext. Nunca lanza por datos que faltan: lo reporta en .errors/.ok."""
    exp_path = resolve_exp_path(exp)
    scene_dir = exp_path / scene
    errors: list[str] = []

    if not scene_dir.is_dir():
        available = sorted(d.name for d in exp_path.iterdir() if d.is_dir())
        errors.append(f"escena '{scene}' no existe en {exp_path.name}. Disponibles: {', '.join(available) or '(ninguna)'}")
        return ExpContext(exp_path.name, scene, exp_path, "none", {}, None, None, None, ok=False, errors=errors)

    config = _load_config(scene_dir)
    kind, ckpt_path = _resolve_ckpt(exp_path, scene, config)

    verdicts_path = _first_existing(
        scene_dir / "fusion" / "contest" / "contest_verdicts.csv",
        scene_dir / "contest_verdicts.csv",
    )
    contest_json_path = _first_existing(
        scene_dir / "fusion" / "contest" / "contest.json",
        scene_dir / "contest.json",
    )

    if kind == "none":
        errors.append(
            "el config de este run no genera sustrato de contest "
            "(sin restore_pre_fusion_checkpoint ni jump_drift+save_pre_fusion_checkpoint)"
        )
    elif ckpt_path is None or not ckpt_path.exists():
        errors.append(
            f"config indica {kind} pero el ckpt no está en disco: {ckpt_path} "
            "(run incompleto o checkpoint borrado)"
        )

    ok = kind != "none" and ckpt_path is not None and ckpt_path.exists()
    return ExpContext(
        exp_id=exp_path.name, scene=scene, exp_path=exp_path, kind=kind,
        config=config, ckpt_path=ckpt_path, verdicts_path=verdicts_path,
        contest_json_path=contest_json_path, ok=ok, errors=errors,
    )
