"""Carga el sustrato (MapState + store del contest) desde el pre_fusion.ckpt resuelto.

Pesado (torch.load ~1s, ~250MB). La TUI lo llama UNA vez por escena y cachea.
Soporta pre_fusion.ckpt (clave `ovo_params`, owners PRE-fusión) y, por compatibilidad,
el ovo_map.ckpt final (clave `ovo_map_params`, owners post-fusión).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ovo.entities.contest.store import ContestStore

from ..common.mapstate import MapState
from ..common.resolve import ExpContext


@dataclass
class Substrate:
    """Lo que el motor necesita: mapa vivo + store crudo + config del run."""

    map_state: MapState
    store: ContestStore
    contest_cfg: dict          # config del run (umbrales del contest bajo ["semantic"]["contest"])


def _ovo_params(ckpt: dict) -> dict:
    """El bloque semántico, se llame como se llame según el tipo de ckpt."""
    if "ovo_params" in ckpt:            # pre_fusion.ckpt
        return ckpt["ovo_params"]
    if "ovo_map_params" in ckpt:        # ovo_map.ckpt final (legacy)
        return ckpt["ovo_map_params"]
    raise KeyError("ckpt sin 'ovo_params' ni 'ovo_map_params'")


def _map_state(ckpt: dict) -> MapState:
    mp, om = ckpt["map_params"], _ovo_params(ckpt)
    ins_ids = [int(i) for i in om["ins_3d_ids"]]
    feats = {i: om[f"ins3d_{i}_clip_feature"] for i in ins_ids if f"ins3d_{i}_clip_feature" in om}
    return MapState(
        point_ids=mp["ids"].flatten(),
        points_ins_ids=mp["obj_ids"].flatten(),
        xyz=mp.get("xyz"),
        normals=mp.get("normals"),   # ckpts viejos no lo tienen -> callback seam se desactiva solo
        colors=mp.get("color"),
        clip_features=feats,
    )


def _store(ckpt: dict) -> ContestStore:
    raw = _ovo_params(ckpt).get("contest", {}) or {}
    store = ContestStore.from_dict(raw.get("grabs", {}))
    store.load_claims(raw.get("claims", {}))
    store.load_sightings(raw.get("sightings", {}))
    return store


def _contest_cfg(config: dict) -> dict:
    """Config semántica del run (mismo shape que consumía el probe): {} si no hay."""
    return (config or {}).get("semantic", config or {})


def load_substrate(ctx: ExpContext) -> Substrate:
    """ExpContext (ya validado) -> Substrate. Asume ctx.ok; lanza si el ckpt desapareció."""
    if ctx.ckpt_path is None or not ctx.ckpt_path.exists():
        raise FileNotFoundError(f"ckpt no disponible: {ctx.ckpt_path}")
    ckpt = torch.load(ctx.ckpt_path, map_location="cpu", weights_only=False)
    return Substrate(
        map_state=_map_state(ckpt),
        store=_store(ckpt),
        contest_cfg=_contest_cfg(ctx.config),
    )
