"""Pre-fusion checkpoints: everything a fusion step reads, saved right before it runs.

YAML `replay.checkpoints`:
    off    nothing is saved (default)
    last   one per fusion, each replacing the previous: the run ends with the last one
    all    one per fusion

File `{scene}/fusion/pre_fusion_{frame:05d}.ckpt` (ovo.layout.pre_fusion_checkpoint), a dict:
    {"format": 1,
     "frame_id": frame of the fusion,
     "config": the run's config,
     "map": slam_backbone.get_map_dict() (xyz, ids, obj_ids, max_id, color),
     "ovo": ovo.replay.state.capture(ovo)}
Poses and keyframes are not saved: a fusion does not read them. Any backbone whose map is a
point cloud (`get_map_dict`) can be checkpointed.
"""

import copy
from pathlib import Path
from typing import Any, Optional

import torch

from .. import layout
from . import state

FORMAT = 1
MODES = ("off", "last", "all")


def mode_from_config(config: dict) -> str:
    raw = dict(config.get("replay") or {})
    unknown = set(raw) - {"checkpoints"}
    if unknown:
        raise ValueError(f"replay: unknown key(s) {sorted(unknown)}; valid: ['checkpoints']")
    mode = raw.get("checkpoints", "off")
    mode = "off" if mode is False else mode  # YAML reads a bare `off` as False
    if mode not in MODES:
        raise ValueError(f"replay.checkpoints: must be one of {list(MODES)}, got {mode!r}")
    return mode


class Checkpointer:
    """Saves the pre-fusion checkpoints of one scene, following `replay.checkpoints`.

    Checkpoints left in the scene by an earlier run are removed on creation (as FusionLog does
    with its CSV): a replay must never pick a fusion this run did not make.
    """

    def __init__(self, mode: str, scene_dir: Path) -> None:
        if mode not in MODES:
            raise ValueError(f"replay.checkpoints: must be one of {list(MODES)}, got {mode!r}")
        self.mode = mode
        self.scene_dir = Path(scene_dir)
        self._last: Optional[Path] = None
        for stale in available(self.scene_dir).values():
            stale.unlink()

    def save(self, frame_id: int, slam_backbone, ovo, config: dict) -> None:
        if self.mode == "off":
            return
        # update_map would run the pending CLIP work first anyway; do it now so it is saved.
        ovo.complete_semantic_info()
        path = layout.pre_fusion_checkpoint(self.scene_dir, frame_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": FORMAT,
            "frame_id": frame_id,
            "config": copy.deepcopy(config),
            "map": slam_backbone.get_map_dict(),
            "ovo": state.capture(ovo),
        }, path)
        if self.mode == "last" and self._last is not None and self._last != path:
            self._last.unlink(missing_ok=True)
        self._last = path


def load(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != FORMAT:
        raise ValueError(f"{path}: checkpoint format {ckpt.get('format')!r}, expected {FORMAT}")
    return ckpt


def available(scene_dir: Path) -> dict[int, Path]:
    """{frame: path} of the pre-fusion checkpoints of a scene."""
    found = {}
    for p in sorted(layout.fusion_dir(scene_dir).glob("pre_fusion_*.ckpt")):
        frame = p.stem.removeprefix("pre_fusion_")
        if frame.isdigit():
            found[int(frame)] = p
    return found
