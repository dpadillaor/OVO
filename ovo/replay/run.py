"""Replay: re-run one fusion step from its pre-fusion checkpoint, with other fusion parameters.

Only that fusion is repeated; the run does not go on from it. No SLAM and no model is built:
the map and OVO's instances come from the checkpoint. Only `semantic.instance_fusion` may
change, since everything else acted before the checkpoint was taken.

Output, laid out like an experiment so it is evaluated like one:
    {experiment}/replays/{name}/{scene}/
        config.yaml           the run's config with the new fusion, plus `replayed_from`
        fusion/fusion_decisions.csv
        ovo_map.ckpt          map and instances after the fusion (as a run with debug off)
        logger/               t_fusion, t_fusion_judge
"""

import copy
from pathlib import Path
from typing import Optional

import torch

from .. import layout
from ..entities.fusion import FusionCfg
from ..entities.fusion.log import FusionLog
from ..entities.logger import Logger
from ..entities.ovo import OVO
from ..utils import io_utils
from . import checkpoint, state


def default_device() -> str:
    """cuda when a GPU is available, else cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def replay_scene(ckpt_path: Path, out_dir: Path, fusion: Optional[dict] = None, device: Optional[str] = None) -> None:
    """Replay the fusion saved in `ckpt_path` into the scene folder `out_dir`.
    `fusion` replaces `semantic.instance_fusion` (None keeps the original)."""
    device = device or default_device()
    ckpt = checkpoint.load(ckpt_path)
    frame_id = ckpt["frame_id"]
    config = copy.deepcopy(ckpt["config"])
    if fusion is not None:
        config["semantic"]["instance_fusion"] = fusion
    config["output_path"] = str(out_dir)
    config["replay"] = {"checkpoints": "off"}
    config["replayed_from"] = {"checkpoint": str(ckpt_path), "frame_id": frame_id}

    ovo = OVO.for_fusion(config["semantic"], device=device)  # validates instance_fusion
    state.restore(ovo, ckpt["ovo"])
    map_dict = ckpt["map"]
    map_data = (map_dict["xyz"].to(device), map_dict["ids"].to(device), map_dict["obj_ids"].squeeze(-1).to(device))

    points_ins_ids, report, seconds = ovo.update_map(map_data)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    io_utils.save_dict_to_yaml(config, "config.yaml", directory=out_dir)
    FusionLog(layout.fusion_decisions(out_dir)).write(frame_id, report)
    logger = Logger(out_dir)
    logger.log_ovo_stats({"t_fusion": seconds, "t_fusion_judge": report.seconds})
    logger.write_stats()
    map_dict = {**map_dict, "obj_ids": points_ins_ids.unsqueeze(-1).cpu()}
    io_utils.save_dict_to_ckpt({"map_params": map_dict, "ovo_map_params": ovo.capture_dict(debug_info=False)},
                               "ovo_map.ckpt", directory=out_dir)


def replay_experiment(dataset: str, experiment: str, name: str, fusion: Optional[dict] = None,
                      scenes: Optional[list[str]] = None, frame: Optional[int] = None,
                      device: Optional[str] = None) -> layout.ExperimentLayout:
    """Replay every scene of an experiment (or `scenes`) from its last checkpoint, or from
    `frame` (one scene only). Returns the replay's layout."""
    exp = layout.ExperimentLayout.named(dataset, experiment)
    if scenes is None:
        scenes = sorted(p.name for p in exp.root.iterdir()
                        if p.is_dir() and p.name != "replays" and checkpoint.available(p))
    if not scenes:
        raise ValueError(f"{exp.root}: no scene has pre-fusion checkpoints (replay.checkpoints)")
    if frame is not None and len(scenes) != 1:
        raise ValueError("--frame needs exactly one scene: each scene fuses at its own frames")

    FusionCfg.from_dict(fusion)  # an invalid fusion raises before any scene is written

    # Find every checkpoint first: a missing one fails before any scene is written.
    chosen = {}
    for scene in scenes:
        found = checkpoint.available(exp.scene(scene))
        if not found:
            raise ValueError(f"{exp.scene(scene)}: no pre-fusion checkpoints")
        if frame is not None and frame not in found:
            raise ValueError(f"{scene}: no checkpoint at frame {frame}; available: {sorted(found)}")
        chosen[scene] = found[frame if frame is not None else max(found)]

    out = exp.replay(name)
    for scene, ckpt_path in chosen.items():
        print(f"Replaying {scene} from {ckpt_path.name}")
        replay_scene(ckpt_path, out.scene(scene), fusion, device)
    return out
