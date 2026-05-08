from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RunConfig:
    device: str = "cuda"
    dataset_name: str = ""
    scene_name: str = ""
    use_wandb: bool = False
    restore_map: bool = False
    debug: bool = False
    debug_info: bool = False
    slam_module: str = "vanilla"
    save_estimated_cam: bool = False
    eval: bool = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RunConfig":
        return cls(
            device=config.get("device", "cuda"),
            dataset_name=config["dataset_name"],
            scene_name=config["data"]["scene_name"],
            use_wandb=config["use_wandb"],
            restore_map=config.get("restore_map", False),
            debug=config.get("debug", False),
            debug_info=config.get("debug_info", False),
            slam_module=config["slam"].get("slam_module", "vanilla"),
            save_estimated_cam=config["slam"].get("save_estimated_cam", False),
        )
