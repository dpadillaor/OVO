"""Where an experiment's artifacts live. Writers and readers both go through here.

    data/output/{dataset}/{experiment}/
        {scene}/                 one OVO run (config.yaml, ovo_map.ckpt, logger/, ...)
            fusion/fusion_decisions.csv   one row per pair judged by instance fusion
            fusion/pre_fusion_{frame:05d}.ckpt   state right before a fusion (ovo.replay.checkpoint)
            simulation.json               SimulatedSLAM only: the jumps and corrections that happened
        instance_pred/           instance predictions on the GT mesh (ovo.eval.predictions)
        instance_ap.json         instance AP, global and per scene (ovo.eval.instances)
        replays/{name}/          a fusion re-run from the scenes' checkpoints (ovo.replay), laid
                                 out like an experiment: {scene}/, instance_pred/, instance_ap.json

    data/input/Datasets/{dataset}/instance_gt/{scene}.txt   instance GT, one id per mesh vertex
"""

from dataclasses import dataclass
from pathlib import Path

OUTPUT_ROOT = Path("data/output")
DATASETS_ROOT = Path("data/input/Datasets")


@dataclass(frozen=True)
class ExperimentLayout:
    root: Path

    @classmethod
    def named(cls, dataset: str, experiment: str) -> "ExperimentLayout":
        return cls(OUTPUT_ROOT / dataset / experiment)

    def scene(self, scene: str) -> Path:
        return self.root / scene

    @property
    def instance_pred(self) -> Path:
        return self.root / "instance_pred"

    @property
    def instance_ap(self) -> Path:
        return self.root / "instance_ap.json"

    def replay(self, name: str) -> "ExperimentLayout":
        return ExperimentLayout(self.root / "replays" / name)


def fusion_dir(scene_dir: Path) -> Path:
    return Path(scene_dir) / "fusion"


def fusion_decisions(scene_dir: Path) -> Path:
    return fusion_dir(scene_dir) / "fusion_decisions.csv"


def pre_fusion_checkpoint(scene_dir: Path, frame_id: int) -> Path:
    return fusion_dir(scene_dir) / f"pre_fusion_{frame_id:05d}.ckpt"


def simulation(scene_dir: Path) -> Path:
    return Path(scene_dir) / "simulation.json"


def instance_gt_dir(dataset: str) -> Path:
    return DATASETS_ROOT / dataset / "instance_gt"
