#!/usr/bin/env python3
"""Replay only the fusion step from a pre-fusion checkpoint.

Usage:
    python scripts/replay_fusion.py \\
        --checkpoint data/checkpoints/20260101_SimSLAM_CLIP_myscene/office0/pre_fusion.ckpt \\
        --experiment-name 20260101_SimSLAM_NewFusion_myscene \\
        --fusion-config data/working/configs/Replica/new_fusion.yaml

The script restores the geometric + semantic state saved just before fusion and
re-runs only the fusion step, then saves the result under a new experiment name
in data/output/. From there, use run_eval.py --segment --eval as normal.
"""

import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ovo.entities.ovomapping import OVOSemMap
from ovo.utils import io_utils


def main():
    parser = argparse.ArgumentParser(description="Replay fusion from a pre-fusion checkpoint.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to pre_fusion.ckpt file.")
    parser.add_argument("--experiment-name", required=True,
                        help="Output experiment name (written to data/output/<dataset>/<experiment-name>/<scene>/).")
    parser.add_argument("--fusion-config", default=None,
                        help="Optional YAML config to overlay on top of checkpoint config (fusion params only).")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    if args.fusion_config:
        overlay = io_utils.load_config(args.fusion_config)
        io_utils.update_recursive(config, overlay)
        print(f"Fusion config overlay applied from {args.fusion_config}")

    dataset = config.get("dataset_name", "Replica").capitalize()
    scene = config["data"]["scene_name"]
    output_path = Path("data/output") / dataset / args.experiment_name / scene

    print(f"Output path: {output_path}")
    gslam = OVOSemMap(config, output_path=str(output_path))
    gslam.run_fusion_from_checkpoint(ckpt_path)

    print(f"\nDone. Run segment + eval with:")
    print(f"  python run_eval.py --dataset_name {dataset} --experiment_name {args.experiment_name} --segment --eval")


if __name__ == "__main__":
    main()
