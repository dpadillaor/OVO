#!/usr/bin/env python3
"""
SAM2 parameter exploration script with layer-by-layer analysis.
Generates Capa 0, 1, 2 + final filtered masks for each frame.
"""
import os
import sys
import yaml
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # For modified SAM2
from ovo.utils import segment_utils
from sam2_amg_modified import SAM2AutomaticMaskGeneratorModified

class SAM2Explorer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = self.cfg["device"]
        self.debug = self.cfg["output"]["debug_level"]

        # Extract dataset and scene from dataset_path
        dataset_path = Path(self.cfg["dataset"]["dataset_path"])
        self._infer_dataset_scene(dataset_path)

    def _infer_dataset_scene(self, dataset_path: Path):
        """Infer dataset name and scene from path."""
        # dataset_path = "data/input/Datasets/replica/office_0/rgb"
        parts = dataset_path.parts

        if "Datasets" in parts:
            idx = parts.index("Datasets")
            self.dataset_name = parts[idx + 1]  # replica
            self.scene_name = parts[idx + 2]    # office_0
        else:
            self.dataset_name = "unknown"
            self.scene_name = "unknown"

        self._log(f"Dataset: {self.dataset_name}, Scene: {self.scene_name}")

    def _get_output_dir(self) -> Path:
        """Get output directory in scripts/sam2_exploration/output."""
        # Config name: sam{points}_iou{iou}_score{score}_inner{inner}
        config_name = (
            f"sam{self.cfg['sam']['points_per_side']}_"
            f"iou{self.cfg['sam']['nms_iou_th']}_"
            f"score{self.cfg['sam']['nms_score_th']}_"
            f"inner{self.cfg['sam']['nms_inner_th']}"
        )

        # scripts/sam2_exploration/output/{dataset}_{scene}_{config}/
        output_dir = (
            Path(__file__).parent / "output" /
            f"{self.dataset_name}_{self.scene_name}_{config_name}"
        )
        return output_dir

    def _log(self, msg: str, level: int = 1):
        if self.debug >= level:
            print(f"[SAM2] {msg}")

    def _load_frames(self) -> List[Tuple[str, np.ndarray]]:
        """Load frames from dataset_path."""
        dataset_path = Path(self.cfg["dataset"]["dataset_path"])

        if dataset_path.is_file():
            frames = [(dataset_path.name, cv2.imread(str(dataset_path)))]
        else:
            # Support both frame*.jpg and *.jpg patterns
            frame_files = (
                sorted(dataset_path.glob("frame*.jpg")) +
                sorted(dataset_path.glob("frame*.png")) +
                sorted(dataset_path.glob("[0-9]*.jpg")) +
                sorted(dataset_path.glob("[0-9]*.png"))
            )
            # Remove duplicates while preserving order
            seen = set()
            frame_files = [f for f in frame_files if not (f in seen or seen.add(f))]

            indices = self.cfg["dataset"]["frame_indices"]
            if indices is not None:
                if isinstance(indices, str) and ":" in indices:
                    parts = [int(x) if x else None for x in indices.split(":")]
                    indices = list(range(*parts))
                frame_files = [f for i, f in enumerate(frame_files) if i in indices]

            frames = []
            for f in frame_files:
                img = cv2.imread(str(f))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append((f.name, img))

        self._log(f"Loaded {len(frames)} frame(s)")
        return frames

    def _generate_masks_by_layer(self, image: np.ndarray) -> Dict[str, List[dict]]:
        """Generate masks and filter by layer (0, 1, 2) using modified SAM2."""
        from sam2.build_sam import build_sam2

        config = self.cfg["sam"]

        # Build SAM2 model
        model_cfg = os.path.join(
            "configs", f"sam{config['sam_version']}",
            f"sam{config['sam_version']}_{config['sam_encoder']}.yaml"
        )
        model_cards = {
            "vit_b": "vit_b_01ec64.pth",
            "vit_h": "vit_h_4b8939.pth",
            "hiera_l": "hiera_large.pt",
            "hiera_t": "hiera_tiny.pt"
        }
        checkpoint_path = os.path.join(
            config["sam_ckpt_path"],
            f"sam{config['sam_version']}_{model_cards[config['sam_encoder']]}"
        )

        sam = build_sam2(model_cfg, checkpoint_path, device=self.device, mode="eval", apply_postprocessing=False)

        # Use MODIFIED mask generator (tracks layer_idx)
        mg = SAM2AutomaticMaskGeneratorModified(
            model=sam,
            points_per_side=config.get("points_per_side", 32),
            pred_iou_thresh=config.get("pred_iou_thresh", 0.8),
            stability_score_thresh=config.get("stability_score_th", 0.95),
            crop_n_layers=2,  # Generate all 3 layers (0, 1, 2)
        )

        # Generate masks (all layers mixed, but with layer_idx for each)
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                all_masks = mg.generate(image)

        # Filter by layer_idx
        layer_masks = {}
        for layer in range(3):
            masks_for_layer = [m for m in all_masks if m.get('layer_idx', -1) == layer]
            layer_masks[f"capa_{layer}"] = masks_for_layer
            self._log(f"Capa {layer}: {len(masks_for_layer)} máscaras (de {len(all_masks)} total)", level=1)

        return layer_masks

    def _post_process_masks(self, masks: List[dict]) -> List[dict]:
        """Apply NMS filtering."""
        if not masks:
            return []

        masks_filtered, = segment_utils.masks_update(
            masks,
            iou_thr=self.cfg["sam"]["nms_iou_th"],
            score_thr=self.cfg["sam"]["nms_score_th"],
            inner_thr=self.cfg["sam"]["nms_inner_th"],
        )
        return masks_filtered

    def _create_overlay(self, image: np.ndarray, seg_map: np.ndarray) -> np.ndarray:
        """Create RGB overlay with colored instance masks."""
        h, w = image.shape[:2]
        overlay = image.copy().astype(np.float32)

        if seg_map.size == 0:
            return overlay.astype(np.uint8)

        num_instances = seg_map.max() + 1
        colors = np.random.RandomState(42).randint(0, 255, (num_instances, 3), dtype=np.uint8)

        for inst_id in range(num_instances):
            mask = seg_map == inst_id
            overlay[mask] = overlay[mask] * 0.6 + colors[inst_id] * 0.4

        return overlay.astype(np.uint8)

    def _masks_to_segmap(self, masks: List[dict], image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert masks to segmentation map."""
        if not masks:
            return np.array([]), np.array([])
        return segment_utils.mask2segmap(masks, image)

    def _save_frame_results(self, frame_name: str, image: np.ndarray,
                           layer_masks: Dict, final_seg_map: np.ndarray):
        """Save all outputs for a frame."""
        output_dir = self._get_output_dir()
        frame_dir = output_dir / Path(frame_name).stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save layer overlays
        for layer_name, masks in layer_masks.items():
            masks_filtered = self._post_process_masks(masks)
            seg_map, _ = self._masks_to_segmap(masks_filtered, image)

            overlay = self._create_overlay(image, seg_map)
            overlay_path = frame_dir / f"{layer_name}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            self._log(f"Saved {layer_name}: {len(masks)} raw → {seg_map.max() + 1 if seg_map.size > 0 else 0} filtered")

        # Save final overlay
        overlay_final = self._create_overlay(image, final_seg_map)
        final_path = frame_dir / "final.png"
        cv2.imwrite(str(final_path), cv2.cvtColor(overlay_final, cv2.COLOR_RGB2BGR))
        self._log(f"Saved final: {final_seg_map.max() + 1 if final_seg_map.size > 0 else 0} masks")

    def _save_metadata(self):
        """Save configuration metadata."""
        output_dir = self._get_output_dir()
        metadata_path = output_dir / "metadata.json"

        metadata = {
            "dataset": self.dataset_name,
            "scene": self.scene_name,
            "timestamp": datetime.now().isoformat(),
            "sam_config": self.cfg["sam"],
            "nms_config": {
                "iou_th": self.cfg["sam"]["nms_iou_th"],
                "score_th": self.cfg["sam"]["nms_score_th"],
                "inner_th": self.cfg["sam"]["nms_inner_th"],
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self._log(f"Saved metadata: {metadata_path}")

    def run(self):
        """Main execution."""
        # Confirm CUDA status
        cuda_available = torch.cuda.is_available()
        device_info = f"GPU ({torch.cuda.get_device_name(0)})" if cuda_available else "CPU"
        print(f"\n{'='*50}")
        print(f"Device: {device_info}")
        print(f"CUDA Available: {cuda_available}")
        print(f"{'='*50}\n")

        frames = self._load_frames()
        output_dir = self._get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Output: {output_dir}")

        for frame_name, image in frames:
            self._log(f"\nProcessing: {frame_name}")

            # Generate masks by layer
            layer_masks = self._generate_masks_by_layer(image)

            # Final: combine all layers and filter
            all_masks = []
            for masks in layer_masks.values():
                all_masks.extend(masks)

            all_masks_filtered = self._post_process_masks(all_masks)
            final_seg_map, _ = self._masks_to_segmap(all_masks_filtered, image)

            # Save results
            self._save_frame_results(frame_name, image, layer_masks, final_seg_map)

        self._save_metadata()
        self._log(f"\nDone. All outputs in: {output_dir}")
        self._print_summary()

    def _print_summary(self):
        """Print config summary."""
        print("\n=== Config Summary ===")
        print(f"Dataset: {self.dataset_name}")
        print(f"Scene: {self.scene_name}")
        print(f"SAM Version: {self.cfg['sam']['sam_version']}")
        print(f"Encoder: {self.cfg['sam']['sam_encoder']}")
        print(f"Points per side: {self.cfg['sam']['points_per_side']}")
        print(f"NMS IoU threshold: {self.cfg['sam']['nms_iou_th']}")
        print(f"NMS score threshold: {self.cfg['sam']['nms_score_th']}")
        print(f"NMS inner threshold: {self.cfg['sam']['nms_inner_th']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 layer-by-layer exploration")
    parser.add_argument(
        "--config",
        default="scripts/sam2_exploration/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        help="Override points_per_side"
    )
    parser.add_argument(
        "--nms-iou-th",
        type=float,
        help="Override nms_iou_th"
    )
    parser.add_argument(
        "--nms-score-th",
        type=float,
        help="Override nms_score_th"
    )
    parser.add_argument(
        "--nms-inner-th",
        type=float,
        help="Override nms_inner_th"
    )
    parser.add_argument(
        "--frames",
        type=str,
        help="Frame indices (e.g., '0:10:2')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    explorer = SAM2Explorer(args.config)

    # Override config with CLI args
    if args.points_per_side:
        explorer.cfg["sam"]["points_per_side"] = args.points_per_side
    if args.nms_iou_th:
        explorer.cfg["sam"]["nms_iou_th"] = args.nms_iou_th
    if args.nms_score_th:
        explorer.cfg["sam"]["nms_score_th"] = args.nms_score_th
    if args.nms_inner_th:
        explorer.cfg["sam"]["nms_inner_th"] = args.nms_inner_th
    if args.frames:
        explorer.cfg["dataset"]["frame_indices"] = args.frames
    if args.verbose:
        explorer.debug = 2

    explorer.run()
