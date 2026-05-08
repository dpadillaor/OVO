from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING
import torch
from pathlib import Path
import numpy as np
import time
import os
import gc
from .logger import Logger
from .ovo import OVO, MapData
from .run_config import RunConfig
from .semantic_config import SemanticConfig
from .datasets import get_dataset
from .visualizers import resolve_rerun_visual_mode, select_visualizer_target, VisualizationManager
from ..slam.vanilla_mapper import VanillaMapper
from ..utils import io_utils

@dataclass
class FrameState:
    frame_id: int
    frame_data: Any
    estimated_c2w: torch.Tensor

@dataclass
class VisConfig:
    stream: bool
    show_stream: bool
    vis_type: str = "open3d"
    rerun_visual_mode: Any = None
    save_rrd: bool = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VisConfig":
        vis = config["vis"]
        show_stream = vis["show_stream"]
        return cls(
            stream=vis["stream"],
            show_stream=show_stream,
            vis_type=vis.get("type", "open3d"),
            rerun_visual_mode=resolve_rerun_visual_mode(vis.get("rerun_visual_mode", None), show_stream),
            save_rrd=vis.get("save_rrd", False),
        )

@dataclass
class SchedulingConfig:
    map_every: int = 10
    segment_every: int = 10
    track_every: int = 1

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SchedulingConfig":
        tracking_cfg = config.get("tracking", None)
        return cls(
            map_every=config["mapping"].get("map_every", 10),
            segment_every=config["semantic"].get("segment_every", 10),
            track_every=1 if tracking_cfg is None else tracking_cfg.get("track_every", 1),
        )


def get_slam_backbone(config: Dict[str, Any], dataset, cam_intrinsics: torch.Tensor) -> VanillaMapper | WrapperGaussianSLAM:
    """Build and return the SLAM backend wrapper configured for the current run.

    Args:
        config: Experiment configuration dictionary.
        dataset: Dataset instance used by the SLAM backend.
        cam_intrinsics: Camera intrinsics tensor.

    Returns:
        The configured SLAM backend wrapper.
    """
    backbone = config["slam"].get("slam_module","vanilla")
    if backbone == "gaussian_slam":
        from ..slam.gaussian_slam import WrapperGaussianSLAM
        return WrapperGaussianSLAM(config, dataset)
    elif backbone ==  "orbslam2":
        from ..slam.orbslam2 import WrapperORBSLAM2
        return WrapperORBSLAM2(config, cam_intrinsics, world_ref=torch.from_numpy(dataset[0][3]))
    elif backbone == "groundtruth":
        from ..slam.groundtruth_slam import GroundTruthSLAM
        return GroundTruthSLAM(config, cam_intrinsics)
    else:
        return VanillaMapper(config, cam_intrinsics)

class OVOSemMap():
    """
    OVOSemMap is a class responsible for managing the semantic mapping process using the OVO framework.
    It initializes the necessary components, sets up the output path, and handles the main program flow
    including tracking, mapping, and semantic segmentation.
    Args:
        config (dict): Configuration dictionary containing experiment settings.
    """

    def __init__(self, config: Dict[str, Any], output_path: str) -> None:
        """Initialize semantic mapping pipeline components and runtime configuration.

        Args:
            config: Full experiment configuration dictionary.
            output_path: Directory where run artifacts and logs are stored.
        """
        # Persist experiment config and normalize output path in config.
        self._setup_output_path(output_path)
        io_utils.save_dict_to_yaml(config, "config.yaml", directory=self.output_path)
        config["output_path"] = str(self.output_path)

        # Store global runtime/config handles.
        self.config = config
        self._init_run_config(config)
        self._init_vis_config(config)
        self._init_scheduling(config)

        # Core services: logger and dataset.
        self.logger = Logger(self.output_path, os.getpid(), self.run_config.use_wandb)
        self.dataset = get_dataset(self.run_config.dataset_name)({**config["data"], **config["cam"]})

        # Camera intrinsics and semantic module.
        cam_intrinsics = torch.tensor(self.dataset.intrinsics.astype(np.float32), device=self.run_config.device)
        semantic_config = SemanticConfig.from_config({**config["semantic"], "debug_info": self.run_config.debug_info})

        # Semantic module and SLAM backend.
        self.ovo = OVO(semantic_config, self.run_config, self.logger, cam_intrinsics)
        self.slam_backbone = get_slam_backbone(config, self.dataset, cam_intrinsics)

        # Visualization Manager
        cam_data = {
            "height": self.dataset.height,
            "width": self.dataset.width,
            "intrinsic": self.dataset.intrinsics,
        }
        self.vis_manager = VisualizationManager(self.vis, cam_data)

        # Optional preprocessing for SAM masks.
        if config["semantic"]["sam"].get("precomputed", False) or config["semantic"]["sam"].get("precompute", False):
            self.ovo.generators.mask.precompute(self.dataset, self.scheduling.segment_every)

        self.first_frame = 0
        if self.run_config.restore_map:
            assert self.run_config.slam_module == "vanilla", "Restoring representation only implemented for 'vanilla' configuration!"
            self.restore_representation()
            self.first_frame = list(self.slam_backbone.estimated_c2ws.keys())[-1] + 1

    def _setup_output_path(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def _init_run_config(self, config: Dict[str, Any]) -> None:
        self.run_config = RunConfig.from_config(config)

    def _init_vis_config(self, config: Dict[str, Any]) -> None:
        self.vis = VisConfig.from_config(config)

    def _init_scheduling(self, config: Dict[str, Any]) -> None:
        self.scheduling = SchedulingConfig.from_config(config)

    def _get_frame_state(self, frame_id: int) -> FrameState | None:
        """Retrieve frame data and track camera pose."""
        frame_data = self.dataset[frame_id]
        self.slam_backbone.track_camera(frame_data)
        
        estimated_c2w = self.slam_backbone.get_c2w(frame_id)
        if estimated_c2w is None:
            return None
            
        return FrameState(frame_id, frame_data, estimated_c2w)

    def _should_run_mapping(self, frame_id: int) -> bool:
        """Check if mapping/fusion should run for current frame."""
        return (frame_id % self.scheduling.map_every == 0 or 
                self.run_config.slam_module == "orbslam2")

    def save_representation(self) -> None:
        """ Saves the current map and scene objects parameters to a checkpoint file.
        This method retrieves the map parameters from the SLAM backbone and the scene
        objects parameters from the OVO module. It then creates a dictionary containing
        these parameters and saves it to a checkpoint file named after the submap ID.
        The checkpoint file is saved in the 'submaps' directory within the specified
        output path.
        """
        map_params = self.slam_backbone.get_map_dict()
        ovo_map_params = self.ovo.capture_dict(debug_info=self.run_config.debug)
        submap_ckpt = {
            "map_params": map_params,
            "ovo_map_params" : ovo_map_params,
        }
        io_utils.save_dict_to_ckpt(
            submap_ckpt, "ovo_map.ckpt", directory=self.output_path)
        if self.run_config.save_estimated_cam:
            c2w = self.slam_backbone.get_cam_dict()
            with open(self.output_path / "estimated_c2w.npy", "wb") as f:
                torch.save(c2w, f)

    def restore_representation(self) -> None:
        """
        Restore map state, semantic state, and optional camera poses from disk.
        """

        ckpt_path = self.output_path / "ovo_map.ckpt"
        assert ckpt_path.exists(), f"Missing required checkpoint to restore: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=self.run_config.device, weights_only=False)

        self.ovo.restore_dict(ckpt["ovo_map_params"], debug_info=self.run_config.debug)
        self.slam_backbone.set_map_dict(ckpt["map_params"])
        
        c2w_path = self.output_path / "estimated_c2w.npy"
        if c2w_path.exists():
            c2w = torch.load(c2w_path)
            self.slam_backbone.set_cam_dict(c2w)
        else:
            print(f"Missing cameras positions to restore: {c2w_path}")
            print("Resotring without cameras positions!")

    def _run_semantic_step(self, frame_state: FrameState) -> float:
        """
        - Run segmentation/semantic update for a frame and optional stream visualization.
        - Args:
            frame_id: Current frame index.
            frame_data: Data for the current frame, typically including RGB and depth information.
            estimated_c2w: Estimated camera-to-world transformation for the current frame.
        - Returns:
            Time taken for the semantic step in seconds.
        """

        # Caputure current map points and instance ids for semantic processing and visualization.
        t_sem_i = time.time()
        with torch.inference_mode() and torch.autocast(device_type=self.run_config.device, dtype=torch.bfloat16):
            if len(frame_state.frame_data) == 5:
                image = frame_state.frame_data[-1]
            else:
                image = frame_state.frame_data[1]

            # If the input image resolution differs from the dataset config, compute the scaling ratio for correct mapping of semantic labels to points.
            if self.dataset.height != image.shape[0] or self.dataset.width != image.shape[1]:
                rgb_depth_ratio = (
                    image.shape[0] / self.dataset.dataset_config["H"],
                    image.shape[1] / self.dataset.dataset_config["W"],
                    self.dataset.crop_edge,
                )
            else:
                rgb_depth_ratio = ()

            # Run OVO semantic segmentation and association for the current frame.
            scene_data = [frame_state.frame_id, image, frame_state.frame_data[2], rgb_depth_ratio]
            map_data = MapData.from_tuple(self.slam_backbone.get_map())
            updated_points_ins_ids = self.ovo.detect_and_track_objects(scene_data, map_data, frame_state.estimated_c2w)

            # If OVO returns updated instance ids for map points, update the SLAM backbone's map representation accordingly.
            if updated_points_ins_ids is not None:
                self.slam_backbone.update_pcd_obj_ids(updated_points_ins_ids)

            # If streaming is enabled, send updated semantic info to visualizer.
            self.ovo.compute_semantic_info()
            self.logger.log_memory_usage(frame_state.frame_id)

        # Synchronize and measure semantic step time.
        t_sem = time.time() - t_sem_i

        # Send updated stream frame to visualizer if in streaming mode, after semantic update.
        self.vis_manager.send_stream_frame(frame_state.frame_id, self.slam_backbone, self.ovo)
        self.vis_manager.handle_semantic_queries(self.ovo)

        return t_sem

    def _run_mapping_and_fusion_step(self, frame_state: FrameState) -> float:
        """
        Run mapping, map fusion, and optional rerun snapshots for one frame.
        Args:
            frame_id: Current frame index.
            frame_data: Data for the current frame, typically including RGB and depth information.
            estimated_c2w: Estimated camera-to-world transformation for the current frame.
        Returns:
            Time taken for the mapping and fusion step in seconds.
        """

        self.slam_backbone.map(frame_state.frame_data, frame_state.estimated_c2w)
        if not self.slam_backbone.map_updated:
            return 0.0

        torch.cuda.synchronize()
        t_lc_i = time.time()
        map_data = MapData.from_tuple(self.slam_backbone.get_map())
        kfs = self.slam_backbone.get_kfs()

        updated_points_ins_ids, fusion_decisions = self.ovo.update_map(map_data, kfs)

        if fusion_decisions:
            self.logger.log_fusion_decisions(frame_state.frame_id, fusion_decisions)

        if updated_points_ins_ids is not None:
            self.slam_backbone.update_pcd_obj_ids(updated_points_ins_ids)

        # Send update_map event to stream visualizer
        n_fused = sum(1 for d in fusion_decisions if d["result"] == "ACCEPTED")
        self.vis_manager.send_update_map(frame_state.frame_id, self.slam_backbone, n_fused, fusion_decisions)

        self.slam_backbone.map_updated = False
        torch.cuda.synchronize()
        t_lc = time.time() - t_lc_i
        return t_lc

    def _should_run_segmentation(self, frame_id: int) -> bool:
        """Check if segmentation/semantic update should run for current frame."""
        return frame_id % self.scheduling.segment_every == 0

    def _should_process_frame(self, frame_id: int) -> bool:
        """Check if frame needs tracking, mapping or segmentation."""
        return (self.scheduling.track_every == 1 or
                frame_id % self.scheduling.track_every == 0 or
                frame_id % self.scheduling.map_every == 0 or
                frame_id % self.scheduling.segment_every == 0)

    def run(self) -> None:
        """
        Starts the main program flow, including tracking and mapping. If stream falg
        is True, a visualizer is launched in a parallel process using the selected visualizaer.
        """

        self.vis_manager.start(self.ovo, str(self.output_path), self.run_config.scene_name, self.vis.show_stream)
        self.logger.start_session()

        try:
            # Main loop over dataset frames
            for frame_id in range(self.first_frame, len(self.dataset)):
                if not self._should_process_frame(frame_id):
                    continue

                if (frame_state := self._get_frame_state(frame_id)) is None:
                    continue

                # Run mapping, fusion and semantic steps on conditions true
                t_lc = self._run_mapping_and_fusion_step(frame_state) if self._should_run_mapping(frame_id) else 0.0
                t_sem = self._run_semantic_step(frame_state) if self._should_run_segmentation(frame_id) else 0.0

                self.logger.record_frame_end(frame_id, t_sem + t_lc)

                if frame_id % 50 == 0:
                    gc.collect()

            # Finalize results
            self.ovo.complete_semantic_info()

            # Wait for visualizer to process remaining messages before shutdown
            self.vis_manager.stop(drain=True)

        finally:
            self.vis_manager.stop(drain=False)

        # Finalize and report statistics
        self.logger.finalize(len(self.dataset))
        self.save_representation()

        self.ovo.cpu()
        del self.slam_backbone, self.ovo
        torch.cuda.empty_cache()