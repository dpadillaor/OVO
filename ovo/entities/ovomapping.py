from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import csv
import torch
import torch.multiprocessing as mp
from pathlib import Path
import numpy as np
import time
import os
import gc

from .logger import Logger
from .ovo import OVO
from .datasets import get_dataset
from .visualizers.selection import resolve_rerun_visual_mode, select_visualizer_target
from ..slam.vanilla_mapper import VanillaMapper
from ..utils import io_utils

_FUSION_LOG_FIELDS = ["frame_id", "result", "i1", "i2", "reason", "centroid_dist", "cos_sim", "p_dist"]

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
    rerun_mode: str = "stream"
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
            rerun_mode=vis.get("rerun_mode", "stream"),
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

def _queue_put_dropping(queue, item) -> None:
    """Put item in queue, dropping oldest item if full. Never blocks."""
    while True:
        try:
            queue.put_nowait(item)
            return
        except Exception:
            try:
                queue.get_nowait()
            except Exception:
                return


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
        self.logger = Logger(self.output_path, os.getpid(), self.run.use_wandb)
        self.dataset = get_dataset(self.run.dataset_name)({**config["data"], **config["cam"]})

        # Camera intrinsics and semantic module.
        cam_intrinsics = torch.tensor(self.dataset.intrinsics.astype(np.float32), device=self.run.device)
        semantic_config = {**config["semantic"], "debug_info": self.run.debug_info}

        # Semantic module and SLAM backend.
        self.ovo = OVO(semantic_config, self.logger, self.run.scene_name, cam_intrinsics, device=self.run.device)
        self.slam_backbone = get_slam_backbone(config, self.dataset, cam_intrinsics)

        # Optional preprocessing for SAM masks.
        if config["semantic"]["sam"].get("precomputed", False) or config["semantic"]["sam"].get("precompute", False):
            self.ovo.mask_generator.precompute(self.dataset, self.scheduling.segment_every)

        # Optional map restoration state.
        self._init_fusion_log()

        self.first_frame = 0
        if self.run.restore_map:
            assert self.run.slam_module == "vanilla", "Restoring representation only implemented for 'vanilla' configuration!"
            self.restore_representation()
            self.first_frame = list(self.slam_backbone.estimated_c2ws.keys())[-1] + 1

    def _setup_output_path(self, output_path: str) -> None:
        """
        Sets up the output path for saving results based on the provided configuration. 
            - Args:
                config: A dictionary containing the experiment configuration including data and output path information.
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def _init_run_config(self, config: Dict[str, Any]) -> None:
        self.run = RunConfig.from_config(config)

    def _init_vis_config(self, config: Dict[str, Any]) -> None:
        self.vis = VisConfig.from_config(config)

    def _init_scheduling(self, config: Dict[str, Any]) -> None:
        self.scheduling = SchedulingConfig.from_config(config)

    @property
    def _fusion_log_path(self) -> Path:
        return self.output_path / "fusion_decisions.csv"

    def _init_fusion_log(self) -> None:
        with open(self._fusion_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_FUSION_LOG_FIELDS).writeheader()

    def _write_fusion_decisions(self, frame_id: int, decisions: list) -> None:
        with open(self._fusion_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FUSION_LOG_FIELDS)
            for d in decisions:
                writer.writerow({"frame_id": frame_id, **d})

    def save_representation(self) -> None:
        """ Saves the current map and scene objects parameters to a checkpoint file.
        This method retrieves the map parameters from the SLAM backbone and the scene
        objects parameters from the OVO module. It then creates a dictionary containing
        these parameters and saves it to a checkpoint file named after the submap ID.
        The checkpoint file is saved in the 'submaps' directory within the specified
        output path.
        """
        map_params = self.slam_backbone.get_map_dict()
        ovo_map_params = self.ovo.capture_dict(debug_info=self.run.debug)
        submap_ckpt = {
            "map_params": map_params,
            "ovo_map_params" : ovo_map_params,
        }
        io_utils.save_dict_to_ckpt(
            submap_ckpt, "ovo_map.ckpt", directory=self.output_path)
        if self.run.save_estimated_cam:
            c2w = self.slam_backbone.get_cam_dict()
            with open(self.output_path / "estimated_c2w.npy", "wb") as f:
                torch.save(c2w, f)

    def restore_representation(self) -> None:
        """
        Restore map state, semantic state, and optional camera poses from disk.
        """

        ckpt_path = self.output_path / "ovo_map.ckpt"
        assert ckpt_path.exists(), f"Missing required checkpoint to restore: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=self.run.device, weights_only=False)

        self.ovo.restore_dict(ckpt["ovo_map_params"], debug_info=self.run.debug)
        self.slam_backbone.set_map_dict(ckpt["map_params"])
        
        c2w_path = self.output_path / "estimated_c2w.npy"
        if c2w_path.exists():
            c2w = torch.load(c2w_path)
            self.slam_backbone.set_cam_dict(c2w)
        else:
            print(f"Missing cameras positions to restore: {c2w_path}")
            print("Resotring without cameras positions!")

    def _start_visualization_process(self) -> tuple[Any, Any, Any, Any]:
        """
        Initialize visualization process and return IPC/process handles.
        - Returns:
            tuple[Any, Any, Any, Any]:
                - process handle `p`
                - queue handle `mpqueue`
                - shared query state `query_flag`
                - main-side pipe endpoint `query_pipe`

            If streaming is disabled, returns (None, None, None, None).
        """
        if not self.vis.stream:
            return None, None, None, None

        cam_data = {
            "height": self.dataset.height,
            "width": self.dataset.width,
            "intrinsic": self.dataset.intrinsics,
        }
        mpqueue = mp.Queue(maxsize=8)
        query_flag = mp.Value('i', 0)  # 0 idle, 1 requested, 2 completed
        query_pipe, vis_pipe = mp.Pipe()
        target_func, proc_name = select_visualizer_target(self.vis.vis_type, self.vis.rerun_mode)

        if self.vis.vis_type == "rerun":
            query_payload = [query_flag, vis_pipe, self.vis.rerun_visual_mode]
            proc_args = (
                self.ovo,
                mpqueue,
                query_payload,
                cam_data,
                self.run.scene_name,
                self.logger.output_path,
                self.vis.show_stream,
                self.vis.save_rrd,
            )
        else:
            query_payload = [query_flag, vis_pipe]
            proc_args = (
                self.ovo,
                mpqueue,
                query_payload,
                cam_data,
                self.run.scene_name,
                self.logger.output_path,
                self.vis.show_stream,
            )

        p = mp.Process(target=target_func, args=proc_args, name=proc_name)
        p.start()
        return p, mpqueue, query_flag, query_pipe

    def _capture_points_and_ids(self, map_data: tuple[Any, Any, Any], points_dtype=np.float16, ids_dtype=np.int16) -> tuple[Any, Any]:
        """
        Extract points and instance ids from map data as CPU numpy arrays.
         - Args:
            map_data: Tuple containing points, colors, and instance ids from the SLAM backbone's map representation.
            points_dtype: Desired numpy dtype for the points array (default: np.float16).
            ids_dtype: Desired numpy dtype for the instance ids array (default: np.int16).
        - Returns:
            Tuple of (points, instance_ids) as numpy arrays on CPU with specified dtypes.
        """
        points, _, ids = map_data
        points_np = points.cpu().numpy().astype(points_dtype)
        ids_np = ids.cpu().numpy().astype(ids_dtype)
        return points_np, ids_np

    def _send_stream_frame(self, frame_id: int, mpqueue) -> None:
        """Capture and send a stream frame snapshot to the visualizer."""
        pcd, _, pcd_obj_ids = self.slam_backbone.get_map()
        c2w = self.slam_backbone.get_c2w(frame_id)
        if c2w is None:
            return
        
        c2w_np = c2w.cpu().numpy().astype(np.float16)
        colors = self.slam_backbone.get_pcd_colors()
        visual_snapshot = self.ovo.get_last_visual_snapshot()

        rgb = None
        ins_map = None
        sam_map = None
        if visual_snapshot is not None and visual_snapshot.get("frame_id") == frame_id:
            rgb = visual_snapshot.get("rgb")
            ins_map = visual_snapshot.get("ins_map")
            sam_map = visual_snapshot.get("sam_map")

        _queue_put_dropping(
            mpqueue,
            {
                "type": "stream_frame",
                "frame_id": frame_id,
                "points": pcd.cpu().numpy().astype(np.float16),
                "obj_ids": pcd_obj_ids.cpu().numpy().astype(np.int16),
                "colors": colors,
                "c2w": c2w_np,
                "rgb": rgb,
                "ins_map": ins_map,
                "sam_map": sam_map,
            },
        )

    def _run_semantic_step(
        self, frame_state: FrameState, mpqueue, query_flag, query_pipe
    ) -> float:
        """
        - Run segmentation/semantic update for a frame and optional stream visualization.
        - Args:
            frame_id: Current frame index.
            frame_data: Data for the current frame, typically including RGB and depth information.
            estimated_c2w: Estimated camera-to-world transformation for the current frame.
            mpqueue: Multiprocessing queue for sending visualization data to the visualizer process.
            query_flag: Shared flag indicating if a semantic query has been requested by the visualizer.
            query_pipe: Pipe for receiving semantic queries and sending back results to the visualizer.
        - Returns:
            Time taken for the semantic step in seconds.
        """

        # Caputure current map points and instance ids for semantic processing and visualization.
        t_sem_i = time.time()
        with torch.inference_mode() and torch.autocast(device_type=self.run.device, dtype=torch.bfloat16):
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
            map_data = self.slam_backbone.get_map()
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
        if self.vis.stream:
            if self.vis.rerun_mode == "stream":
                self._send_stream_frame(frame_state.frame_id, mpqueue)
            if query_flag.value == 1:
                self._handle_semantic_query(query_pipe, query_flag)

        return t_sem

    def _handle_semantic_query(self, query_pipe, query_flag) -> None:
        """Process a semantic query from visualizer and send back results."""
        query = query_pipe.recv()
        self.ovo.complete_semantic_info()
        query_map = self.ovo.query(query).cpu().numpy()
        query_map[query_map < 0] = 0
        with query_flag.get_lock():
            query_pipe.send(query_map)
            query_flag.value = 2

    def _send_loop_closure_event(self, frame_id: int, mpqueue) -> None:
        if not (self.vis.stream and self.vis.rerun_mode in ("loop_closure", "fusion") and getattr(self.slam_backbone, '_lc_pcd_before', None) is not None):
            return
        pcd_after, ids = self._capture_points_and_ids(self.slam_backbone.get_map(), points_dtype=np.float32, ids_dtype=np.int32)
        traj_after = {
            k: v.cpu().numpy().astype(np.float32)
            for k, v in self.slam_backbone.estimated_c2ws.items()
        }
        _queue_put_dropping(mpqueue, {
            "type": "loop_closure",
            "pcd_before": self.slam_backbone._lc_pcd_before.numpy().astype(np.float32),
            "pcd_after": pcd_after,
            "ids": ids,
            "traj_before": {
                k: v.numpy().astype(np.float32)
                for k, v in self.slam_backbone._lc_traj_before.items()
            },
            "traj_after": traj_after,
            "frame_id": frame_id,
        })
        self.slam_backbone._lc_pcd_before = None
        self.slam_backbone._lc_traj_before = None

    def _run_mapping_and_fusion_step(self, frame_state: FrameState, mpqueue,) -> float:
        """
        Run mapping, map fusion, and optional rerun snapshots for one frame.
        Args:
            frame_id: Current frame index.
            frame_data: Data for the current frame, typically including RGB and depth information.
            estimated_c2w: Estimated camera-to-world transformation for the current frame.
            mpqueue: Multiprocessing queue for sending visualization data to the visualizer process.
        Returns:
            Time taken for the mapping and fusion step in seconds.
        """

        self.slam_backbone.map(frame_state.frame_data, frame_state.estimated_c2w)
        if not self.slam_backbone.map_updated:
            return 0.0

        torch.cuda.synchronize()
        t_lc_i = time.time()
        map_data = self.slam_backbone.get_map()
        kfs = self.slam_backbone.get_kfs()

        # Send "before fusion" snapshot to stream visualizer
        self._send_stream_frame(frame_state.frame_id, mpqueue)

        updated_points_ins_ids, fusion_decisions = self.ovo.update_map(map_data, kfs)

        if fusion_decisions:
            self._write_fusion_decisions(frame_state.frame_id, fusion_decisions)

        if updated_points_ins_ids is not None:
            self.slam_backbone.update_pcd_obj_ids(updated_points_ins_ids)
            # Send "after fusion" snapshot to stream visualizer
            self._send_stream_frame(frame_state.frame_id, mpqueue)

        # Send update_map event to stream visualizer
        if self.vis.stream and self.vis.rerun_mode == "stream":
            c2w = self.slam_backbone.get_c2w(frame_state.frame_id)
            if c2w is not None:
                _queue_put_dropping(mpqueue, {
                    "type": "update_map",
                    "frame_id": frame_state.frame_id,
                    "c2w": c2w.cpu().numpy().astype(np.float32),
                    "n_fused": sum(1 for d in fusion_decisions if d["result"] == "ACCEPTED"),
                    "decisions": fusion_decisions,
                })

        # Send loop closure snapshot to visualizer
        self._send_loop_closure_event(frame_state.frame_id, mpqueue)

        self.slam_backbone.map_updated = False
        torch.cuda.synchronize()
        t_lc = time.time() - t_lc_i
        print(f"Sem LC update took {t_lc};")
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

    def _log_final_stats(self, spf: list, fps: float, duration: float) -> None:
        """Centralized final logging and stats reporting."""
        self.logger.log_total_time(duration)
        self.logger.log_fps(fps)
        self.logger.log_spf(spf)
        self.logger.log_max_memory_usage()
        self.logger.write_stats()
        self.logger.print_final_stats()

    def run(self) -> None:
        """
        Starts the main program flow, including tracking and mapping. If stream falg
        is True, a visualizer is launched in a parallel process using the selected visualizaer.
        """

        spf = []
        fps = 0
        p, mpqueue, query_flag, query_pipe = self._start_visualization_process()

        torch.cuda.synchronize()
        t_start = time.time()

        try:
            # Main loop over dataset frames
            for frame_id in range(self.first_frame, len(self.dataset)):
                if not self._should_process_frame(frame_id):
                    continue

                frame_state = self._get_frame_state(frame_id)
                if frame_state is None:
                    continue
                            
                # Run mapping, fusion and semantic steps
                t_lc = self._run_mapping_and_fusion_step(frame_state, mpqueue) if self._should_run_mapping(frame_id) else 0.0
                t_sem = self._run_semantic_step(frame_state, mpqueue, query_flag, query_pipe) if self._should_run_segmentation(frame_id) else 0.0

                if t_sem + t_lc > 0:
                    spf.append(t_sem + t_lc)

                if frame_id % 50 == 0:
                    gc.collect()

            # After processing all frames, if streaming is enabled, wait for the visualizer process to finish processing any remaining messages and handle any outstanding semantic queries before shutting down the visualizer process gracefully.
            self.ovo.complete_semantic_info()
            # If streaming, wait for visualizer to process remaining messages and handle outstanding queries before shutdown
            torch.cuda.synchronize()
            t_end = time.time()
            fps = len(self.dataset)/self.scheduling.segment_every/(t_end-t_start)

            # If streaming is enabled, wait for visualizer to process remaining messages and handle outstanding queries before shutdown
            if self.vis.stream and p.is_alive():
                while mpqueue.qsize()>0 and p.is_alive():
                    if query_flag.value == 1:
                        query = query_pipe.recv()
                        query_map = self.ovo.query(query).cpu().numpy()
                        with query_flag.get_lock():
                            query_pipe.send(query_map)
                            query_flag.value = 2
                    time.sleep(2)
                time.sleep(5)

        # Clean up visualizer process on exit, ensuring it is terminated gracefully.
        finally:
            if self.vis.stream and p.is_alive():
                mpqueue.put(None)  # Signal the visualizer to exit gracefully
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=3)
                if p.is_alive():
                    p.kill()

        # Finalize and report statistics
        self._log_final_stats(spf, fps, t_end - t_start)
        self.save_representation()

        self.ovo.cpu()
        del self.slam_backbone, self.ovo
        torch.cuda.empty_cache()