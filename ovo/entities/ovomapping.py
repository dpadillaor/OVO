from __future__ import annotations
from typing import Any, Dict
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
    elif backbone == "simulated":
        from ..slam.simulated import SimulatedSLAM
        return SimulatedSLAM(config, cam_intrinsics, dataset=dataset)
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
        self.device = config.get("device", "cuda")
        self.dataset_name = config["dataset_name"]

        # Visualization configuration.
        self.stream = config["vis"]["stream"]
        self.show_stream = config["vis"].get("show_stream", False)
        self.vis_type = config["vis"].get("type", "open3d")
        self.rerun_mode = config["vis"].get("rerun_mode", "stream")  # "stream" or "fusion"
        self.rerun_visual_mode = resolve_rerun_visual_mode(config["vis"].get("rerun_visual_mode", None), self.show_stream)
        self.save_rrd = config["vis"].get("save_rrd", False)

        # Scheduling configuration.
        self.map_every = config["mapping"].get("map_every", 10)
        self.segment_every = config["semantic"].get("segment_every", 10)
        tracking_cfg = config.get("tracking", None)
        self.track_every = 1 if tracking_cfg is None else tracking_cfg.get("track_every", 1)

        # Core services: logger and dataset.
        self.logger = Logger(self.output_path, os.getpid(), config["use_wandb"])
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})

        # Camera intrinsics and semantic defaults.
        cam_intrinsics = torch.tensor(self.dataset.intrinsics.astype(np.float32), device=self.device)
        config["semantic"]["debug_info"] = self.config.get("debug_info", False)
        config["semantic"]["fusion_method"] = config["semantic"].get("fusion_method", "CLIP")

        # Semantic module and SLAM backend.
        self.ovo = OVO(config["semantic"], self.logger, config["data"]["scene_name"], cam_intrinsics, device=self.device)
        self.ovo.contest.set_output_dir(self.output_path)
        self.slam_backbone = get_slam_backbone(config, self.dataset, cam_intrinsics)

        # Optional preprocessing for SAM masks.
        if config["semantic"]["sam"].get("precomputed", False) or config["semantic"]["sam"].get("precompute", False):
            self.ovo.mask_generator.precompute(self.dataset, self.segment_every)

        # Optional map restoration state.
        self.first_frame = 0
        if self.config.get("restore_map", False):
            assert config["slam"].get("slam_module", "vanilla") == "vanilla", "Restoring representation only implemented for 'vanilla' configuration!"
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

    def save_representation(self) -> None:
        """ Saves the current map and scene objects parameters to a checkpoint file.
        This method retrieves the map parameters from the SLAM backbone and the scene
        objects parameters from the OVO module. It then creates a dictionary containing
        these parameters and saves it to a checkpoint file named after the submap ID.
        The checkpoint file is saved in the 'submaps' directory within the specified
        output path.
        """
        map_params = self.slam_backbone.get_map_dict()
        ovo_map_params = self.ovo.capture_dict(debug_info=self.config.get("debug", False))
        submap_ckpt = {
            "map_params": map_params,
            "ovo_map_params" : ovo_map_params,
        }
        io_utils.save_dict_to_ckpt(
            submap_ckpt, "ovo_map.ckpt", directory=self.output_path)    
        self.ovo.contest.dump(str(self.output_path / "contest.json"))
        self.ovo.contest.dump_verdicts(str(self.output_path / "contest_verdicts.csv"))
        if self.config["slam"].get("save_estimated_cam", False):
            c2w = self.slam_backbone.get_cam_dict()
            with open(self.output_path / "estimated_c2w.npy", "wb") as f:
                torch.save(c2w, f)

    def restore_representation(self) -> None:
        """
        Restore map state, semantic state, and optional camera poses from disk.
        """

        ckpt_path = self.output_path / "ovo_map.ckpt"
        assert ckpt_path.exists(), f"Missing required checkpoint to restore: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.ovo.restore_dict(ckpt["ovo_map_params"], debug_info=self.config.get("debug", False))
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
        if not self.stream:
            return None, None, None, None

        cam_data = {
            "height": self.dataset.height,
            "width": self.dataset.width,
            "intrinsic": self.dataset.intrinsics,
        }
        mpqueue = mp.Queue(maxsize=8)
        query_flag = mp.Value('i', 0)  # 0 idle, 1 requested, 2 completed
        query_pipe, vis_pipe = mp.Pipe()
        target_func, proc_name = select_visualizer_target(self.vis_type, self.rerun_mode)

        if self.vis_type == "rerun":
            query_payload = [query_flag, vis_pipe, self.rerun_visual_mode]
            proc_args = (
                self.ovo,
                mpqueue,
                query_payload,
                cam_data,
                self.config["data"]["scene_name"],
                self.logger.output_path,
                self.show_stream,
                self.save_rrd,
            )
        else:
            query_payload = [query_flag, vis_pipe]
            proc_args = (
                self.ovo,
                mpqueue,
                query_payload,
                cam_data,
                self.config["data"]["scene_name"],
                self.logger.output_path,
                self.show_stream,
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
        if not self.stream or self.rerun_mode != "stream":
            return

        pcd, _, pcd_obj_ids = self.slam_backbone.get_map()
        c2w = self.slam_backbone.get_c2w(frame_id)
        if c2w is None:
            return

        c2w_np = c2w.cpu().numpy().astype(np.float16)
        colors = self.slam_backbone.get_pcd_colors()
        normals = self.slam_backbone.get_point_normals()
        normals_np = normals.cpu().numpy().astype(np.float16) if normals is not None and normals.shape[0] == pcd.shape[0] else None
        visual_snapshot = self.ovo.get_last_visual_snapshot()

        rgb = None
        ins_map = None
        assigned_ins_map = None
        sam_map = None
        kf_id = None
        if visual_snapshot is not None and visual_snapshot.get("frame_id") == frame_id:
            rgb = visual_snapshot.get("rgb")
            ins_map = visual_snapshot.get("ins_map")
            assigned_ins_map = visual_snapshot.get("assigned_ins_map")
            sam_map = visual_snapshot.get("sam_map")
            kf_id = visual_snapshot.get("kf_id")

        corrected_trajectory = None
        if (
            getattr(self.slam_backbone, "correction_done", False)
            and not getattr(self, "_stream_traj_reset_done", False)
        ):
            corrected_trajectory = [
                v.cpu().numpy()[:3, 3].tolist()
                for _, v in sorted(self.slam_backbone.estimated_c2ws.items())
            ]
            self._stream_traj_reset_done = True

        _queue_put_dropping(
            mpqueue,
            {
                "type": "stream_frame",
                "frame_id": frame_id,
                "points": pcd.cpu().numpy().astype(np.float16),
                "obj_ids": pcd_obj_ids.cpu().numpy().astype(np.int16),
                "colors": colors,
                "normals": normals_np,
                "c2w": c2w_np,
                "rgb": rgb,
                "ins_map": ins_map,
                "assigned_ins_map": assigned_ins_map,
                "sam_map": sam_map,
                "kf_id": kf_id,
                "corrected_trajectory": corrected_trajectory,
            },
        )

    def _run_semantic_step(
        self, frame_id: int, frame_data, estimated_c2w: torch.Tensor, mpqueue, query_flag, query_pipe
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
        if frame_id % self.segment_every != 0:
            return 0.0

        t_sem_i = time.time()
        with torch.inference_mode() and torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            if len(frame_data) == 5:
                image = frame_data[-1]
            else:
                image = frame_data[1]

            if self.dataset.height != image.shape[0] or self.dataset.width != image.shape[1]:
                rgb_depth_ratio = (
                    image.shape[0] / self.dataset.dataset_config["H"],
                    image.shape[1] / self.dataset.dataset_config["W"],
                    self.dataset.crop_edge,
                )
            else:
                rgb_depth_ratio = ()

            scene_data = [frame_id, image, frame_data[2], rgb_depth_ratio]

            map_data = self.slam_backbone.get_map()
            updated_points_ins_ids = self.ovo.detect_and_track_objects(scene_data, map_data, estimated_c2w)

            if updated_points_ins_ids is not None:
                self.slam_backbone.update_pcd_obj_ids(updated_points_ins_ids)

            self.ovo.compute_semantic_info()
            self.logger.log_memory_usage(frame_id)

        t_sem = time.time() - t_sem_i

        self._send_stream_frame(frame_id, mpqueue)

        if self.stream and self.rerun_mode == "stream":
            if query_flag.value == 1:
                query = query_pipe.recv()
                self.ovo.complete_semantic_info()
                query_map = self.ovo.query(query).cpu().numpy()
                query_map[query_map < 0] = 0
                with query_flag.get_lock():
                    query_pipe.send(query_map)
                    query_flag.value = 2

        return t_sem

    def _dispatch_jump_events(self, frame_id: int, mpqueue) -> None:
        if not self.stream or self.rerun_mode != "stream":
            return
        pending = getattr(self.slam_backbone, "pending_jump_events", [])
        if not pending:
            return
        c2w = self.slam_backbone.get_c2w(frame_id)
        if c2w is None:
            return
        c2w_np = c2w.cpu().numpy().astype(np.float32)
        for event in pending:
            _queue_put_dropping(mpqueue, {
                "type": "jump_event",
                "frame_id": frame_id,
                "c2w": c2w_np,
                "kf_index": event["kf_index"],
                "translation_magnitude": event["translation_magnitude"],
                "rotation_magnitude": event["rotation_magnitude"],
            })
        self.slam_backbone.pending_jump_events = []

    def _run_mapping_and_fusion_step(
        self,
        frame_id: int,
        frame_data,
        estimated_c2w: torch.Tensor,
        mpqueue,
    ) -> float:
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

        if frame_id % self.map_every != 0 and self.config["slam"]["slam_module"] != "orbslam2":
            return 0.0

        self.slam_backbone.map(frame_data, estimated_c2w)
        self._dispatch_jump_events(frame_id, mpqueue)
        if not self.slam_backbone.map_updated:
            return 0.0
        torch.cuda.synchronize()
        t_lc_i = time.time()
        map_data = self.slam_backbone.get_map()
        kfs = self.slam_backbone.get_kfs()
        point_obs = self.slam_backbone.get_point_observations()
        point_normals = self.slam_backbone.get_point_normals()
        point_colors = torch.as_tensor(self.slam_backbone.get_pcd_colors())

        # Send "before fusion" snapshot to stream visualizer
        self._send_stream_frame(frame_id, mpqueue)

        noise_cfg = self.config.get("noise", {})
        if noise_cfg.get("jump_drift_enabled", False) and noise_cfg.get("save_pre_fusion_checkpoint", False):
            self._save_pre_fusion_checkpoint(frame_id)

        updated_points_ins_ids, fusion_decisions, t_fusion, criterion_times = self.ovo.update_map(map_data, kfs, point_obs, point_normals=point_normals, point_colors=point_colors)

        if fusion_decisions:
            self.logger.log_fusion_decisions(frame_id, fusion_decisions)
        _EXTRA_STAT_KEYS = {"t_precompute_fusion", "t_descriptor_update", "n_instances_alive", "n_pairs_evaluated"}
        extra_stats = {k: v for k, v in criterion_times.items() if k in _EXTRA_STAT_KEYS or k.startswith("sc_")}
        crit_only = {k: v for k, v in criterion_times.items() if k not in extra_stats}
        self.logger.log_fusion_timings(t_fusion, crit_only)
        if extra_stats:
            self.logger.log_ovo_stats(extra_stats)

        if updated_points_ins_ids is not None:
            self.slam_backbone.update_pcd_obj_ids(updated_points_ins_ids)
            # Send "after fusion" snapshot to stream visualizer
            self._send_stream_frame(frame_id, mpqueue)

        # Send update_map event to stream visualizer
        if self.stream and self.rerun_mode == "stream":
            c2w = self.slam_backbone.get_c2w(frame_id)
            if c2w is not None:
                _queue_put_dropping(mpqueue, {
                    "type": "update_map",
                    "frame_id": frame_id,
                    "c2w": c2w.cpu().numpy().astype(np.float32),
                    "n_fused": sum(1 for d in fusion_decisions if d["result"] == "ACCEPTED"),
                    "decisions": fusion_decisions,
                })

        # Send loop closure snapshot to visualizer
        if (self.stream and self.rerun_mode in ("loop_closure", "fusion") and getattr(self.slam_backbone, '_lc_pcd_before', None) is not None):
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

        self.slam_backbone.map_updated = False
        torch.cuda.synchronize()
        t_lc = time.time() - t_lc_i
        print(f"Sem LC update took {t_lc};")
        self.logger.log_ovo_stats({"t_loop_closure_refusion": round(t_lc, 3)})
        return t_lc

    def _save_pre_fusion_checkpoint(self, frame_id: int) -> None:
        """Snapshot geometric + semantic state just before fusion runs.

        Drains the segmentation queue first so the saved state is complete.
        Only called when jump_drift_enabled and save_pre_fusion_checkpoint are set.
        Saved to data/checkpoints/<experiment>/<scene>/pre_fusion.ckpt.
        """
        self.ovo.complete_semantic_info()

        noise_cfg = self.config.get("noise", {})
        ckpt_root = Path("data/checkpoints")
        # Mirror output_path structure under data/checkpoints/
        try:
            rel = self.output_path.relative_to("data/output")
        except ValueError:
            rel = Path(self.output_path.name)
        ckpt_dir = ckpt_root / rel
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        controller = getattr(self.slam_backbone, "jump_controller", None)
        jump_events = controller.fired_events[:] if controller is not None else []

        ckpt = {
            "frame_id": frame_id,
            "jump_events": jump_events,
            "map_params": self.slam_backbone.get_map_dict(),
            "cam_params": self.slam_backbone.get_cam_dict(),
            "kfs_params": self.slam_backbone.get_kfs_dict(),
            "ovo_params": self.ovo.capture_dict(debug_info=True),
            "config": self.config,
        }
        ckpt_path = ckpt_dir / "pre_fusion.ckpt"
        io_utils.save_dict_to_ckpt(ckpt, "pre_fusion.ckpt", directory=ckpt_dir)
        print(f"Pre-fusion checkpoint saved to {ckpt_path}")

    def run_fusion_from_checkpoint(self, ckpt_path: str) -> None:
        """Restore pre-fusion state from a checkpoint and run only the fusion step.

        Replaces the normal run() flow: no tracking, no mapping, no segmentation.
        Useful for re-running fusion with different parameters on a saved jump-drift state.
        """
        ckpt_path = Path(ckpt_path)
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.slam_backbone.set_map_dict(ckpt["map_params"])
        self.slam_backbone.set_cam_dict(ckpt["cam_params"])
        self.slam_backbone.set_kfs_dict(ckpt["kfs_params"])
        self.ovo.restore_dict(ckpt["ovo_params"], debug_info=True)

        source_output = ckpt["config"].get("output_path")
        if source_output:
            self.logger.load_stats_from(source_output)

        map_data = self.slam_backbone.get_map()
        kfs = self.slam_backbone.get_kfs()
        frame_id = ckpt["frame_id"]

        print(f"Running fusion from checkpoint (frame_id={frame_id})...")
        point_obs = self.slam_backbone.get_point_observations()
        point_normals = self.slam_backbone.get_point_normals()
        point_colors = torch.as_tensor(self.slam_backbone.get_pcd_colors())
        updated_points_ins_ids, fusion_decisions, t_fusion, criterion_times = self.ovo.update_map(map_data, kfs, point_obs, point_normals=point_normals, point_colors=point_colors)

        if fusion_decisions:
            self.logger.log_fusion_decisions(frame_id, fusion_decisions)

        if updated_points_ins_ids is not None:
            self.slam_backbone.update_pcd_obj_ids(updated_points_ins_ids)

        _EXTRA_STAT_KEYS = {"t_precompute_fusion", "t_descriptor_update", "n_instances_alive", "n_pairs_evaluated"}
        extra_stats = {k: v for k, v in criterion_times.items() if k in _EXTRA_STAT_KEYS or k.startswith("sc_")}
        crit_only = {k: v for k, v in criterion_times.items() if k not in extra_stats}
        self.logger.log_fusion_timings(t_fusion, crit_only)
        if extra_stats:
            self.logger.log_ovo_stats(extra_stats)
        self.logger.log_ovo_stats({"t_loop_closure_refusion": round(t_fusion, 3)})

        self.logger.write_stats()
        self.logger.print_final_stats()
        self.save_representation()

        self.ovo.cpu()
        del self.slam_backbone, self.ovo
        torch.cuda.empty_cache()
        print(f"Fusion replay complete. Results saved to {self.output_path}")
    


    def run(self) -> None:
        """
        Starts the main program flow, including tracking and mapping. If stream falg
        is True, a visualizer is launched in a parallel process using the selected visualizaer.
        """

        spf = []

        with mp.Manager() as manager: 
            p, mpqueue, query_flag, query_pipe = self._start_visualization_process()

            torch.cuda.synchronize()
            t_start = time.time()

            try:
                # Main loop over dataset frames
                for frame_id in range(self.first_frame, len(self.dataset)):
                    if self.track_every == 1 or frame_id%self.track_every==0 or frame_id%self.map_every==0 or frame_id%self.segment_every==0:
                        frame_data = self.dataset[frame_id]
                        self.slam_backbone.track_camera(frame_data)

                        estimated_c2w = self.slam_backbone.get_c2w(frame_id)
                        missing_depth = not (frame_data[2]>0).any()
                        if estimated_c2w is None or missing_depth :
                            continue
                        t_lc = self._run_mapping_and_fusion_step(
                            frame_id,
                            frame_data,
                            estimated_c2w,
                            mpqueue,
                        )
                        t_sem = self._run_semantic_step(
                            frame_id,
                            frame_data,
                            estimated_c2w,
                            mpqueue,
                            query_flag,
                            query_pipe,
                        )
                        if t_sem+t_lc > 0:
                            spf.append(t_sem + t_lc)         

                        if frame_id % 50 == 0:
                            gc.collect()
                    
                self.ovo.complete_semantic_info()
                
                torch.cuda.synchronize()
                t_end = time.time()
                fps = len(self.dataset)/self.segment_every/(t_end-t_start)
                if self.stream and p.is_alive():
                    while mpqueue.qsize()>0 and p.is_alive():
                        if query_flag.value == 1:
                            query = query_pipe.recv()
                            query_map = self.ovo.query(query).cpu().numpy()
                            with query_flag.get_lock():
                                query_pipe.send(query_map)
                                query_flag.value = 2
                        time.sleep(2)
                    time.sleep(5)

            finally:
                # Always clean up the visualizer process, even on Ctrl+C or crash
                if self.stream and p.is_alive():
                    mpqueue.put(None)  # Signal the visualizer to exit gracefully
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=3)
                    if p.is_alive():
                        p.kill()
                
        if 'fps' not in dir():
            fps = 0
            t_end = time.time()
        self.logger.log_fps(fps)
        for s in spf:
            self.logger.log_spf(s)
        self.logger.log_ovo_stats({"total_time": round(t_end - t_start, 3)})
        self.logger.log_max_memory_usage()
        self.logger.write_stats()
        self.logger.print_final_stats()

        self.save_representation()

        self.ovo.cpu()
        del self.slam_backbone, self.ovo
        torch.cuda.empty_cache()