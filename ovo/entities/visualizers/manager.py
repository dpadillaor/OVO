from __future__ import annotations
import torch
import torch.multiprocessing as mp
import numpy as np
import time
from typing import Any, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.sharedctypes import Synchronized
    from ..ovo import OVO
    from ...slam.vanilla_mapper import VanillaMapper
    from ..ovomapping import VisConfig

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

class VisualizationManager:
    """Manages visualization process lifecycle, IPC, and data serialization."""

    def __init__(self, vis_config: VisConfig, cam_data: Dict[str, Any]):
        self.vis = vis_config
        self.cam_data = cam_data
        
        # IPC handles
        self.process: Optional[mp.Process] = None
        self.mpqueue: Optional[mp.Queue] = None
        self.query_flag: Optional[Synchronized] = None
        self.query_pipe: Optional[Connection] = None

    def start(self, ovo: OVO, output_path: str, scene_name: str, show_stream: bool):
        """Initialize and start the visualization process."""
        if not self.vis.stream:
            return

        self.mpqueue = mp.Queue(maxsize=8)
        self.query_flag = mp.Value('i', 0)  # 0 idle, 1 requested, 2 completed
        self.query_pipe, vis_pipe = mp.Pipe()
        
        from .selection import select_visualizer_target
        target_func, proc_name = select_visualizer_target(self.vis.vis_type)

        if self.vis.vis_type == "rerun":
            query_payload = [self.query_flag, vis_pipe, self.vis.rerun_visual_mode]
            proc_args = (
                ovo,
                self.mpqueue,
                query_payload,
                self.cam_data,
                scene_name,
                output_path,
                show_stream,
                self.vis.save_rrd,
            )
        else:
            query_payload = [self.query_flag, vis_pipe]
            proc_args = (
                ovo,
                self.mpqueue,
                query_payload,
                self.cam_data,
                scene_name,
                output_path,
                show_stream,
            )

        self.process = mp.Process(target=target_func, args=proc_args, name=proc_name)
        self.process.start()

    def stop(self, drain: bool = True):
        """Gracefully shut down the visualization process."""
        if not (self.vis.stream and self.process and self.process.is_alive()):
            return

        if drain:
            try:
                while self.mpqueue.qsize() > 0 and self.process.is_alive():
                    if self.query_flag.value == 1:
                        self.handle_semantic_queries(None) # ovo is not needed for loop closure check
                    time.sleep(1)
            except Exception:
                pass

        try:
            self.mpqueue.put(None)
        except Exception:
            pass
            
        self.process.join(timeout=5)
        
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)

    def send_stream_frame(self, frame_id: int, slam_backbone: VanillaMapper, ovo: OVO):
        """Capture and send a stream frame snapshot to the visualizer."""
        if not self.vis.stream:
            return
            
        pcd, _, pcd_obj_ids = slam_backbone.get_map()
        c2w = slam_backbone.get_c2w(frame_id)
        if c2w is None:
            return
        
        c2w_np = c2w.cpu().numpy().astype(np.float16)
        colors = slam_backbone.get_pcd_colors()
        visual_snapshot = ovo.get_last_visual_snapshot()

        rgb = None
        ins_map = None
        sam_map = None
        if visual_snapshot is not None and visual_snapshot.get("frame_id") == frame_id:
            rgb = visual_snapshot.get("rgb")
            ins_map = visual_snapshot.get("ins_map")
            sam_map = visual_snapshot.get("sam_map")

        _queue_put_dropping(
            self.mpqueue,
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

    def handle_semantic_queries(self, ovo: Optional[OVO]):
        """Process a semantic query from visualizer and send back results."""
        if not self.vis.stream or self.query_flag.value != 1:
            return
            
        if ovo is None:
            # When draining, we just consume the request if possible but can't compute
            return

        query = self.query_pipe.recv()
        ovo.complete_semantic_info()
        query_map = ovo.query(query).cpu().numpy()
        query_map[query_map < 0] = 0
        with self.query_flag.get_lock():
            self.query_pipe.send(query_map)
            self.query_flag.value = 2

    def send_update_map(self, frame_id: int, slam_backbone: VanillaMapper, n_fused: int, decisions: list):
        """Send update_map event to stream visualizer."""
        if not self.vis.stream:
            return
            
        c2w = slam_backbone.get_c2w(frame_id)
        if c2w is not None:
            _queue_put_dropping(self.mpqueue, {
                "type": "update_map",
                "frame_id": frame_id,
                "c2w": c2w.cpu().numpy().astype(np.float32),
                "n_fused": n_fused,
                "decisions": decisions,
            })
