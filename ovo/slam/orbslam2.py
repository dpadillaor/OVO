from typing import Any, Dict, List
import time
import orbslam3 as orbslam
import torch
from pathlib import Path

from .vanilla_mapper import VanillaMapper


def convert_pose(traj, device):
    pose = torch.cat([
        torch.tensor(traj[-12:], device = device).reshape((3,4)),
        torch.tensor([[0,0,0,1]], device = device)
    ])
    return pose


class WrapperORBSLAM2(VanillaMapper):
    """This class uses ORB-SLAM 2 to estimate camera posses and generates a vanilla point-cloud reconstruction by unprojecting depths"""
    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor, world_ref=torch.eye(4)) -> None:
        super().__init__(config, cam_intrinsics)

        self.close_loops = config["slam"].get("close_loops", True)
        self.last_big_change_id = 0
        self.last_map_change_id = 0  # mnMapChange: bumps on local BA too (not just LC/GBA)
        self.map_updated = False
        self.geometry_refreshed = False  # light refresh moved poses w/o map_updated (viz traj signal)
        # Local-BA geometry refresh: move the dense cloud to follow local-BA pose updates between big
        # changes (off => original behaviour, only refresh on loop-closure/GBA).
        self.localba_refresh_enabled = config["slam"].get("localba_refresh", True)
        # A KF whose points would move less than this (m) is left untouched on refresh: below it the
        # transform is just inv() numerical noise, and re-applying it each poll drifts old KFs.
        self.localba_min_disp = config["slam"].get("localba_refresh_min_disp", 0.005)
        # Refresh internal profiler (getkf / per-KF loop / full pcd concat), accumulated over the run.
        self._profile_refresh = config["slam"].get("profile_refresh", False)
        self.refresh_prof = {"getkf": 0.0, "loop": 0.0, "cat": 0.0, "n": 0}
        self.world_ref = world_ref.to(self.device)
        self.kfs = {}

        configs_path = Path(config["slam"]["config_path"]) / "orbslam3"
        vocab_path = configs_path  / "vocabulary" / "ORBvoc.txt"
        print(f"Loading ORB-SLAM2 vocabulary from {vocab_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"ORB-SLAM2 vocabulary not found at {vocab_path}!")
        else:
            print("ORB-SLAM2 vocabulary found.")
            print(f"Loading ORB-SLAM2 vocabulary from {vocab_path}")

        if (configs_path/ config["dataset_name"].lower()/ f"{config['data']['scene_name']}.yaml").exists():
            orbslam_config_path = configs_path / config["dataset_name"].lower()/ f"{config['data']['scene_name']}.yaml"
        else:
            orbslam_config_path = configs_path / f"{config['dataset_name']}.yaml"

        self.orbslam = orbslam.System(str(vocab_path), str(orbslam_config_path), orbslam.Sensor.RGBD, config["slam"].get("use_viewer",False), not self.close_loops)
        self.orbslam.initialize()

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id, rgb_image, depth_image = frame_data[:3]
        tframe = frame_id
        self.orbslam.process_image_rgbd(rgb_image, depth_image, tframe) # This actually blocks untill tracking is completed
        tracking_state = self.orbslam.get_tracking_state()
        if tracking_state == orbslam.TrackingState.OK:
            orb_c2w = self.orbslam.get_last_trajectory_point()
            assert int(orb_c2w[0]) == frame_id, "Retrieved wrong frame pose" # This should never happen
            self.estimated_c2ws[frame_id] = self.world_ref@convert_pose(orb_c2w, device = self.device)
        else:
            print(f"Tracking state: {tracking_state}!")
        return 
    
    def map(self, frame_data, c2w) -> None:
        # check if frame is a KeyFrame
        if self.orbslam.is_last_frame_kf(): # If tracking and maping are parallelized, this call would have a racing condition with ORB-SLAM2 tracking thread
            frame_id = frame_data[0]
            first_p_idx = self.pcd_ids.shape[0]
            super().map(frame_data, c2w)
            last_p_idx = self.pcd_ids.shape[0]
            self.kfs[frame_id] = {"id": frame_id , "pcd_idxs":(first_p_idx, last_p_idx)} # Assumes pcd is not pruned outside of self._update_map

        # detect loop-closure of GBA
        last_big_change_id = self.orbslam.get_last_big_change_idx()
        # LC and GBA happen one after the other, we could save some computation detecting only GBA
        if self.close_loops and last_big_change_id != self.last_big_change_id:
            self.last_big_change_id = last_big_change_id
            self.update_map()

    def refresh_geometry_if_local_ba(self) -> bool:
        """Rebuild the dense cloud if local BA moved poses since the last poll, WITHOUT
        triggering semantic re-fusion. mnMapChange (unlike mnBigChangeIdx) bumps on local BA.
        Returns True if the geometry was rebuilt."""
        if not self.localba_refresh_enabled:
            return False
        map_change_id = self.orbslam.get_map_change_index()
        if map_change_id == self.last_map_change_id:
            return False
        self.update_map(trigger_refusion=False)
        self.geometry_refreshed = True  # tell the viz to re-draw the corrected trajectory
        return True

    def _psync(self):
        if self._profile_refresh:
            torch.cuda.synchronize()
        return time.time()

    def update_map(self, trigger_refusion: bool = True):
        prof = self._profile_refresh
        if trigger_refusion or prof:  # quiet on the frequent light refresh; loud on big change
            print("Updating dense map ...")
        # update kfs and pcd poses:
        _t = self._psync()
        updated_kfs = self.orbslam.get_keyframe_points()
        if prof: self.refresh_prof["getkf"] += self._psync() - _t

        _t = self._psync()
        new_kfs = {}
        new_pcd = []
        new_pcd_ids = []
        new_pcd_obj_ids = []
        new_pcd_colors = []
        new_pcd_obs = []
        new_pcd_normals = []
        new_c2w = {}

        # Gather the KFs we still track, in ORB's order (pruned KFs simply drop out).
        entries = []  # (kf_id, s0, s1, updated_kf)
        for updated_kf in updated_kfs:
            kf_id = int(updated_kf[0])
            kf = self.kfs.get(kf_id)
            if kf is None:
                # Only deleted in update_map, so shouldn't still be in ORB's list — skip defensively.
                continue
            s0, s1 = kf["pcd_idxs"]
            entries.append((kf_id, s0, s1, updated_kf))

        # Vectorized skip test. A KF's point displacement equals its camera-centre shift
        # (transform @ old_centre == new_centre), so skip-vs-move is decided from a single batched
        # centre diff — NO per-KF convert_pose / inv / matmul. Only the few moved KFs pay the full
        # transform below. A KF ORB left fixed yields a ~mm residual (inv of a non-orthonormal
        # reconstructed rotation); re-applying it every refresh would drift old KFs, so we skip it.
        moved_mask = []
        if entries:
            arr = torch.tensor([list(e[3]) for e in entries], device=self.device, dtype=self.world_ref.dtype)  # (M,13)
            new_centres = arr[:, [4, 8, 12]] @ self.world_ref[:3, :3].T + self.world_ref[:3, 3]  # (M,3)
            old_centres = torch.stack([self.estimated_c2ws[e[0]][:3, 3] for e in entries])        # (M,3)
            disp = torch.linalg.norm(new_centres - old_centres, dim=1)                             # (M,)
            moved_mask = (disp >= self.localba_min_disp).tolist()

        n_points = 0
        n_moved = 0
        for idx, (kf_id, s0, s1, updated_kf) in enumerate(entries):
            old_n_points = n_points
            n_points += (s1 - s0)
            new_kfs[kf_id] = {"id": kf_id, "pcd_idxs": (old_n_points, n_points)}

            if not moved_mask[idx]:
                # unchanged: reuse the existing slice as-is, keep the old baseline
                new_pcd.append(self.pcd[s0:s1])
                new_pcd_normals.append(self.pcd_normals[s0:s1])
                new_c2w[kf_id] = self.estimated_c2ws[kf_id]
            else:
                n_moved += 1
                kf_c2w = self.estimated_c2ws[kf_id]
                updated_kf_c2w = self.world_ref @ convert_pose(updated_kf[1:13], device=self.device)
                transform = updated_kf_c2w @ torch.linalg.inv(kf_c2w)
                updated_kf_pcd = torch.einsum('mn,bn->bm', transform, torch.cat([self.pcd[s0:s1], torch.ones((s1 - s0, 1), device=self.device)], dim=1))[:, :3]
                # Normals are directions: rotate only (no translation) by the kf transform.
                updated_kf_normals = torch.einsum('ij,bj->bi', transform[:3, :3], self.pcd_normals[s0:s1])
                new_pcd.append(updated_kf_pcd)
                new_pcd_normals.append(updated_kf_normals)
                new_c2w[kf_id] = updated_kf_c2w

            # kfs that are not in updated_kfs were pruned by ORB_SLAM -> dropped with their pcd
            new_pcd_ids.append(self.pcd_ids[s0:s1])
            new_pcd_obj_ids.append(self.pcd_obj_ids[s0:s1])
            new_pcd_colors.append(self.pcd_colors[s0:s1])
            new_pcd_obs.append(self.pcd_obs[s0:s1])

        if prof:
            self.refresh_prof["loop"] += self._psync() - _t
            print(f"  refreshed geometry: {n_moved}/{len(new_kfs)} keyframes moved (> {self.localba_min_disp*1000:.0f}mm)")
        _t = self._psync()
        self.estimated_c2ws = new_c2w
        self.kfs = new_kfs
        self.pcd = torch.cat(new_pcd, dim=0)
        self.pcd_ids = torch.cat(new_pcd_ids, dim=0)
        self.pcd_obj_ids = torch.cat(new_pcd_obj_ids, dim=0)
        self.pcd_colors = torch.cat(new_pcd_colors, dim=0)
        self.pcd_obs = torch.cat(new_pcd_obs, dim=0)
        self.pcd_normals = torch.cat(new_pcd_normals, dim=0)
        if prof:
            self.refresh_prof["cat"] += self._psync() - _t
            self.refresh_prof["n"] += 1
        # Keep the local-BA guard in sync for both paths (LC/GBA also bumps mnMapChange).
        self.last_map_change_id = self.orbslam.get_map_change_index()
        # Only the heavy path flags a semantic re-fusion; the light refresh leaves it untouched.
        if trigger_refusion:
            self.map_updated = True

    

    def __del__(self) -> None:
        self.orbslam.shutdown()