from typing import Any, Dict, List, NamedTuple, Optional, Set
import time
import orbslam3 as orbslam
import torch
from pathlib import Path

from .vanilla_mapper import VanillaMapper


class _KFEntry(NamedTuple):
    """One keyframe taking part in a cloud rebuild.

    `pose` is ORB's fresh pose tuple, or None when the keyframe sits in an archived map and
    must be carried over untouched.
    """
    kf_id: int
    s0: int
    s1: int
    pose: Optional[Any]

    @property
    def frozen(self) -> bool:
        return self.pose is None


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
        self.localba_refresh_enabled = config["slam"].get("localba_refresh", False)
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
        # Archived-map protection needs the active map id (see _classify_keyframes). Older
        # builds of the binding lack it; degrade to the previous behaviour instead of failing.
        self._supports_map_id = hasattr(self.orbslam, "get_current_map_id")
        if not self._supports_map_id:
            print("ORB-SLAM binding has no get_current_map_id(): a new Atlas map will drop the"
                  " cloud built so far. Rebuild thirdParty/ORB_SLAM3 to enable the guard.")

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
            # map_id records which Atlas map the KF was born in, so a later rebuild can tell a
            # culled keyframe (drop its points) from an archived one (keep them). See
            # _classify_keyframes. Assumes pcd is not pruned outside of update_map.
            self.kfs[frame_id] = {"id": frame_id, "pcd_idxs": (first_p_idx, last_p_idx),
                                  "map_id": self._current_map_id()}

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

    def _current_map_id(self) -> Optional[int]:
        """Id of the Atlas' active map, or None when the binding cannot report it."""
        if not self._supports_map_id:
            return None
        return int(self.orbslam.get_current_map_id())

    def _classify_keyframes(self, updated_kfs, map_id: Optional[int]) -> List[_KFEntry]:
        """Decide, for every KF we track, whether it takes part in the rebuild and how.

        ORB only lists the ACTIVE map, so a KF missing from that listing means one of two very
        different things, and they must not be conflated:
          - it was culled as redundant  -> its points are stale, drop them (the old behaviour);
          - its map was archived after a tracking loss -> the KF is alive elsewhere, so freeze
            its slice and pose until a map merge brings it back.
        The per-KF `map_id` tag tells them apart. It is refreshed on every listing, because a
        merge makes the surviving map adopt the other's id (LoopClosing::MergeLocal), so only a
        tag kept up to date stays meaningful for the KFs that are absent.
        """
        entries: List[_KFEntry] = []
        listed: Set[int] = set()

        for updated_kf in updated_kfs:
            kf_id = int(updated_kf[0])
            kf = self.kfs.get(kf_id)
            if kf is None:
                # Only deleted in update_map, so shouldn't still be in ORB's list -- skip defensively.
                continue
            listed.add(kf_id)
            kf["map_id"] = map_id
            s0, s1 = kf["pcd_idxs"]
            entries.append(_KFEntry(kf_id, s0, s1, updated_kf))

        if map_id is not None:
            for kf_id, kf in self.kfs.items():
                tag = kf.get("map_id")
                if kf_id in listed or tag is None or tag == map_id:
                    # listed, of unknown provenance (restored from a checkpoint), or culled from
                    # the map we are in: nothing to rescue, keep the previous behaviour.
                    continue
                s0, s1 = kf["pcd_idxs"]
                entries.append(_KFEntry(kf_id, s0, s1, None))

        entries.sort(key=lambda e: e.kf_id)  # keep the cloud laid out in keyframe order
        return entries

    def _moved_kf_ids(self, entries: List[_KFEntry]) -> Set[int]:
        """Ids of the listed KFs whose camera centre shifted at least `localba_min_disp`.

        Point displacement equals the camera-centre shift, so this is a batched centre diff (no
        per-KF inv/matmul); only the KFs it returns pay the transform. See design doc §9c.
        Frozen KFs never appear here: their pose is the one we already applied.
        """
        live = [e for e in entries if not e.frozen]
        if not live:
            return set()
        arr = torch.tensor([list(e.pose) for e in live], device=self.device, dtype=self.world_ref.dtype)  # (M,13)
        new_centres = arr[:, [4, 8, 12]] @ self.world_ref[:3, :3].T + self.world_ref[:3, 3]  # (M,3)
        old_centres = torch.stack([self.estimated_c2ws[e.kf_id][:3, 3] for e in live])        # (M,3)
        disp = torch.linalg.norm(new_centres - old_centres, dim=1)                            # (M,)
        return {e.kf_id for e, moved in zip(live, (disp >= self.localba_min_disp).tolist()) if moved}

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
        new_c2w = {}

        map_id = self._current_map_id()
        entries = self._classify_keyframes(updated_kfs, map_id)
        moved = self._moved_kf_ids(entries)

        n_points = 0
        n_moved = 0
        for kf_id, s0, s1, updated_kf in entries:
            old_n_points = n_points
            n_points += (s1 - s0)
            new_kfs[kf_id] = {"id": kf_id, "pcd_idxs": (old_n_points, n_points),
                              "map_id": self.kfs[kf_id].get("map_id")}

            if kf_id not in moved:
                # unchanged, or frozen in an archived map: reuse the slice as-is and keep the pose
                new_pcd.append(self.pcd[s0:s1])
                new_c2w[kf_id] = self.estimated_c2ws[kf_id]
            else:
                n_moved += 1
                kf_c2w = self.estimated_c2ws[kf_id]
                updated_kf_c2w = self.world_ref @ convert_pose(updated_kf[1:13], device=self.device)
                transform = updated_kf_c2w @ torch.linalg.inv(kf_c2w)
                updated_kf_pcd = torch.einsum('mn,bn->bm', transform, torch.cat([self.pcd[s0:s1], torch.ones((s1 - s0, 1), device=self.device)], dim=1))[:, :3]
                new_pcd.append(updated_kf_pcd)
                new_c2w[kf_id] = updated_kf_c2w

            # kfs missing from BOTH the listing and self.kfs' live maps were culled -> dropped
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