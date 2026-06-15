from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from ..vanilla_mapper import VanillaMapper
from ...utils import geometry_utils
from .tracking import TrackingStrategy, GroundTruthTracking, JumpTracking, NoisyTracking
from .jump_drift import JumpDriftController
from .keyframes import KeyframeSelector


def load_trajectory(dataset_name: str, scene_name: str) -> List[torch.Tensor]:
    """Load a ground-truth camera trajectory (4x4 poses) from a Replica-style traj.txt."""
    traj_file = f"data/input/Datasets/{dataset_name.capitalize()}/{scene_name}/traj.txt"
    trajectory: List[torch.Tensor] = []
    with open(traj_file, 'r') as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            if len(values) == 16:
                trajectory.append(torch.tensor(values).reshape(4, 4))
    return trajectory


class SimulatedSLAM(VanillaMapper):
    """Simulates a SLAM system on top of ground-truth trajectories.

    Provides a deterministic environment to develop and test OVO's semantic fusion
    without depending on a real SLAM backend. On top of the GT trajectory it can
    inject controllable error models — per-frame noise (NoisyTracking) or discrete
    jump drift (JumpTracking) — and simulate a loop closure via a global geometric
    correction at the end of the sequence (correct_map_globally).

    Responsibilities are split across collaborators: load_trajectory (I/O),
    TrackingStrategy (pose computation), KeyframeSelector (KF decision) and
    JumpDriftController (jump state). This class wires them together.
    """

    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor) -> None:
        super().__init__(config, cam_intrinsics)

        dataset_name = self.config["dataset_name"]
        scene_name = self.config["data"]["scene_name"]
        self.trajectory = load_trajectory(dataset_name, scene_name)
        print(f"Initialized SimulatedSLAM with {len(self.trajectory)} poses for scene {scene_name}.")

        noise_config = self.config.get("noise", {})
        self.noise_enabled = noise_config.get("noise_enabled", False)
        if self.noise_enabled:
            print(f"Noise enabled: translation std={noise_config.get('translation_noise_std', 0.01)}, "
                  f"rotation std={noise_config.get('rotation_noise_std', 0.5)} degrees, "
                  f"seed={noise_config.get('noise_seed', 42)}")

        self.jump_controller = JumpDriftController(noise_config, self.device)
        self.jump_drift_enabled = self.jump_controller.enabled
        self.pending_jump_events: list = []

        self.keyframe_selector = KeyframeSelector(
            self.config.get("kf_dist_thresh", 0.1),
            self.config.get("kf_rot_thresh", 5.0),
        )
        self.tracking_strategy = self._build_tracking_strategy(noise_config)

        self.close_loops = self.config.get("slam", {}).get("close_loops", True)
        print(f"SimulatedSLAM: Loop closure (Global Correction) enabled: {self.close_loops}")

        self.map_every = self.config.get("mapping", {}).get("map_every", 10)
        self.correction_done = False

        self._lc_pcd_before = None    # snapshot of pcd XYZ before correction
        self._lc_traj_before = None   # snapshot of estimated_c2ws before correction

        self.last_big_change_id = -1

        # Points before this index are excluded from the dedup matching in map().
        # Bumped to the current pcd size whenever a jump fires, so re-observed
        # surfaces after the jump create fresh (displaced) points — a "ghost"
        # instance — instead of being absorbed back into the pre-jump map.
        # 0 means dedup against the whole map (default / non-jump behaviour).
        self._dedup_min_idx = 0

    def _build_tracking_strategy(self, noise_config: Dict[str, Any]) -> TrackingStrategy:
        """Select the pose-computation strategy. Noise takes priority over jump drift,
        which takes priority over plain ground truth (preserves prior dispatch order)."""
        if self.noise_enabled:
            rotation_noise_std_rad = noise_config.get("rotation_noise_std", 0.5) * (torch.pi / 180.0)
            return NoisyTracking(
                self.trajectory, self.device,
                noise_config.get("translation_noise_std", 0.01),
                rotation_noise_std_rad,
                noise_config.get("noise_seed", 42),
            )
        if self.jump_drift_enabled:
            return JumpTracking(self.trajectory, self.device, self.jump_controller)
        return GroundTruthTracking(self.trajectory, self.device)

    def _fallback_pose(self) -> torch.Tensor:
        """Pose to reuse when a frame is out of trajectory bounds (last known KF pose)."""
        return self.kfs[list(self.kfs.keys())[-1]]["pose"]

    def track_camera(self, frame_data: List[Any]) -> None:
        """Compute and store the camera pose for the current frame via the active strategy."""
        frame_id = frame_data[0]
        self.c2w = self.tracking_strategy.compute_pose(frame_id, self._fallback_pose)
        self.estimated_c2ws[frame_id] = self.c2w

    def map(self, frame_data: List[Any], c2w: torch.Tensor) -> None:
        """Create a new keyframe and add points to the map when warranted, and
        trigger the end-of-sequence global correction."""
        frame_id = frame_data[0]

        last_kf_pose = self.kfs[list(self.kfs.keys())[-1]]["pose"] if self.kfs else None
        if self.keyframe_selector.is_new_keyframe(c2w, last_kf_pose):
            c2w = self._apply_pending_jump(frame_id, c2w)

            # 1. Unproject depth and add points to the map
            super().map(frame_data, c2w)

            # 2. Store KeyFrame info
            pcd_end_idx = self.pcd.shape[0]
            pcd_start_idx = self.kfs[list(self.kfs.keys())[-1]]["pcd_idxs"][1] if len(self.kfs) > 0 else 0
            self.kfs[frame_id] = {"id": frame_id, "pcd_idxs": (pcd_start_idx, pcd_end_idx), "pose": c2w}

        self._maybe_correct_at_end(frame_id)

    def _apply_pending_jump(self, frame_id: int, c2w: torch.Tensor) -> torch.Tensor:
        """At a new keyframe, trigger any jump configured for this KF index and return
        the (possibly jump-adjusted) pose for the keyframe being stored."""
        if not self.jump_drift_enabled:
            return c2w

        kf_count = len(self.kfs)  # index of the keyframe about to be created
        events = self.jump_controller.maybe_trigger(kf_count)
        if not events:
            return c2w

        self.pending_jump_events.extend(events)
        # Open a new map "epoch": points added from now on must not dedup against
        # the pre-jump map, so the re-observed (now displaced) surfaces are added
        # as a fresh ghost instead of being absorbed back into the old points.
        self._dedup_min_idx = self.pcd.shape[0]
        # Retroactively reflect the jump on the stored pose for this frame.
        gt_pose = self.trajectory[frame_id].to(self.device) if frame_id < len(self.trajectory) else c2w
        c2w = self.jump_controller.offset @ gt_pose
        self.estimated_c2ws[frame_id] = c2w
        return c2w

    def _add_points(self, frame_data: List[Any], c2w: torch.Tensor) -> None:
        """Unproject the frame's depth into world points and append them to the map.

        Mirrors VanillaMapper.map's unprojection/append, but with an epoch-aware
        dedup policy (see _dedup_mask). A jump opens a new epoch, so re-observed
        surfaces after it are kept as fresh (displaced) points rather than matched
        away against the pre-jump map.
        """
        image, depth = frame_data[1], frame_data[2]
        depth = torch.from_numpy(depth.astype(np.float32)).to(self.device)

        mask = self._dedup_mask(depth, c2w)
        if mask.sum() == 0:
            return

        points, colors = self._unproject(image, depth, mask, c2w)
        self._append(points, colors)

    def _dedup_mask(self, depth: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        """Pixel mask of depth values to project: valid depth minus pixels already
        covered by points of the current epoch (`self.pcd[self._dedup_min_idx:]`).

        This is the single dedup-policy seam: epoch-awareness lives here and nowhere
        else, so a jump's displaced re-observations are not matched away."""
        mask = depth > 0

        candidates = self.pcd[self._dedup_min_idx:]
        if candidates.shape[0] > 0:
            camera_frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
            # compute_frustum_point_ids already returns indices into `candidates`, not a bool mask.
            frustum_local_indices = geometry_utils.compute_frustum_point_ids(candidates, camera_frustum_corners, device=self.device)
            frustum_subset = candidates[frustum_local_indices]
            frustum_global_indices = frustum_local_indices + self._dedup_min_idx
            matched_mask, matches = geometry_utils.match_3d_points_to_2d_pixels(depth, torch.linalg.inv(c2w), frustum_subset, self.cam_intrinsics, self.match_distance_th)
            self.pcd_obs[frustum_global_indices[matched_mask]] += 1
            mask[matches[:,1], matches[:,0]] = False
            mask = self.pooling(mask)

        return mask

    def _unproject(self, image, depth: torch.Tensor, mask: torch.Tensor, c2w: torch.Tensor):
        """Back-project the masked depth pixels to world coordinates.
        Returns (points_xyz [N,3], colors [N,3] uint8)."""
        h, w = image.shape[0], image.shape[1]
        y, x = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )

        y, x = self.downscale(y), self.downscale(x)
        depth, mask, image = self.downscale(depth), self.downscale(mask), self.downscale(image)

        x = x[mask]
        y = y[mask]
        depth = depth[mask]

        x_3d = (x - self.cam_intrinsics[0, 2]) * depth / self.cam_intrinsics[0, 0]
        y_3d = (y - self.cam_intrinsics[1, 2]) * depth / self.cam_intrinsics[1, 1]
        z_3d = depth

        points = torch.hstack((x_3d.reshape(-1, 1), y_3d.reshape(-1, 1), z_3d.reshape(-1, 1),
                               torch.ones((x_3d.shape[0], 1), device=self.device)))
        points = torch.einsum("ij,mj->mi", c2w, points)

        colors = torch.from_numpy(image.astype(np.uint8)).to(self.device)[mask].reshape(-1, 3)
        return points[:, :3], colors

    def _append(self, points: torch.Tensor, colors: torch.Tensor) -> None:
        """Append new world points (and their colors) to the map arrays, unlabelled."""
        n_new = points.shape[0]
        self.pcd = torch.vstack((self.pcd, points))
        self.pcd_ids = torch.vstack((self.pcd_ids, torch.arange(self.max_id, self.max_id + n_new, device=self.device, dtype=torch.int32).unsqueeze(1)))
        self.pcd_obj_ids = torch.vstack((self.pcd_obj_ids, torch.ones((n_new, 1), device=self.device, dtype=torch.int32) * -1))
        self.pcd_colors = torch.vstack((self.pcd_colors, colors))
        self.pcd_obs = torch.vstack((self.pcd_obs, torch.ones((n_new, 1), device=self.device, dtype=torch.int32)))
        self.max_id += n_new

    def _maybe_correct_at_end(self, frame_id: int) -> None:
        """Trigger the global correction once, within the last `map_every` window.
        Only needed when noise or jump drift is active; otherwise tracking already
        uses GT poses."""
        if self.correction_done or frame_id < len(self.trajectory) - self.map_every - 1:
            return

        if self.close_loops:
            if self.noise_enabled or self.jump_drift_enabled:
                self.correct_map_globally()
            else:
                self.map_updated = True
        else:
            self.map_updated = False
        self.correction_done = True

    def correct_map_globally(self) -> None:
        """Force every keyframe to its ground-truth pose, moving its points accordingly.
        Run once at the end of the sequence to trigger a massive fusion/cleanup event
        (a simulated loop closure)."""
        print("Starting Global Geometric Correction...")
        n_kfs = len(self.kfs)

        self._snapshot_before_correction()
        self._snap_keyframes_to_gt()
        self._reset_estimated_poses_to_gt()
        self._reset_drift_state()

        # Signal OVO that the whole map changed (0 implies from the start).
        self.last_big_change_id = 0
        self.map_updated = True
        print(f"Global Geometric Correction completed for {n_kfs} keyframes and {len(self.estimated_c2ws)} total frames.")

    def _snapshot_before_correction(self) -> None:
        """Capture pcd XYZ and trajectory before correction, for visualization."""
        self._lc_pcd_before = self.pcd.clone().cpu()
        self._lc_traj_before = {k: v.clone().cpu() for k, v in self.estimated_c2ws.items()}

    def _snap_keyframes_to_gt(self) -> None:
        """Set every keyframe pose to its GT and move its point-cloud slice accordingly."""
        for kf_id in list(self.kfs.keys()):
            est_pose = self.kfs[kf_id]["pose"]
            gt_pose = self.trajectory[kf_id].to(self.device)
            # T moves estimated world points onto GT: T @ est_pose = gt_pose.
            T = gt_pose @ torch.inverse(est_pose)
            self.kfs[kf_id]["pose"] = gt_pose
            self.estimated_c2ws[kf_id] = gt_pose
            self._transform_pcd_slice(self.kfs[kf_id]["pcd_idxs"], T)

    def _reset_estimated_poses_to_gt(self) -> None:
        """Snap all tracked (non-keyframe) poses to GT — fixes the trajectory sawtooth."""
        for frame_id in self.estimated_c2ws.keys():
            if frame_id < len(self.trajectory):
                self.estimated_c2ws[frame_id] = self.trajectory[frame_id].to(self.device)

    def _reset_drift_state(self) -> None:
        """After correction the map is consistent: clear the jump offset and dedup epoch."""
        if self.jump_drift_enabled:
            self.jump_controller.reset()
            self._dedup_min_idx = 0

    def _transform_pcd_slice(self, pcd_idxs: Tuple[int, int], T: torch.Tensor) -> None:
        """Apply a rigid transform to the point-cloud slice [start, end) of a keyframe."""
        start, end = pcd_idxs
        if end <= start:
            return
        pcd_slice = self.pcd[start:end]
        pcd_slice_hom = torch.cat([pcd_slice, torch.ones((pcd_slice.shape[0], 1), device=self.device)], dim=1)
        self.pcd[start:end] = (T @ pcd_slice_hom.T).T[:, :3]
        # Rotate the slice's normals too (rotation part of T only).
        if self.pcd_normals.shape[0] >= end:
            self.pcd_normals[start:end] = (T[:3, :3] @ self.pcd_normals[start:end].T).T
