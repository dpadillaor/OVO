from typing import Any, Dict, List, Tuple
import torch

from .vanilla_mapper import VanillaMapper
from ..utils import geometry_utils

class GroundTruthSLAM(VanillaMapper):
    """
    This class simulates a SLAM system by using ground truth data from datasets like Replica.
    It inherits from VanillaMapper to reuse the point cloud generation logic.
    Its main purpose is to provide a deterministic environment for developing and testing
    the semantic fusion logic in OVO, without depending on a functional SLAM system.
    """
    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor) -> None:
        """
        Initializes the GroundTruthSLAM.
        - Loads camera intrinsics.
        - Loads the ground truth trajectory for the scene.
        - Initializes thresholds for KeyFrame selection and loop closure.
        """
        super().__init__(config, cam_intrinsics)
        
        dataset_name = self.config["dataset_name"]
        scene_name = self.config["data"]["scene_name"]
        traj_file = f"data/input/Datasets/{dataset_name.capitalize()}/{scene_name}/traj.txt"

        self.trajectory = []
        with open(traj_file, 'r') as f:
            for line in f:
                values = [float(v) for v in line.strip().split()]
                if len(values) == 16:
                    pose = torch.tensor(values).reshape(4, 4)
                    self.trajectory.append(pose)
        
        print(f"Initialized GroundTruthSLAM with {len(self.trajectory)} poses for scene {scene_name}.")

        self._init_noise_params()

        # Set thresholds for KF selection and loop closure from config
        self.kf_dist_thresh = self.config.get("kf_dist_thresh", 0.1)
        self.kf_rot_thresh = self.config.get("kf_rot_thresh", 5.0) # in degrees
        self.lc_dist_thresh = self.config.get("lc_dist_thresh", 0.2)
        self.lc_rot_thresh = self.config.get("lc_rot_thresh", 10.0) # in degrees

        self.map_every = self.config.get("mapping", {}).get("map_every", 10)
        self.correction_done = False

        self.last_big_change_id = -1
        self.kfs = {}
        self.last_processed_frame_id = -1

    def _init_noise_params(self) -> None:
        """
        Initializes noise parameters for simulating sensor noise and sets up the random number generator.
        The noise parameters are read from the configuration, and a seeded torch.Generator
        is created to ensure reproducible noise generation.
        """
        noise_config = self.config.get("noise", {})
        self.noise_enabled = noise_config.get("noise_enabled", False)
        if self.noise_enabled:
            self.translation_noise_std = noise_config.get("translation_noise_std", 0.01)
            self.rotation_noise_std = noise_config.get("rotation_noise_std", 0.5) # in degrees
            self.rotation_noise_std_rad = self.rotation_noise_std * (torch.pi / 180.0)
            self.noise_seed = noise_config.get("noise_seed", 42)
            self.rng = torch.Generator(device=self.device).manual_seed(self.noise_seed)
            self.last_noisy_c2w = None
            print(f"Noise enabled: translation std={self.translation_noise_std}, rotation std={self.rotation_noise_std} degrees, seed={self.noise_seed}")
        else:
            self.translation_noise_std = 0.0
            self.rotation_noise_std = 0.0

    def track_camera(self, frame_data: List[Any]) -> None:
        """
        Tracks the camera's pose for the current frame.
        This function acts as a dispatcher, calling either `_noisy_tracking` or
        `_ground_truth_tracking` based on whether noise simulation is enabled.
        """
        frame_id = frame_data[0]
        if self.noise_enabled:
            self._noisy_tracking(frame_data)
        else:
            self._ground_truth_tracking(frame_data)
        
        # Store the calculated pose (either noisy or ground truth)
        self.estimated_c2ws[frame_id] = self.c2w

    def _ground_truth_tracking(self, frame_data: List[Any]) -> None:
        """
        "Tracks" the camera by directly setting its pose to the ground truth pose
        for the current frame. No noise is applied.
        """
        frame_id = frame_data[0]
        if frame_id < len(self.trajectory):
            self.c2w = self.trajectory[frame_id].to(self.device)
        else:
            # Handle case where we've run out of trajectory data
            print(f"Warning: Frame ID {frame_id} is out of bounds for the trajectory of length {len(self.trajectory)}.")
            self.c2w = self.kfs[list(self.kfs.keys())[-1]]["pose"] # Reuse last known pose

    def _noisy_tracking(self, frame_data: List[Any]) -> None:
        """
        Calculates the camera pose for the current frame by applying cumulative noise
        to the ground truth trajectory. This simulates realistic SLAM drift.
        The noise is generated reproducibly using a seeded generator.
        """
        frame_id = frame_data[0]

        if frame_id >= len(self.trajectory):
            # Fallback for out-of-bounds frames
            print(f"Warning: Frame ID {frame_id} is out of bounds for the trajectory of length {len(self.trajectory)}.")
            self.c2w = self.kfs[list(self.kfs.keys())[-1]]["pose"]
            return

        current_gt_c2w = self.trajectory[frame_id].to(self.device)

        if self.last_processed_frame_id < 0:
            self.c2w = current_gt_c2w
            self.last_noisy_c2w = self.c2w
            self.last_processed_frame_id = frame_id
            return
        
        # Calculate the ideal, perfect relative motion between the last PROCESSED and current ground truth poses.
        prev_gt_c2w = self.trajectory[self.last_processed_frame_id].to(self.device)
        real_relative_motion = geometry_utils.get_relative_pose(prev_gt_c2w, current_gt_c2w)
        
        # Create a random perturbation using the seeded generator for reproducibility.
        translation_noise = torch.randn(3, device=self.device, generator=self.rng) * self.translation_noise_std
        rotation_noise = torch.randn(3, device=self.device, generator=self.rng) * self.rotation_noise_std_rad
        
        # Using axis-angle representation for a more robust rotation perturbation
        rotation_vector = rotation_noise
        rotation_matrix = geometry_utils.rodrigues_rotation_matrix(rotation_vector)
        perturbation = geometry_utils.create_transformation_matrix(rotation_matrix, translation_noise)

        # Apply the perturbation to the ideal relative motion, creating a noisy relative motion and noisy pose
        noisy_relative_motion = real_relative_motion @ perturbation
        self.c2w = self.last_noisy_c2w @ noisy_relative_motion
        self.last_noisy_c2w = self.c2w
        self.last_processed_frame_id = frame_id

    def map(self, frame_data: List[Any], c2w: torch.Tensor) -> None:
        """
        Decides whether to create a new KeyFrame and add points to the map,
        and checks for loop closures.
        """
        frame_id = frame_data[0]

        if self._is_new_keyframe(c2w):
            # 1. Add points to the map using the parent's map method
            super().map(frame_data, c2w)
            
            # 2. Store KeyFrame info
            pcd_end_idx = self.pcd.shape[0]
            pcd_start_idx = self.kfs[list(self.kfs.keys())[-1]]["pcd_idxs"][1] if len(self.kfs) > 0 else 0
            self.kfs[frame_id] = {"id": frame_id, "pcd_idxs": (pcd_start_idx, pcd_end_idx), "pose": c2w}

            # 3. Check for loop closures
            # self._check_for_loop_closure(frame_id, c2w) # DISABLED for Global Correction

        # Trigger Global Correction near the end of the sequence
        # We check if we are within the last 'map_every' window to ensure we catch the final map() call.
        if not self.correction_done and frame_id >= len(self.trajectory) - self.map_every - 1:
             self.correct_map_globally()
             self.correction_done = True

    def _is_new_keyframe(self, current_c2w: torch.Tensor) -> bool:
        """
        Determines if the current frame should be a new KeyFrame based on camera movement.
        """
        if not self.kfs:
            return True

        last_kf_pose = self.kfs[list(self.kfs.keys())[-1]]["pose"]
        
        # Calculate translational distance
        dist = torch.norm(current_c2w[:3, 3] - last_kf_pose[:3, 3])
        
        # Calculate rotational distance
        rot_diff = current_c2w[:3, :3] @ last_kf_pose[:3, :3].T
        trace = torch.trace(rot_diff)
        # Clamp the value to avoid numerical issues with acos
        angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        angle_deg = angle_rad * (180 / torch.pi)

        return dist > self.kf_dist_thresh or angle_deg > self.kf_rot_thresh

    def _check_for_loop_closure(self, current_kf_id: int, current_c2w: torch.Tensor) -> None:
        """
        Checks if the current KeyFrame is close to a previous KeyFrame. If so, it
        simulates a loop closure by calculating the corrective transformation and
        applying it internally to the map and poses.
        """
        # Reset the loop closure signal, it will be set if a loop is found
        self.last_big_change_id = -1

        # Don't check for loop closures until there are enough keyframes
        if len(self.kfs) < 10:
            return

        # Iterate through all keyframes except the last few
        kf_ids = list(self.kfs.keys())
        for old_kf_id in kf_ids[:-5]:
            kf_data = self.kfs[old_kf_id]
            dist = torch.norm(current_c2w[:3, 3] - kf_data["pose"][:3, 3])
            
            if dist < self.lc_dist_thresh:
                rot_diff = current_c2w[:3, :3] @ kf_data["pose"][:3, :3].T
                trace = torch.trace(rot_diff)
                angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
                angle_deg = angle_rad * (180 / torch.pi)

                if angle_deg < self.lc_rot_thresh:
                    print(f"Loop closure detected between KF {current_kf_id} and KF {old_kf_id}")
                    
                    # --- Calculate Transformation T ---
                    pose_est_current = current_c2w
                    pose_est_old = kf_data["pose"]
                    pose_gt_current = self.trajectory[current_kf_id].to(self.device)
                    pose_gt_old = self.trajectory[old_kf_id].to(self.device)
                    T_real_motion = torch.inverse(pose_gt_old) @ pose_gt_current
                    pose_corrected_current = pose_est_old @ T_real_motion
                    T = pose_corrected_current @ torch.inverse(pose_est_current)
                    
                    # --- Apply Internal Correction ---
                    # Find all keyframes and points that occurred after the old keyframe
                    old_kf_index = kf_ids.index(old_kf_id)
                    kfs_to_correct = kf_ids[old_kf_index + 1:]

                    for kf_to_correct_id in kfs_to_correct:
                        # Correct Keyframe Poses
                        self.kfs[kf_to_correct_id]["pose"] = T @ self.kfs[kf_to_correct_id]["pose"]
                        self.estimated_c2ws[kf_to_correct_id] = T @ self.estimated_c2ws[kf_to_correct_id]

                        # Correct Associated Point Cloud Slice
                        pcd_indices = self.kfs[kf_to_correct_id]["pcd_idxs"]
                        pcd_slice = self.pcd[pcd_indices[0]:pcd_indices[1]]
                        
                        # Add homogeneous coordinate for transformation
                        pcd_slice_hom = torch.cat([pcd_slice, torch.ones((pcd_slice.shape[0], 1), device=self.device)], dim=1)
                        
                        # Apply transformation
                        pcd_slice_transformed = (T @ pcd_slice_hom.T).T
                        
                        # Update the main point cloud
                        self.pcd[pcd_indices[0]:pcd_indices[1]] = pcd_slice_transformed[:, :3]

                    print(f"Applied geometric correction to {len(kfs_to_correct)} keyframes and their points.")
                    self.last_big_change_id = old_kf_id
                    self.map_updated = True
                    break # Found a loop, correction applied, no need to check further

    def correct_map_globally(self) -> None:
        """
        Corrects the entire map (poses and points) by forcing all KeyFrames to their
        Ground Truth poses. This is intended to be run at the absolute end of the
        sequence to trigger a massive fusion/cleanup event.
        """
        print("Starting Global Geometric Correction...")
        kf_ids = list(self.kfs.keys())
        
        for kf_id in kf_ids:
            # 1. Get current estimated pose and GT pose
            est_pose = self.kfs[kf_id]["pose"]
            gt_pose = self.trajectory[kf_id].to(self.device)
            
            # 2. Calculate correction transformation T: est_pose * T = gt_pose  =>  T = inv(est_pose) * gt_pose
            # Wait, we want to transform points: P_world_corrected = T * P_world_est
            # And we want the new pose to be GT.
            # So, T * est_pose = gt_pose  =>  T = gt_pose * inv(est_pose)
            T = gt_pose @ torch.inverse(est_pose)
            
            # 3. Update Pose
            self.kfs[kf_id]["pose"] = gt_pose
            self.estimated_c2ws[kf_id] = gt_pose
            
            # 4. Update Points
            pcd_indices = self.kfs[kf_id]["pcd_idxs"]
            # Check if there are points for this keyframe
            if pcd_indices[1] > pcd_indices[0]:
                pcd_slice = self.pcd[pcd_indices[0]:pcd_indices[1]]
                
                # Add homogeneous coordinate
                pcd_slice_hom = torch.cat([pcd_slice, torch.ones((pcd_slice.shape[0], 1), device=self.device)], dim=1)
                
                # Apply transformation
                pcd_slice_transformed = (T @ pcd_slice_hom.T).T
                
                # Update the main point cloud
                self.pcd[pcd_indices[0]:pcd_indices[1]] = pcd_slice_transformed[:, :3]

        # 5. CORRECT ALL ESTIMATED POSES (Fixes "sawtooth" effect)
        # Iterate through all frames that have been tracked and stored
        for frame_id in self.estimated_c2ws.keys():
            if frame_id < len(self.trajectory):
                self.estimated_c2ws[frame_id] = self.trajectory[frame_id].to(self.device)

        # 6. Signal OVO that the whole map has changed
        print(f"Global Geometric Correction completed for {len(kf_ids)} keyframes and {len(self.estimated_c2ws)} total frames.")
        self.last_big_change_id = 0 # 0 implies the map changed from the start
        self.map_updated = True
