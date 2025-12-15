# 08 - Global Geometric Correction

## Objective

Implement a mechanism to perform a "Global Geometric Correction" at the absolute end of a sequence.

Instead of correcting drift incrementally during the run (via loop closures), the system will allow drift to accumulate, generating multiple spatially separated instances of the same objects. 

Once the sequence finishes, this feature will:
1.  Force all KeyFrame poses to their exact Ground Truth values.
2.  Apply the corresponding corrective transformations to the 3D point clouds associated with each KeyFrame.
3.  Trigger a massive "clean-up" or fusion event where these now-aligned instances are merged.

## Motivation

This approach serves as a robust stress test for the semantic fusion logic (Instance Fusion). By ensuring "perfect geometry" at the final step, we isolate the performance of the semantic merging algorithms (CLIP/DINO comparisons, IoU checks) from any potential SLAM localization errors.

## Technical Implementation

### 1. Architecture Consistency
Analysis of `ovo/entities/instance3d.py` confirms that `Instance3D` objects store **point IDs**, not raw coordinates. This is ideal. It means we only need to update the `pcd` (Point Cloud) and `kfs` (Keyframes) data within `GroundTruthSLAM`. The semantic objects will automatically "inherit" these corrections because they reference the SLAM data by ID.

### 2. Detailed Steps

#### A. Disable Incremental Loop Closure
*   **File:** `ovo/slam/groundtruth_slam.py`
*   **Action:** Modify `map()` to stop calling `_check_for_loop_closure()`. We want drift to accumulate.

#### B. Implement `correct_map_globally()`
*   **File:** `ovo/slam/groundtruth_slam.py`
*   **Action:** Add a new method `correct_map_globally(self)`.
*   **Logic:**
    1.  Iterate through all stored KeyFrames in `self.kfs`.
    2.  For each KeyFrame $i$:
        *   Get `est_pose` (Estimated/Drifted Pose).
        *   Get `gt_pose` (Ground Truth Pose) from `self.trajectory`.
        *   Calculate Correction: $T_i = gt\_pose @ inverse(est\_pose)$.
        *   **Update Pose:** Set `self.kfs[i]["pose"]` and `self.estimated_c2ws[i]` to `gt_pose`.
        *   **Update Points:**
            *   Retrieve point indices: `start, end = self.kfs[i]["pcd_idxs"]`.
            *   Get points slice: `points = self.pcd[start:end]`.
            *   Apply $T_i$ to these points.
            *   Update `self.pcd[start:end]` with transformed points.
    3.  **Set Signals for OVO:**
        *   Set `self.last_big_change_id = 0`. (CRITICAL: Tells OVO the whole map changed from the start).
        *   Set `self.map_updated = True`.

#### C. Triggering the Correction
*   **File:** `run_eval.py`
*   **Action:**
    1.  After the main frame loop finishes, check if the mapper has the method `correct_map_globally`.
    2.  If yes, call `mapper.correct_map_globally()`.
    3.  Immediately after, force a map update processing in the OVO system (e.g., `ovo_sem_map.process_map_update()`) to ensure the fusion logic runs on the corrected map *before* the final evaluation/saving.

## Status

COMPLETED.

## Verification Plan
1.  Run the pipeline on a scene (e.g., `office0`).
2.  Observe (via logs or visualization) that drift accumulates during the run (no loop closure messages).
3.  Verify that at the end of the sequence, a "Global Correction" message appears.
4.  Check the final output: The map should be perfectly aligned with GT, and duplicate instances should have been merged.