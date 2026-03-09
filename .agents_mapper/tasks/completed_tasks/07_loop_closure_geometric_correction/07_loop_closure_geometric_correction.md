# 07 - Loop Closure Geometric Correction

## Objective

Implement a realistic simulation of geometric correction during loop closure events. This will allow the instance fusion algorithm to be tested under significant trajectory drift, which is a more robust and realistic scenario than the current low-drift simulation.

This will be achieved by using the **Direct Calculation Method**, which leverages the availability of ground truth data to calculate the required alignment transformation without needing a complex optimization algorithm. The geometric correction will be handled *internally* within the `GroundTruthSLAM` module to be consistent with the project's existing architecture (e.g., `WrapperORBSLAM2`).

## Status: CANCELLED

**Reason:** Following a discussion with the project tutor, the strategy has shifted. Instead of simulating incremental loop closures during the sequence, the project will now focus on a "Big Bang" fusion approach. The system will allow drift to accumulate naturally throughout the entire sequence to generate maximum duplicate instances. At the very end of the run, a **Global Geometric Correction** will be applied (forcing all poses to Ground Truth), triggering a massive instance fusion event.

This new approach is tracked in **Feature 08 - Global Geometric Correction**.

## Theoretical Explanation (Archived)

The core of this task is to find a 4x4 transformation matrix, which we'll call `T`. The purpose of `T` is to correct the positions of points in the map that have been misplaced due to simulated trajectory drift.

### The Information We Have

At the moment of a loop closure between a `Keyframe_Old` and a `Keyframe_Current`, our simulation has access to four crucial pieces of information:

1.  `pose_gt_old`: The perfect Ground Truth pose of `Keyframe_Old`.
2.  `pose_estimated_old`: The drifted (our estimated) pose of `Keyframe_Old`.
3.  `pose_gt_current`: The perfect Ground Truth pose of `Keyframe_Current`.
4.  `pose_estimated_current`: The drifted (our estimated) pose of `Keyframe_Current`.

### The Direct Calculation of `T`

We do not need to "guess" or "optimize" to find `T`. We can calculate it directly in a single step by constructing the "correct" pose and finding the difference with our "drifted" pose.

**Step 1: Calculate the True Motion**
First, we use the perfect Ground Truth poses to determine what the actual, real-world movement was between the two keyframes.

*   `T_real_motion = inverse(pose_gt_old) @ pose_gt_current`

This matrix perfectly describes how the camera moved in the real world between the first and second sightings of the object.

**Step 2: Calculate the "Corrected" Current Pose**
Next, we calculate where `Keyframe_Current` *should have been* in our map if no new drift had occurred. We do this by taking the starting point of this trajectory segment (the drifted `pose_estimated_old`) and applying the `T_real_motion` to it.

*   `pose_corrected_current = pose_estimated_old @ T_real_motion`

This `pose_corrected_current` represents the ideal position and orientation of the current keyframe within our map's existing drifted coordinate system.

**Step 3: Calculate the Correction Transformation `T`**
At this point, we have two poses for `Keyframe_Current`:
*   `pose_corrected_current` (where it *should* be in our map).
*   `pose_estimated_current` (where our drifted system *thinks* it is).

The transformation required to move points from the incorrect drifted space to the corrected space is our final transformation `T`.

*   `T = pose_corrected_current @ inverse(pose_estimated_current)`