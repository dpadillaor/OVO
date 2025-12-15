# Directory Summary: OVO SLAM Wrappers

This directory (`ovo/slam/`) contains wrappers and interfaces for integrating different Simultaneous Localization and Mapping (SLAM) backbones with the OVO system. These wrappers abstract away the specifics of each SLAM implementation, providing a consistent interface for OVO to consume camera poses, point clouds, and loop closure signals.

## Key Files:
*   `vanilla_mapper.py`: Base mapper component for basic point cloud management.
*   `orbslam2.py`: Interface for ORB-SLAM2/3. Inherits from `VanillaMapper`.
*   `gaussian_slam.py`: Interface for Gaussian-SLAM.
*   `groundtruth_slam.py`: Ground truth SLAM implementation for testing and replay. Inherits from `VanillaMapper`. Includes drift simulation and loop closure correction mechanisms.

## Purpose:
The `slam` directory provides all geometric information (camera poses, point clouds, loop closures) to the OVO system. It enables OVO to be modular with respect to the underlying SLAM technology, facilitating experimentation with different localization and mapping solutions.

