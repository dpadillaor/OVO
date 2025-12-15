# Directory Summary: OVO Main Module

This directory (`ovo/`) contains the core components and logic for the Open-Vocabulary Online Semantic Mapping (OVO) project. It integrates various functionalities including semantic instance management, SLAM wrappers, and utility functions.

## Subdirectories:
*   `entities/`: Orchestrates the integration between SLAM geometric information and semantic segmentation. Handles 2D segmentation, semantic feature extraction, 3D instance creation and tracking, instance merging, and visualization.
*   `slam/`: Provides geometric information (poses, point clouds, loop closures) through wrappers for different SLAM backends.
*   `utils/`: Provides utility functions supporting various aspects of the OVO pipeline, such as geometry, I/O, and instance management.

## Key Purpose:
The `ovo/` directory is central to the project, orchestrating the creation, tracking, and merging of 3D semantic instances.