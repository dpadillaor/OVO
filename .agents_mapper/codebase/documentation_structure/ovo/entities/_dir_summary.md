# Directory Summary: OVO Entities

This directory (`ovo/entities/`) defines the core data structures and processing units that constitute the OVO semantic mapping pipeline. Each file typically corresponds to a key component responsible for a specific aspect of instance management, feature extraction, or overall orchestration.

## Key Components:
*   `ovomapping.py`: Contains the `OVOSemMap` class, which orchestrates the entire SLAM + semantics pipeline.
*   `ovo.py`: Contains the `OVO` class, managing instance creation, feature extraction, and merging logic.
*   `instance3d.py`: Defines the `Instance3D` class, representing a single 3D object instance with its properties.
*   `clip_generator.py`: Manages the extraction of CLIP features for instances.
*   `mask_generator.py`: Interfaces with segmentation models (like SAM) to generate 2D masks.
*   `dino_generator.py`: (Under development/integration) Manages the extraction of DINO descriptors.
*   `logger.py`: Handles logging mechanisms for the OVO system.

## Purpose:
The `entities` directory is crucial for defining the "what" and "who" of the OVO system's internal representation and processing, establishing the building blocks for semantic mapping.