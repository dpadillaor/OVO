# OVO Project Codebase Overview

This section provides a high-level architectural overview of the OVO project, detailing the interaction between its main components, and explaining the purpose of its main directories and modules.

## Main Components:
*   **SLAM Backbone:** Handles camera pose estimation and 3D point cloud generation.
*   **MaskGenerator (SAM):** Segments 2D images to find potential objects.
*   **Instance3D:** Represents detected 3D object instances.
*   **CLIPGenerator:** Extracts appearance features for instances.
*   **OVO (Semantic Core):** Manages instance creation, feature extraction, and merging.
*   **OVOSemMap (Orchestrator):** Integrates SLAM and semantic processing.

## Structure:
The `documentation_structure/` subdirectory mirrors the relevant parts of the main project codebase, with `_dir_summary.md` files providing context for each directory.

## Key Documentation Files:
*   `design_principles.md`: Documentation of recurring design patterns or important conventions.
*   `core_concepts.md`: Explanations of key concepts in OVO.
*   `configuration_reference.md`: Guide to project configuration files.
