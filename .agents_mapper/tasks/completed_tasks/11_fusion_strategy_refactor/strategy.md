## Context
The current implementation of object fusion in `ovo/utils/instance_utils.py` (`same_instance` function) is tightly coupled with CLIP embeddings. As we integrate new modalities like PE (Pose Embeddings) and DINO, or geometric-only approaches, the current structure would require complex and unmaintainable `if/else` chains.

## Decision: Strategy Pattern
We have decided to adopt the **Strategy Pattern** to encapsulate the fusion logic. This allows the `OVO` class to be agnostic to the specific algorithm used to compare and fuse instances.

### Architecture Changes

1.  **New Entity: `ovo/entities/fusion.py`**
    *   Will define an Abstract Base Class (ABC) `FusionStrategy`.
    *   **Interface Method:** `same_instance(self, instance1, instance2, data1, data2) -> bool`.
    *   **Concrete Strategies:**
        *   `SemanticGeometricFusion`: Generic class for CLIP, DINO, PE. Configurable via `feature_attr` (e.g., "clip_feature", "dino_feature").
        *   `GeometricOnlyFusion`: For fusion based solely on spatial overlap and distance.

2.  **Refactor: `ovo/utils/instance_utils.py`**
    *   Will be stripped of decision logic.
    *   Will serve as a library of **pure mathematical/geometric functions**:
        *   `compute_centroid_distance(...)`
        *   `compute_pcd_overlap(...)`
        *   `fuse_data_structures(...)` (The logic to merge IDs and lists).

3.  **Integration: `ovo/entities/ovo.py`**
    *   **Initialization:** The specific strategy is selected in `__init__` based on `config["fusion_method"]`.
    *   **Execution:** `update_map` delegates the decision: `if self.fusion_strategy.same_instance(...):`.

## Benefits

*   **Extensibility:** Adding DINO or LLM-based fusion only requires adding a new strategy class or configuring the generic one.
*   **Clean Code:** `OVO` core logic remains clean and focused on the pipeline flow.
*   **Configurability:** Fusion methods can be switched strictly through the YAML configuration.
