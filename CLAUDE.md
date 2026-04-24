# OVO (Open-Vocabulary Online Semantic Mapping)

OVO is a research project for 3D instance-aware semantic mapping. It integrates SLAM backbones with semantic recognition using open-vocabulary models to build 3D maps where objects are recognized as distinct instances. Original work was using CLIP for the semantic part. We'll be testing new things. The current focus is on improving semantic fusion during map optimizations and loop closures.

## Project Structure
```text
/
├── ovo/                    # Core OVO system
│   ├── entities/           # Instance management, generators, orchestrator, semantic classes
│   │   ├── visualizers/    # Rerun and Open3D visualization handlers
│   │   ├── clip_generator.py, pe_generator.py, sam3_generator.py, dino_generator.py
│   │   ├── fusion.py, clips_merging.py, fusion_encoders.py
│   │   ├── instance3d.py, logger.py, mask_generator.py
│   │   ├── ovo.py, ovomapping.py, visualizer.py
│   │   └── datasets.py     # Semantic class definitions and label mapping
│   ├── slam/               # SLAM backbone wrappers
│   │   ├── gaussian_slam.py, orbslam2.py, groundtruth_slam.py, vanilla_mapper.py
│   │   └── sem_gaussian_model.py
│   ├── submodules/         # Internal submodules (e.g., gaussian_slam core)
│   └── utils/              # Geometry, instance, and eval utility functions
│       ├── geometry_utils.py, instance_utils.py, io_utils.py, vis_utils.py
│       ├── segment_utils.py, clip_utils.py, eval_utils.py, gen_utils.py
│       └── ins_eval_utils.py, results_utils.py, scannet200_ins.py, replica_ins.py
├── scripts/                # Experiment orchestration and preprocessing
├── .agents_mapper/         # Task tracking and detailed documentation
├── data/                   # Configs, input datasets, and experiment outputs
├── thirdParty/             # External modules (ORB-SLAM3, SAM 2, etc.)
├── tests/                  # Unit and integration tests
└── run_eval.py             # Main entry point for experiments
```

## Architecture & Workflow
*   **`ovomapping.py`**: The central orchestrator. It manages the data flow between the SLAM backbone and the semantic modules, handles initial configuration, and triggers semantic updates.
*   **`ovo.py`**: Manages the semantic workflow, including instance creation, feature extraction, and the fusion/merging process.
*   **SLAM Integration**: All SLAM-specific logic is contained within `ovo/slam/`.
*   **Semantic Classes**: Management of dataset-specific semantic classes, label mappings, and category definitions is handled within `ovo/entities/` (specifically in `datasets.py`).
*   **Semantic Fusion**: The core logic for merging instances and handling descriptors (CLIP/DINO) resides in `ovo/entities/fusion.py` and `ovo/utils/instance_utils.py`.

## Arquitectura

PlantUML diagrams in `docs/uml/`. Render with `plantuml *.puml` or [plantuml.com](https://www.plantuml.com/plantuml).

| Diagram | Covers | Consult when… |
|---|---|---|
| [`docs/uml/01_class-orchestration.puml`](docs/uml/01_class-orchestration.puml) | `OVOSemMap`, `Logger`, `RunConfig`, `VisConfig`, `SchedulingConfig`, `FrameState`, `VisualizationManager` (stub) | Understanding main loop, frame scheduling, config wiring, logging |
| [`docs/uml/02_class-semantic-core.puml`](docs/uml/02_class-semantic-core.puml) | `OVO`, `Instance3D` | Instance lifecycle (segment → track → fuse → classify), descriptor fields, semantic pipeline methods |
| [`docs/uml/03_class-slam.puml`](docs/uml/03_class-slam.puml) | `VanillaMapper`, `GroundTruthSLAM`, `WrapperORBSLAM2`, `WrapperGaussianSLAM`, `SemGaussianModel` | Adding/changing SLAM backend, understanding map/track interface, point cloud ownership |
| [`docs/uml/04_class-datasets.puml`](docs/uml/04_class-datasets.puml) | `BaseDataset`, `Replica`, `ScanNet`, `ScanNetPP`, `Matterport` | Adding new dataset, understanding frame iteration interface, intrinsics/pose fields |
| [`docs/uml/05_class-fusion.puml`](docs/uml/05_class-fusion.puml) | `FusionStrategy`, `SemanticGeometricFusion`, `GeometricOnlyFusion`, `FusionEncoderAdapter`, `PEFusionAdapter`, `SAM3FusionAdapter`, all feature generators (`CLIP`, `PE`, `SAM3`, `Mask`) | Adding fusion strategy or encoder, understanding descriptor pipeline, merge decision logic |
| [`docs/uml/06_class-visualization.puml`](docs/uml/06_class-visualization.puml) | `VisualizationManager`, `QueueRendererOrchestrator`, `BaseRerunRenderer`, `StreamRenderer`, IPC contracts (`StreamFrameMessage`, `UpdateMapMessage`) | Adding visualization, understanding multiprocess IPC, Rerun renderer structure |

## Rules for Working with Code

### Diagrams first, code second

For any discussion — features, architecture questions, how something works, design decisions — **start with the UML diagrams above**. They are the source of truth for structure and ownership.

- If the answer is not in the diagrams: either the diagram needs updating, or the topic is too implementation-specific.
- Only look at code when: a design is agreed and needs implementation details, debugging a specific function, or writing/modifying actual code.

### Always update UML after code changes

**MANDATORY: any code file change → update its corresponding UML diagram in the same task/commit.**

Use the table above to find which diagram covers the modified file. Update signatures, attributes, and relationships to match the new code exactly. No exceptions.

### Diagrams own structure. Code owns implementation.

**Need methods, attributes, or signatures of a class? Read the correct UML diagram — not the code.** If it's not in the diagram, the diagram needs updating.

Only open a source file when you need the **implementation body** of a specific function (debugging or writing code). Even then:

```bash
# Step 1: get line number from grep
grep -n "def function_name" path/to/file.py

# Step 2: read only that function's body
# Read(file, offset=<line>, limit=<estimated_size>)
```

- `__init__` is intentionally omitted from diagrams but is valid to grep+read when you need to understand how a class instance is constructed.
- If function body is cut short, re-read with a larger limit. Never assume incomplete output is enough.
- Never load a full file unless the user explicitly instructs it.
- Never grep for method lists or class structure — that's what diagrams are for.

## Task Management Workflow
Work is organized by tasks tracked in `.agents_mapper/tasks/`.
*   **Index**: `_index.json` maintains the status (active, completed, pending) of all tasks.
*   **Active Tasks**: Detailed logs and documentation for currently `IN_PROGRESS` work are found in `active_tasks/`.
*   **Archive**: Completed or cancelled tasks are moved to `completed_tasks/` for historical reference.
