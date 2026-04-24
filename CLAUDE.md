# OVO (Open-Vocabulary Online Semantic Mapping)

OVO is a research project for 3D instance-aware semantic mapping. It integrates SLAM backbones with semantic recognition using open-vocabulary models to build 3D maps where objects are recognized as distinct instances. Original work was using CLIP for the semantic part. We'll be testing new things. The current focus is on improving semantic fusion during map optimizations and loop closures.

## Project Structure
```text
/
├── ovo/                    # Core OVO system
│   ├── entities/           # Instance management, generators, orchestrator, semantic classes
│   │   ├── ovomapping.py   # Main orchestrator (SLAM + Semantics)
│   │   ├── ovo.py          # Semantic workflow & instance handling
│   │   ├── fusion.py       # Fusion and merging logic
│   │   └── datasets.py     # Semantic class definitions and label mapping
│   ├── slam/               # SLAM backbone wrappers (ORB-SLAM, GT, etc.)
│   └── utils/              # Geometry and instance utility functions
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

| Diagram | Consult when… |
|---|---|
| [`docs/uml/class-diagram.puml`](docs/uml/class-diagram.puml) | Adding/modifying a class, tracing ownership or inheritance (SLAM backends, datasets, fusion strategies, feature generators) |
| [`docs/uml/component-overview.puml`](docs/uml/component-overview.puml) | Onboarding, planning cross-cutting changes, understanding module boundaries and data flow |
| [`docs/uml/sequence-main-loop.puml`](docs/uml/sequence-main-loop.puml) | Debugging frame-processing order, tracing when segmentation/fusion/CLIP steps fire, understanding the `run()` loop |

See [`docs/uml/README.md`](docs/uml/README.md) for diagram-to-source-file mapping.

## Task Management Workflow
Work is organized by tasks tracked in `.agents_mapper/tasks/`.
*   **Index**: `_index.json` maintains the status (active, completed, pending) of all tasks.
*   **Active Tasks**: Detailed logs and documentation for currently `IN_PROGRESS` work are found in `active_tasks/`.
*   **Archive**: Completed or cancelled tasks are moved to `completed_tasks/` for historical reference.
