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

## Task Management Workflow
Work is organized by tasks tracked in `.agents_mapper/tasks/`.
*   **Index**: `_index.json` maintains the status (active, completed, pending) of all tasks.
*   **Active Tasks**: Detailed logs and documentation for currently `IN_PROGRESS` work are found in `active_tasks/`.
*   **Archive**: Completed or cancelled tasks are moved to `completed_tasks/` for historical reference.

## Worktree Setup (Required After `git worktree add`)

New worktrees ship with empty submodule stubs and no heavy data. Before running experiments in a worktree, symlink `thirdParty/` and `data/` subdirs to the primary repo at `/home/padidavid/repos/OVO/`. This reuses built submodules, model weights, datasets, and centralizes experiment outputs.

```bash
PRIMARY=/home/padidavid/repos/OVO
WT=<absolute-path-to-new-worktree>

# thirdParty: replace empty submodule stubs + add ORB_SLAM3
for sub in perception_models sam3 segment-anything-2; do
  [ -d "$WT/thirdParty/$sub" ] && [ ! -L "$WT/thirdParty/$sub" ] && rmdir "$WT/thirdParty/$sub"
  ln -sfn "$PRIMARY/thirdParty/$sub" "$WT/thirdParty/$sub"
done
ln -sfn "$PRIMARY/thirdParty/ORB_SLAM3" "$WT/thirdParty/ORB_SLAM3"

# data: heavy + write paths. If targets pre-exist as real dirs in the worktree,
# `rm -rf` them first (ln -sfn nests inside otherwise).
mkdir -p "$WT/data/input" "$WT/data/working"
ln -sfn "$PRIMARY/data/baselines"              "$WT/data/baselines"
ln -sfn "$PRIMARY/data/output"                 "$WT/data/output"
ln -sfn "$PRIMARY/data/input/Datasets"         "$WT/data/input/Datasets"
ln -sfn "$PRIMARY/data/input/sam_ckpts"        "$WT/data/input/sam_ckpts"
ln -sfn "$PRIMARY/data/working/config_preview" "$WT/data/working/config_preview"
```

Notes:
*   Centralized `data/output/` → every branch writes here; folder names (`{DATE}_{SLAM_CONFIG}_{FUSION}_{LABEL}`) must encode branch in `LABEL` to avoid clobbering.
*   Removing a worktree (`git worktree remove` or `rm -rf <wt>`) does **not** follow symlinks → primary stays intact.
*   Tracked subdirs (`data/input/{ReadMe.md,replica_semantic_gt,weights_predictor}`, `data/working/configs`) stay local — per-subdir symlinks preserve them.
