# GEMINI Project Context: OVO (Open-Vocabulary Online Semantic Mapping)

## Project Overview

OVO is a Python-based research project for 3D instance-aware semantic mapping. It integrates a SLAM backbone (like ORB-SLAM2/3 or Gaussian-SLAM) with deep learning models for segmentation (SAM/SAM2) and feature extraction (CLIP) to build a 3D map of a scene where individual objects are recognized as distinct instances.

The core pipeline is as follows:
1.  A SLAM backbone processes RGB-D video frames to estimate camera poses and build a 3D point cloud map.
2.  For given keyframes, a `MaskGenerator` (SAM) segments the image to find potential objects.
3.  These 2D masks are associated with the 3D point cloud to create `Instance3D` objects.
4.  A `CLIPGenerator` extracts appearance features for these instances.
5.  During a loop closure (signaled by the SLAM backbone), the system attempts to merge duplicate instances by comparing their centroid distance, point cloud overlap, and the cosine similarity of their CLIP features.

The project is configured via YAML files located in `data/working/configs/`, with `ovo.yaml` being the base configuration.

## Building and Running

The project is managed using a Conda environment. There is no single `requirements.txt` file; dependencies are listed in the `ReadMe.md`.

### Setup

Follow the installation instructions in `ReadMe.md` to create the `ovo` conda environment and install dependencies, including the external `segment-anything-2` and optional SLAM backbones.

### Running an Experiment

The main entry point is `run_eval.py`. Experiments are run on specific scenes from datasets like `Replica` or `ScanNet`.

**Example Command:**
```bash
# To run OVO on the Replica office0 scene, then segment the ground truth, and evaluate
python run_eval.py --dataset_name Replica --experiment_name ovo_mapping --run --segment --eval --scenes office0
```

**Key Arguments for `run_eval.py`:**
*   `--dataset_name`: The dataset to use (e.g., `Replica`, `ScanNet`).
*   `--scenes`: A list of scenes to process.
*   `--experiment_name`: The name of the output folder for results.
*   `--run`: Executes the main OVO mapping pipeline.
*   `--segment`: Segments the ground truth point cloud using the generated map.
*   `--eval`: Computes the final evaluation metrics.

## Operating Modes

Our collaboration will follow three distinct modes:

1.  **Dialogue & Decision-Making Mode:** In this mode, we will discuss requirements, define strategies, and structure the work. The primary output of this mode is updated documentation (e.g., `DECISION_LOG.md`, feature logs) that reflects the decisions made.

2.  **Co-Development Mode:** This mode is for writing and modifying code. The workflow is strictly sequential and requires explicit user approval for *every* action.
    *   I will propose a single, specific action (e.g., reading a file, replacing a block of code).
    *   I will wait for your explicit permission before executing the action.
    *   After executing the action, I will await your next instruction or propose the next single action.
    *   I will not chain or sequence multiple actions without approval for each one.
    *   At the end of each co-development session, I will update the relevant documentation files to log the work that was completed.

3.  **Professor/Reviewer Mode:** In this mode, you will take the role of the programmer and I will act as a critical code reviewer.
    *   You will write code based on our previously agreed-upon plans.
    *   You will then ask me to review the code you have written.
    *   My review will be critical and will cover:
        *   **Good Points:** What is well-done and why.
        *   **Bad Points:** What can be improved or might cause issues.
        *   **Suggestions for Improvement:** Code alternatives, more efficient or idiomatic structures, and best practice recommendations.

## General Operating Rules

The following rules are always in place, across all modes:

0.  **Initial Interaction:** At the absolute start of any new session, my first action must be to ask you what you want to do and which operating mode we should use. I will not perform any other action until you provide this initial direction. After you state your goal, I will then proceed with the session initialization.
1.  **Session Initialization:** Once the initial goal and mode are set, at the beginning of every new session, I must:
    1.  Read `GEMINI.md` and `documentacion/DECISION_LOG.md`.
    2.  From `documentacion/DECISION_LOG.md`, identify the feature that is `in_progress`.
    3.  Read the corresponding detailed feature log file (e.g., `documentacion/feature_logs/XX_feature_name.md`) to get the full, up-to-date context on the current task.
2.  **Execution Permission:** I must ask for explicit permission before executing any Python script (`.py`).
3.  **File Modification:** I have permission to modify files in the `documentacion/` directory and other files as explicitly approved for the current task. The initial, restrictive rule regarding `scripts/extract_slam_baseline.py` is considered outdated.
4.  **Feature Completion Protocol:** I am not allowed to unilaterally mark a feature as `COMPLETED`. When I believe a feature is finished, I must change its status to `IN_REVIEW` in `documentacion/DECISION_LOG.md` and then notify you. I will wait for your explicit confirmation ("OKEY" or similar) before changing the status to `COMPLETED`.

## Development Conventions

### Language Convention
*   **Code and Comments Language:** All code and comments generated by me (the agent) must be exclusively in English.

### Architecture
*   **`ovo/entities/`**: Contains the core logic and data structures.
    *   `ovomapping.py`: The `OVOSemMap` class orchestrates the entire pipeline (SLAM + Semantics).
    *   `ovo.py`: The `OVO` class manages the semantic side (instance creation, feature extraction, merging).
    *   `instance3d.py`: Defines a 3D instance object.
*   **`ovo/slam/`**: Contains wrappers for different SLAM backbones (e.g., `orbslam2.py`).
*   **`ovo/utils/`**: Contains utility functions.
    *   `instance_utils.py`: Holds the critical `same_instance` function that decides if two instances should be merged.
*   **`run_eval.py`**: The main script for running experiments and evaluations.
*   **`documentacion/`**: Contains project documentation.

### Logging and Documentation

To maintain a clear and organized project history, the following documentation strategy is in place:

*   **High-Level Log (`documentacion/DECISION_LOG.md`):** This file acts as a main index. It tracks the overall project status and links to detailed logs for each major feature.
*   **Detailed Feature Logs:** For each significant new feature (e.g., "Replay Logging"), a dedicated Markdown file will be created inside `documentacion/feature_logs/`. These files will contain all technical details, decisions, and progress for that feature.
*   **Naming Convention:** Feature logs are named with a numerical prefix to maintain chronological order (e.g., `01_replay_logging.md`).

### Current Task: Improving Instance Merging with DINO

The current development goal is to improve the instance merging logic in `ovo/utils/instance_utils.py` by using DINO descriptors instead of (or in addition to) CLIP.

To achieve this in a rigorous and reproducible manner, the following **"Hybrid Replay" methodology** has been decided upon:

1.  **Record:** Run the full ORB-SLAM pipeline on a scene **once**.
2.  **Log:** During that run, save a complete log of the data passed from the SLAM backbone to the OVO system. This log must include, for each frame:
    *   Camera pose.
    *   The 3D map points (point cloud).
    *   The signal indicating a loop closure (`last_big_change_idx`).
3.  **Replay:** Create a new, "dummy" SLAM backbone (e.g., `ReplaySLAM`) that reads this log and provides the exact same data to the OVO system in a deterministic way.
4.  **Develop:** Use this replay environment to safely develop and test the DINO descriptor integration, ensuring that any changes in results are due to the new code, not random variations in the SLAM system.

**The immediate next step is to implement the logging mechanism. Based on the new rules, this will be done in `scripts/extract_slam_baseline.py`.**
