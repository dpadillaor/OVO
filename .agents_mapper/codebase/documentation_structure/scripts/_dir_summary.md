# Directory Summary: Project Scripts

This directory (`scripts/`) contains various utility scripts used for running experiments, preprocessing data, or performing other operational tasks related to the OVO project. These scripts often serve as entry points for specific workflows or automation.

## Key Files:
*   `run_experiments_batch.py`: Script for running multiple OVO experiments in a batch.
*   `extract_slam_baseline.py`: Used to extract SLAM data (poses, point clouds, loop closures) from a full SLAM run to create a baseline log for replay. This is crucial for the "Hybrid Replay" methodology.
*   `scannet_preprocess.py`: Script for preprocessing ScanNet dataset data.
*   `experiments_manifest.yaml`: Configuration file likely used by `run_experiments_batch.py` to define experiment parameters.

## Purpose:
The `scripts` directory facilitates the execution and management of experiments, data preparation, and baseline generation, which are essential for research and development within the OVO project. `extract_slam_baseline.py` is especially relevant for the current DINO integration task.