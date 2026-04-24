# Logging and Runtime Monitoring

This document describes how OVO logs runtime performance, timing, and system resources. These logs are used for debugging, profiling, and monitoring experiments in real-time.

---

## Logging Architecture

The logging system is managed by the `Logger` class (`ovo/entities/logger.py`). It handles three main sinks:
1.  **Local Files**: Structured `.log` and `.csv` files in the experiment directory.
2.  **Weights & Biases (W&B)**: Optional cloud tracking for real-time visualization.
3.  **Console**: High-signal progress updates and final summaries.

---

## Directory Structure (Logger)

Logs are stored within each scene's output directory:
```text
data/output/{Dataset}/{Experiment}/{Scene}/
├── logger/                     # Runtime performance logs
│   ├── avg_fps.log             # Final average FPS for the run
│   ├── spf.log                 # Seconds Per Frame (semantic + mapping)
│   ├── ram.log                 # System RAM usage (GB) per tracked frame
│   ├── vram.log                # GPU VRAM usage (GB) per tracked frame
│   ├── t_sam.log               # Time spent in SAM segmentation
│   ├── t_clip.log              # Time spent in CLIP feature extraction
│   ├── t_seg.log               # Time spent in semantic segmentation
│   ├── t_obj.log               # Time spent in object processing
│   └── ...                     # Other component-specific timings
└── fusion_decisions.csv        # Detailed instance fusion logs (see data structure doc)
```

---

## Timing Metrics

### Pipeline Latency
Timing is measured using `time.time()` and synchronized with `torch.cuda.synchronize()` to ensure accurate GPU measurement.

| Metric | Start Location | End Location | Stored? | Storage Method |
| :--- | :--- | :--- | :--- | :--- |
| **Total Scene Time** | `ovomapping.py` (L388) | `ovomapping.py` (L412) | **Yes** | `logger/total_time.log` |
| **SPF** (Frame) | `ovomapping.py` (L388) | `ovomapping.py` (L412) | **Yes** | `logger/spf.log` |
| **FPS** (Total) | `ovomapping.py` (L388) | `ovomapping.py` (L417) | **Yes** | `logger/avg_fps.log` |
| **Mapping (t_lc)** | `ovomapping.py` (L337) | `ovomapping.py` (L380) | **Yes** | `logger/t_up.log`* |
| **SAM (t_sam)** | `ovo.py` (`@profil`) | `ovo.py` (L237) | **Yes** | `logger/t_sam.log` |
| **CLIP (t_clip)** | `ovo.py` (`@profil`) | `ovo.py` (L443) | **Yes** | `logger/t_clip.log` |
| **Object (t_obj)**| `ovo.py` (`@profil`) | `ovo.py` (L238) | **Yes** | `logger/t_obj.log` |

*\*Note: `t_lc` is measured as a local variable but its results are typically aggregated into `t_up` or `spf` during logging.*

## Storage Details
*   **Per-Component Logs**: Every time a metric is logged via `Logger.log_ovo_stats` or `Logger.log_spf`, it is added to an internal list.
*   **Final Flush**: When `Logger.write_stats()` is called at the end of a run, each list is joined into a newline-separated string and saved as a `.log` file in the `logger/` directory.
*   **W&B**: If enabled, these metrics are also sent to the cloud as a time-series.

---

## Resource Monitoring

The system monitors hardware usage every `segment_every` frames to detect memory leaks or bottlenecks.

*   **RAM**: Captured using `psutil`. Measures the Resident Set Size (RSS) of the main process in GB.
*   **VRAM**: Captured using `torch.cuda.memory_allocated()`. Measures active GPU memory in GB.
*   **Peak Usage**: At the end of a run, the system calculates `max_ram` and `max_vram` using `torch.cuda.max_memory_allocated()`.

---

## Weights & Biases (W&B) Integration

If enabled (`use_wandb: True`), the following metrics are streamed:
*   **Prefix**: `Semantic/`
*   **Metrics**: `vram`, `ram`, `avg_fps`, `n_obj`, and all component timings (`t_sam`, `t_clip`, etc.).
*   **Grouping**: Runs are grouped by `scene_name` for easy comparison across experiments.

---

## Final Summary
At the end of every scene run, a summary is printed to the console:
```text
Final statistics:
{'Avg avg_fps': 12.4,
 'Avg ram': 4.2,
 'Avg t_clip': 0.045,
 'Avg vram': 2.1,
 'Max RAM': 4.8,
 'Max vRAM': 2.5,
 ...}
```
*Note: Averages are calculated as the `nanmean` of all values captured during the run.*
