# config["data"] + config["cam"] usage — Datasets

Datasets receive merged dict: `{**config["data"], **config["cam"]}` at instantiation.
All subclasses (Replica, ScanNet, ScanNetPP, Matterport) call `super().__init__()` — inherit BaseDataset accesses.

## Keys from config["cam"]
| Key | file:line | Usage |
|---|---|---|
| `H` | datasets.py:21 | `self.height` (scaled by resize_ratio) |
| `W` | datasets.py:22 | `self.width` (scaled by resize_ratio) |
| `fx` | datasets.py:23 | `self.fx` (scaled) |
| `fy` | datasets.py:24 | `self.fy` (scaled) |
| `cx` | datasets.py:25 | `self.cx` (scaled) |
| `cy` | datasets.py:26 | `self.cy` (scaled) |
| `depth_scale` | datasets.py:28 | `self.depth_scale` |
| `distortion` | datasets.py:30 | `self.distortion` (optional, checked with `in`) |
| `crop_edge` | datasets.py:31 | `self.crop_edge` (optional, default 0) |
| `depth_th` | datasets.py:89 | `self.depth_th` in ScanNet (optional, default 0) |
| `H`, `W` | datasets.py:114 | re-accessed via `self.dataset_config` in ScanNet.__getitem__ for resize |

## Keys from config["data"]
| Key | file:line | Usage |
|---|---|---|
| `input_path` | datasets.py:17 | `self.dataset_path` (Path object) |
| `frame_limit` | datasets.py:18 | `self.frame_limit` (default -1) |
| `resize_ratio` | datasets.py:20 | scales intrinsics (default 1.0) |
| `use_train_split` | datasets.py:132 | `self.use_train_split` in ScanNetPP (required) |

## Derived attributes (computed from above)
- `self.fovx`, `self.fovy` — from fx/fy/W/H
- `self.intrinsics` — 3×3 matrix from fx/fy/cx/cy
- Used downstream by SLAM and semantic modules

## Subclass-specific additions
| Subclass | Extra keys |
|---|---|
| Replica | none |
| ScanNet | `depth_th` |
| ScanNetPP | `use_train_split` (required) |
| Matterport | none |
