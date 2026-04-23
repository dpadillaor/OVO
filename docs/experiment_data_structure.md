# Experiment Data Structure

Reference document for how OVO experiment results are stored on disk and how the GUI/analysis code loads them.
See `ovo/utils/results_utils.py` for all parsing logic and `gui/app.py` for the Streamlit frontend.

---

## Directory Layout

```text
data/output/
└── {Dataset}/                          # e.g. "Replica", "ScanNet"
    └── {EXPERIMENT_FOLDER}/            # one folder per experiment run
        ├── replica/
        │   ├── statistics.txt          # aggregate semantic metrics (all scenes combined)
        │   ├── statistics_{scene}.txt  # per-scene semantic metrics (one file per scene)
        │   ├── instance_ap.txt               # aggregate instance AP (all scenes combined)
        │   ├── instance_ap_{scene}.txt       # per-scene instance AP
        │   └── confmat.png             # confusion matrix image
        ├── {scene}/                    # one folder per scene (written during run)
        │   └── fusion_decisions.csv    # per-frame fusion pair decisions for that scene
        └── instance_pred/
            ├── {scene}.txt             # predicted instance list for that scene
            └── predicted_masks/        # (optional) per-instance mask JSON files
                ├── {scene}_000.json
                └── ...
```

The dataset name is inferred from the **parent directory** of the experiment folder, not from the folder name itself.

---

## Experiment Folder Naming Convention

```text
{DATE}_{SLAM_CONFIG}_{FUSION}_{LABEL}
```

| Part | Format | Example | Notes |
| --- | --- | --- | --- |
| `DATE` | `YYYYMMDD` | `20260305` | 8-digit date |
| `SLAM_CONFIG` | hyphen-separated token | `GT`, `GTNoise-T0p01-R0p01`, `GTJump-J1-T0p05-R2p0`, `ORBSLAM3` | Uses only hyphens internally |
| `FUSION` | string | `GT`, `Vanilla`, `SAM3`, `CLIP`, `PE-Core` | Fusion method used |
| `LABEL` | free string (underscores OK) | `ComparativaPaper` | Joins all remaining `_`-split parts |

Derived fields computed at parse time:

- `Method` = `{FUSION}_{LABEL}` (used as the primary grouping key in all plots)
- `Trans_Noise` / `Rot_Noise` = extracted from `T{int}p{dec}` / `R{int}p{dec}` tokens in `SLAM_CONFIG`; zero if no noise tokens are found (e.g. `GT`, `ORBSLAM3`)
- `Jump_Count` = extracted from `J{int}` token in `SLAM_CONFIG` (e.g. `GTJump-J1-T0p05-R2p0` → `1`); zero if no `J` token found

**Example:**

```text
20260305_GT_SAM3_ComparativaPaper
  → Date=20260305, SLAM_Config=GT, Fusion=SAM3,
    Label=ComparativaPaper, Method=SAM3_ComparativaPaper,
    Trans_Noise=0.0, Rot_Noise=0.0
```

Folders that do not match the convention (fewer than 4 `_`-split parts or no valid date) are silently skipped.

---

## File Formats

### `replica/statistics.txt` — Aggregate metrics

Aggregate metrics over **all scenes** combined. One file per experiment.

```text
label, acc, iou,
wall, 0.8508876332635238, 0.7473786731230466,
ceiling, 0.8733531348397222, 0.7493387487411673,
...
switch, 0.3203125, 0.22631094756209752,
```

- Line 1: header (skipped by parser)
- Each subsequent line: `class_name, acc, iou,` (trailing comma is stripped)
- Values are **decimals in [0, 1]**, not percentages
- `mIoU` and `mAcc` are computed as `nanmean` over all class rows

### `replica/statistics_{scene}.txt` — Per-scene metrics

Identical format to `statistics.txt` but scoped to a single scene. The scene name is extracted from the filename stem: `statistics_{scene}.txt` → scene = `{scene}` (e.g. `office0`, `room1`).

### `replica/instance_ap.txt` — Instance AP metrics

Aggregate instance segmentation AP over all scenes. Written by `eval_instance_ap` (`--eval_instances` stage).

```text
metric, value
AP, 0.012
AP_50, 0.021
AP_25, 0.038
AP_agnostic, 0.106
AP_agnostic_50, 0.188
AP_agnostic_25, 0.286
```

- `AP` / `AP_50` / `AP_25`: class-aware AP at IoU thresholds [0.5:0.95], 0.5, 0.25
- `AP_agnostic` / `AP_agnostic_50` / `AP_agnostic_25`: class-agnostic AP (ignores predicted class label)

### `instance_pred/{scene}.txt` — Predicted instances

One line per predicted instance. Line count = `Num_Instances`.

```text
./predicted_masks/office0_000.json 6 0.9999
./predicted_masks/office0_001.json 3 0.9998
./predicted_masks/office0_002.json 6 0.9995
```

Fields per line: `mask_path  class_id  confidence`

The parser only counts non-empty lines; it does not parse the fields.

### `{scene}/fusion_decisions.csv` — Per-scene fusion decisions

One row per fusion candidate pair evaluated during a scene run. Written by the fusion module; one file per scene, located in `{EXPERIMENT}/{scene}/fusion_decisions.csv` (not under `replica/`).

```text
frame_id,result,i1,i2,reason,centroid_dist,cos_sim,p_dist
42,ACCEPTED,,84,130,,0.31,0.87,0.12
42,REJECTED,,92,105,centroid,1.83,0.72,0.34
```

| Column | Type | Description |
| --- | --- | --- |
| `frame_id` | `int` | Frame index when the pair was evaluated (`update_map` call) |
| `result` | `str` | `ACCEPTED` (instances fused) or `REJECTED` |
| `i1` | `int` | First instance ID in the candidate pair |
| `i2` | `int` | Second instance ID in the candidate pair |
| `reason` | `str` | Rejection reason: `centroid`, `cos_sim`, or `overlap`; empty for accepted |
| `centroid_dist` | `float` | Euclidean distance between instance centroids |
| `cos_sim` | `float` | Cosine similarity between feature descriptors |
| `p_dist` | `float` | Probability distance (fusion gate metric) |

Aggregate statistics derived from this file (used in GUI):

- `Fusion_Total` — total candidate pairs evaluated
- `Fusion_Accepted` — count of `ACCEPTED` rows
- `Fusion_Accept_Rate` — `Fusion_Accepted / Fusion_Total`
- `Fusion_Reject_Centroid` / `Fusion_Reject_CosSim` / `Fusion_Reject_Overlap` — per-reason rejection counts

See `scripts/analyze_fusion_decisions.py` for a CLI analysis tool.

### `replica/confmat.png` — Confusion matrix

Pre-rendered PNG image. Loaded as-is and displayed in the GUI's Conf. Matrix tab.

---

## DataFrames Produced by the Loader

### `load_experiments(output_dir)` → `(df_exp, df_class)`

**`df_exp`** — one row per experiment:

| Column | Type | Source |
| --- | --- | --- |
| `Date` | `datetime` (parsed from `YYYYMMDD`) | folder name |
| `Dataset` | `str` | parent directory name |
| `SLAM_Config` | `str` | folder name |
| `Fusion` | `str` | folder name |
| `Label` | `str` | folder name |
| `Method` | `str` | `{Fusion}_{Label}` |
| `Trans_Noise` | `float` | parsed from SLAM_Config |
| `Rot_Noise` | `float` | parsed from SLAM_Config |
| `Jump_Count` | `int` | parsed from `J{int}` token in SLAM_Config; 0 if absent |
| `mIoU` | `float` | nanmean over all classes in statistics.txt |
| `mAcc` | `float` | nanmean over all classes in statistics.txt |
| `Head_mIoU` | `float` | nanmean over first third of classes (by dataset order) |
| `Head_mAcc` | `float` | nanmean over first third of classes |
| `Common_mIoU` | `float` | nanmean over middle third of classes |
| `Common_mAcc` | `float` | nanmean over middle third of classes |
| `Tail_mIoU` | `float` | nanmean over last third of classes |
| `Tail_mAcc` | `float` | nanmean over last third of classes |
| `Num_Instances` | `int` | line count in instance_pred/{folder_name}.txt |
| `AP` | `float` | class-aware AP at IoU [0.5:0.95] from instance_ap.txt; NaN if absent |
| `AP_50` | `float` | class-aware AP at IoU 0.5 |
| `AP_25` | `float` | class-aware AP at IoU 0.25 |
| `AP_agnostic` | `float` | class-agnostic AP at IoU [0.5:0.95] |
| `AP_agnostic_50` | `float` | class-agnostic AP at IoU 0.5 |
| `AP_agnostic_25` | `float` | class-agnostic AP at IoU 0.25 |
| `Fusion_Total` | `int` | total fusion candidate pairs evaluated (sum across scenes) |
| `Fusion_Accepted` | `int` | count of ACCEPTED pairs |
| `Fusion_Accept_Rate` | `float` | Fusion_Accepted / Fusion_Total; NaN if no data |
| `Fusion_Reject_Centroid` | `int` | pairs rejected due to centroid distance |
| `Fusion_Reject_CosSim` | `int` | pairs rejected due to cosine similarity |
| `Fusion_Reject_Overlap` | `int` | pairs rejected due to overlap gate |
| `Experiment_ID` | `str` | folder name (raw) |

The Head/Common/Tail split mirrors the computation in `eval_utils.eval_semantics`: classes are divided into three equal thirds (`len(classes)//3`) in the order they appear in `statistics.txt`, which matches the dataset label ordering (most to least frequent). The same split is applied to per-scene DataFrames.

**`df_class`** — one row per (experiment × class):

All `df_exp` columns plus:

| Column | Type | Source |
| --- | --- | --- |
| `Class` | `str` | class label in statistics.txt |
| `IoU` | `float` | per-class IoU |
| `Acc` | `float` | per-class Acc |

### `load_scene_results(output_dir)` → `(df_scene, df_scene_class)`

Same structure as above but scoped to individual scenes. Each `statistics_{scene}.txt` generates one row.

**`df_scene`** — one row per (experiment × scene): all `df_exp` columns + `Scene` (str). Head/Common/Tail values are computed per-scene from `statistics_{scene}.txt`. AP columns come from `instance_ap_{scene}.txt`. Fusion columns come from `{scene}/fusion_decisions.csv`.

**`df_scene_class`** — one row per (experiment × scene × class): all `df_scene` columns + `Class`, `IoU`, `Acc`.

`Num_Instances` in scene DataFrames comes from `instance_pred/{scene}.txt` (not the experiment folder name).

---

## Static Reference File

Paper baselines are stored separately and are **not** scanned from `data/output/`:

```text
references/paper_results.csv
```

Same column schema as `df_exp`. Head/Common/Tail columns are populated directly from the paper's reported values (pre-computed).

Loaded by `load_reference_results(csv_path)` in `results_utils.py`. The GUI merges these rows into the main DataFrame when the "Show paper baselines" checkbox is active. All metric values use the same decimal format `[0, 1]` as live experiments.

---

## Canonical File-Path Summary

| What | Path |
| --- | --- |
| Aggregate stats | `data/output/{Dataset}/{EXPERIMENT}/replica/statistics.txt` |
| Per-scene stats | `data/output/{Dataset}/{EXPERIMENT}/replica/statistics_{scene}.txt` |
| Instance AP (aggregate) | `data/output/{Dataset}/{EXPERIMENT}/replica/instance_ap.txt` |
| Instance AP (per-scene) | `data/output/{Dataset}/{EXPERIMENT}/replica/instance_ap_{scene}.txt` |
| Fusion decisions (per-scene) | `data/output/{Dataset}/{EXPERIMENT}/{scene}/fusion_decisions.csv` |
| Instance predictions | `data/output/{Dataset}/{EXPERIMENT}/instance_pred/{scene}.txt` |
| Confusion matrix | `data/output/{Dataset}/{EXPERIMENT}/replica/confmat.png` |
| Paper baselines | `references/paper_results.csv` |
