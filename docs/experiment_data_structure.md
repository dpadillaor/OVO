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
        │   ├── statistics.txt          # aggregate metrics (all scenes combined)
        │   ├── statistics_{scene}.txt  # per-scene metrics (one file per scene)
        │   └── confmat.png             # confusion matrix image
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
| `SLAM_CONFIG` | hyphen-separated token | `GT`, `GTNoise-T0p01-R0p01`, `ORBSLAM3` | Uses only hyphens internally |
| `FUSION` | string | `GT`, `Vanilla`, `SAM3`, `CLIP`, `PE-Core` | Fusion method used |
| `LABEL` | free string (underscores OK) | `ComparativaPaper` | Joins all remaining `_`-split parts |

Derived fields computed at parse time:

- `Method` = `{FUSION}_{LABEL}` (used as the primary grouping key in all plots)
- `Trans_Noise` / `Rot_Noise` = extracted from `T{int}p{dec}` / `R{int}p{dec}` tokens in `SLAM_CONFIG`; zero if no noise tokens are found (e.g. `GT`, `ORBSLAM3`)

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

### `instance_pred/{scene}.txt` — Predicted instances

One line per predicted instance. Line count = `Num_Instances`.

```text
./predicted_masks/office0_000.json 6 0.9999
./predicted_masks/office0_001.json 3 0.9998
./predicted_masks/office0_002.json 6 0.9995
```

Fields per line: `mask_path  class_id  confidence`

The parser only counts non-empty lines; it does not parse the fields.

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
| `mIoU` | `float` | nanmean over classes in statistics.txt |
| `mAcc` | `float` | nanmean over classes in statistics.txt |
| `Num_Instances` | `int` | line count in instance_pred/{folder_name}.txt |
| `Experiment_ID` | `str` | folder name (raw) |

**`df_class`** — one row per (experiment × class):

All `df_exp` columns plus:

| Column | Type | Source |
| --- | --- | --- |
| `Class` | `str` | class label in statistics.txt |
| `IoU` | `float` | per-class IoU |
| `Acc` | `float` | per-class Acc |

### `load_scene_results(output_dir)` → `(df_scene, df_scene_class)`

Same structure as above but scoped to individual scenes. Each `statistics_{scene}.txt` generates one row.

**`df_scene`** — one row per (experiment × scene): all `df_exp` columns + `Scene` (str).

**`df_scene_class`** — one row per (experiment × scene × class): all `df_scene` columns + `Class`, `IoU`, `Acc`.

`Num_Instances` in scene DataFrames comes from `instance_pred/{scene}.txt` (not the experiment folder name).

---

## Static Reference File

Paper baselines are stored separately and are **not** scanned from `data/output/`:

```text
references/paper_results.csv
```

Same column schema as `df_exp`, plus extra columns for frequency-tertile breakdowns:
`Head_mIoU`, `Head_mAcc`, `Common_mIoU`, `Common_mAcc`, `Tail_mIoU`, `Tail_mAcc`.

Loaded by `load_reference_results(csv_path)` in `results_utils.py`. The GUI merges these rows into the main DataFrame when the "Show paper baselines" checkbox is active. All metric values use the same decimal format `[0, 1]` as live experiments.

---

## Canonical File-Path Summary

| What | Path |
| --- | --- |
| Aggregate stats | `data/output/{Dataset}/{EXPERIMENT}/replica/statistics.txt` |
| Per-scene stats | `data/output/{Dataset}/{EXPERIMENT}/replica/statistics_{scene}.txt` |
| Instance predictions | `data/output/{Dataset}/{EXPERIMENT}/instance_pred/{scene}.txt` |
| Confusion matrix | `data/output/{Dataset}/{EXPERIMENT}/replica/confmat.png` |
| Paper baselines | `references/paper_results.csv` |
