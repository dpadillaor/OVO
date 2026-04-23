# OVO Results Dashboard — User Guide

Streamlit dashboard for exploring and comparing OVO experiment results.

```
streamlit run gui/app.py
```

All parsing and plotting logic lives in `ovo/utils/results_utils.py`.
Data structures and file formats are documented in `docs/experiment_data_structure.md`.

---

## Loading Data

1. Enter the **Output Directory** path in the sidebar (default: `data/output`).
2. Click **Load Experiments**.

The loader scans `{output_dir}/{Dataset}/{EXPERIMENT}/` recursively and builds four DataFrames held in `st.session_state`:

| Key | Content |
|---|---|
| `df_exp` | One row per experiment — aggregate metrics |
| `df_class` | One row per (experiment × class) |
| `df_scene` | One row per (experiment × scene) |
| `df_scene_class` | One row per (experiment × scene × class) |

Experiments whose folder names do not match the naming convention `{DATE}_{SLAM_CONFIG}_{FUSION}_{LABEL}` are silently skipped.

---

## Sidebar Filters

Filters apply to all tabs **except** the Compare tab (which has its own experiment picker).

| Filter | Effect |
|---|---|
| **Dates** | Keep only experiments from selected dates |
| **Dataset** | Keep only selected datasets |
| **Methods** | Keep only selected `{SLAM_Config}_{Fusion}_{Label}` strings |
| **Fusion** | Keep only selected fusion methods |
| **SLAM** | Filter by SLAM config base name (noise/jump tokens stripped) |
| **Trans Noise** | Keep only experiments with selected translational noise level |
| **Exclude experiments** | Drop specific experiment IDs regardless of other filters |
| **Scenes** | Filter per-scene views to selected scenes |
| **Paper baselines** | Toggle which paper baselines appear in plots |

The status line shows `N experiments loaded · M after filter`.

---

## Tabs

### 📊 Overview

Bar chart of any scalar metric grouped by experiment. Controls:

- **Segment**: All / Head / Common / Tail — selects which frequency third of classes to plot. Head = most frequent, Tail = least frequent.
- **Type**: IoU or Acc.
- **Color by**: SLAM - Fusion, Fusion, Method, SLAM_Config, or Label.

### 📈 Drift Analysis

Line plots showing how metrics degrade with increasing SLAM drift.

| Control | Options |
|---|---|
| **X axis** | `Trans_Noise` (continuous Gaussian noise) or `Jump_Count` (discrete trajectory jumps) |
| **Metrics** | mIoU, mAcc, AP_agnostic, AP_agnostic_50 — any subset |
| **Per-scene lines** | Off: one line per experiment (aggregate). On: uses per-scene data, color = experiment, line style = scene. Shows which scenes are most sensitive to drift. |
| **Color by** | Method / SLAM_Config / Fusion / Label |

When `Jump_Count` is selected but only one value exists in the loaded data, a single-point warning is shown (no line to draw until more jump experiments are run).

### 🎨 Per-class

Heatmap of IoU or Acc per class across experiments.

- **Aggregate** sub-tab: all scenes combined.
- **By Scene** sub-tab: select a single scene, see per-class breakdown for that scene.

Rows = classes (dataset label order, most to least frequent), columns = experiments.

### 🗂️ Conf. Matrix

Select one experiment → displays its pre-rendered `confmat.png`.

### 🗺️ By Scene

Three visualisations using per-scene data:

1. **Bar chart**: mIoU or mAcc per scene, grouped by experiment.
2. **Instance counts**: `Num_Instances` per scene.
3. **Per-class heatmap**: class × experiment for the filtered scene set.

### 🔬 Compare

Dedicated A/B (or A/B/C…) comparison view, **independent of sidebar filters**.

**Experiment picker** (top):
- **Baseline**: the reference experiment. All deltas computed against it.
- **Compare against**: one or more experiments to compare.

Labels show `{Label} · {date} [{SLAM_Config}]` for easy identification.

#### Summary Table

One row per experiment. Columns: Label, SLAM_Config, mIoU, mAcc, AP_agnostic, AP_agnostic_50, Num_Instances, Fusion_Accept_Rate. For each compare experiment, `Δ {metric}` columns show `↑/↓ ±value` relative to baseline.

#### Radar Chart

Spider chart with one polygon per experiment. Default axes: mIoU, mAcc, AP_agnostic, Num_Instances, Fusion_Accept_Rate. Each axis normalised to the max across selected experiments; NaN → 0. Useful for a quick "shape" comparison. Axes are configurable via multiselect.

#### Scene-by-Scene Bars

Faceted bar chart — one subplot per selected metric (mIoU, AP_agnostic, Num_Instances by default), scenes on X, experiments as hue. All subplots share the X axis. Shows where drift hurts scene by scene.

#### Delta Heatmap

Rows = scenes, columns = (experiment × metric) pairs. Cell value = `experiment_value − baseline_value`. Diverging colormap (green = better than baseline, red = worse). Annotated with exact delta values. Default metrics: mIoU, mAcc, AP_agnostic.

#### Scene Summary Table

Collapsible table — one row per (experiment × scene). Columns: Scene, Experiment (Label), then all metrics. For every compare experiment, `Δ {metric}` columns show `↑/↓ ±value` relative to baseline for that scene. Full numeric view of what the delta heatmap visualises.

#### Scene Deep-Dive

Pick one scene from a selectbox. Shows three things for that scene only:

1. **Metric table**: one row per selected experiment — all scalar metrics for that scene.
2. **Per-class heatmap**: class × experiment IoU or Acc — reveals which object categories drive the scene-level difference.
3. **Fusion decisions**: accept rate + rejection breakdown bars, scoped to that scene. Only shown when fusion data exists.

#### Fusion Behaviour (all scenes)

Two panels using per-scene fusion decision data (`{scene}/fusion_decisions.csv`):

- **Top**: Accept rate per scene per experiment (grouped bars). Low accept rate = many candidate pairs rejected.
- **Bottom**: Rejection reason breakdown stacked bar — `Centroid` (centroids too far), `CosSim` (feature similarity too low), `Overlap` (mask overlap gate). Summed across all scenes per experiment.

Only shown when fusion decision data is present (non-empty `fusion_decisions.csv` files).

---

## Metrics Reference

| Metric | Description |
|---|---|
| `mIoU` / `mAcc` | Mean IoU / Acc over all semantic classes |
| `Head_mIoU` / `Head_mAcc` | First third of classes (most frequent) |
| `Common_mIoU` / `Common_mAcc` | Middle third |
| `Tail_mIoU` / `Tail_mAcc` | Last third (least frequent) |
| `AP` | Class-aware instance AP at IoU [0.5:0.95] |
| `AP_50` / `AP_25` | Class-aware AP at fixed IoU thresholds |
| `AP_agnostic` | Class-agnostic AP — ignores predicted class label |
| `AP_agnostic_50` / `AP_agnostic_25` | Class-agnostic AP at fixed thresholds |
| `Num_Instances` | Predicted instance count |
| `Fusion_Accept_Rate` | Fraction of candidate pairs that were fused |

All IoU/Acc/AP values are decimals in `[0, 1]`, not percentages.

---

## Saving Figures

Every plot has a **💾 Save figure** expander below it. Set the output path and DPI, then click **Save**.

---

## SLAM_Config Naming

| Token | Meaning | Example |
|---|---|---|
| `GT` | Ground-truth poses, no noise | `GT` |
| `GTNoise-T{t}-R{r}` | GT poses + Gaussian noise | `GTNoise-T0p005-R0p01` |
| `GTJump-J{n}-T{t}-R{r}` | GT poses + discrete trajectory jumps + noise | `GTJump-J1-T0p05-R2p0` |
| `ORBSLAM3` | ORB-SLAM3 estimated poses | `ORBSLAM3` |

`T0p05` = 0.05, `R2p0` = 2.0 (integer part `p` decimal part). `J1` = 1 jump event.
