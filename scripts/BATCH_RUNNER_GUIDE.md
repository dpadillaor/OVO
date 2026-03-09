# Batch Runner Guide

`scripts/run_experiments_batch.py` reads a manifest YAML file and runs a sequence
of OVO experiments, handling config backup/restore automatically.

---

## Usage

```bash
# Run all experiments in the manifest
python scripts/run_experiments_batch.py

# Show output from run_eval.py
python scripts/run_experiments_batch.py --verbose

# Preview configs without running anything (see §10)
python scripts/run_experiments_batch.py --preview

# Use a custom manifest or preview directory
python scripts/run_experiments_batch.py --manifest path/to/manifest.yaml
python scripts/run_experiments_batch.py --preview --preview-dir /tmp/my_preview
```

---

## 1. Experiment name format

Every experiment is assigned a name with four underscore-separated parts:

```
{DATE}_{SLAM}_{FUSION}_{LABEL}
```

| Part | Example | Description |
|------|---------|-------------|
| `DATE` | `20260305` | Today's date (YYYYMMDD) |
| `SLAM` | `GT`, `GTNoise-T0p001-R0p01`, `ORBSLAM3`, `Vanilla` | SLAM config token |
| `FUSION` | `CLIP`, `PE-Core`, `PE-Spatial`, `SAM3` | Fusion method token |
| `LABEL` | `baseline` | Free-text label from the manifest |

Full example: `20260305_GTNoise-T0p001-R0p01_PE-Core_noisetest`

---

## 2. Manifest structure

The manifest is a YAML file (default: `scripts/experiments_manifest.yaml`).

```yaml
default_dataset: Replica          # used when an experiment has no 'dataset' key

experiments:
  - label: baseline               # required; becomes the LABEL part of the name
    dataset: Replica              # optional — overrides default_dataset
    scenes_id: office0            # optional — see §3
    stages: [run, segment, eval]  # optional — default is all three

    ovo_config:
      slam:
        slam_module: groundtruth
      semantic:
        fusion_method: clip

    slam_config:
      noise:                      # optional — see §5
        translation_noise_std: 0.001
        rotation_noise_std: 0.01
```

### Full field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `label` | string | — | **Required.** Free text tag. |
| `dataset` | string | `default_dataset` | Dataset name (case-sensitive). |
| `scenes_id` | string or list | — | Scene(s) to run. |
| `scenes_list` | string | — | Path to a `.txt` file with one scene per line. |
| `stages` | list | `[run, segment, eval]` | Subset of stages to execute. |
| `ovo_config.slam` | dict | `{}` | Key/value pairs merged into `ovo.yaml > slam:`. |
| `ovo_config.semantic` | dict | `{}` | Key/value pairs merged into `ovo.yaml > semantic:`. |
| `slam_config.noise` | dict | `{}` | Noise parameters — written to `ovo.yaml > noise:` (see §5). |

---

## 3. Scene selection

Three mutually exclusive ways to specify scenes (priority order):

| Option | Example | Behaviour |
|--------|---------|-----------|
| `scenes_list` | `scenes_list: scenes.txt` | Reads scene names from the file, one per line. **Highest priority.** |
| `scenes_id` (string) | `scenes_id: office0` | Runs a single named scene. |
| `scenes_id` (list) | `scenes_id: [room0, room1]` | Runs each listed scene. |
| *(omitted)* | — | Uses the scenes defined in the dataset config. |

---

## 4. Fusion methods

Set `ovo_config.semantic.fusion_method` to one of:

| Value | Token | Extra fields |
|-------|-------|--------------|
| `clip` (or omitted) | `CLIP` | — |
| `pe` | `PE`, `PE-Core`, `PE-Spatial` | `ovo_config.semantic.pe.model_card` |
| `sam3` | `SAM3` | — |

PE model card examples:
- `PE-Core-L14-336` → token `PE-Core`
- `PE-Spatial-L14-448` → token `PE-Spatial`

Convenience shortcut: `fusion_method` can be placed directly under `ovo_config`
(without nesting it under `semantic`) and the runner will move it automatically.

---

## 5. Noise with groundtruth SLAM

Noise parameters belong under `slam_config.noise:` in the manifest. The runner
automatically writes them to the `noise:` section at the **root of `ovo.yaml`**
(which is where `GroundTruthSLAM` reads them) and sets `noise_enabled: true`.

```yaml
slam_config:
  noise:
    translation_noise_std: 0.001   # metres
    rotation_noise_std: 0.01       # degrees
```

> **Note:** You do not need to set `noise_enabled: true` manually — the runner
> does this automatically whenever `slam_config.noise` is non-empty.

---

## 6. SLAM modules

| `slam_module` | Token | Applicable noise | Notes |
|---------------|-------|-----------------|-------|
| `groundtruth` | `GT` / `GTNoise-…` | `translation_noise_std`, `rotation_noise_std` | Default. |
| `orbslam3` | `ORBSLAM3` | — | Noise not applicable. |
| `vanilla` | `Vanilla` | — | Noise not applicable. |

The slam config file path is resolved as:
```
data/working/configs/slam/{slam_module}/{dataset.lower()}.yaml
```

---

## 7. Dataset per experiment

```yaml
default_dataset: Replica      # fallback for all experiments

experiments:
  - label: scannet_run
    dataset: ScanNet           # overrides default for this experiment only
    ...
  - label: replica_run
    # no 'dataset' key — uses Replica
    ...
```

---

## 8. Common patterns

### Noise sweep

```yaml
default_dataset: Replica
experiments:
  - label: noise_001
    slam_config:
      noise: {translation_noise_std: 0.001, rotation_noise_std: 0.01}
    ovo_config:
      slam: {slam_module: groundtruth}
      semantic: {fusion_method: clip}

  - label: noise_005
    slam_config:
      noise: {translation_noise_std: 0.005, rotation_noise_std: 0.01}
    ovo_config:
      slam: {slam_module: groundtruth}
      semantic: {fusion_method: clip}
```

### Compare fusion methods

```yaml
default_dataset: Replica
experiments:
  - label: fusion_clip
    ovo_config:
      slam: {slam_module: groundtruth}
      semantic: {fusion_method: clip}

  - label: fusion_pe_core
    ovo_config:
      slam: {slam_module: groundtruth}
      semantic:
        fusion_method: pe
        pe:
          model_card: PE-Core-L14-336
```

### Multi-scene run

```yaml
experiments:
  - label: multi_scene
    scenes_id: [office0, office1, room0]
    ovo_config:
      slam: {slam_module: groundtruth}
      semantic: {fusion_method: clip}
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Noise has no effect | Old code bug (fixed). Upgrade to latest version. | Ensure `ovo.yaml > noise > noise_enabled` is `true` after setup. |
| `ValueError: Slam module '...' not recognized` | Typo in `slam_module`. | Use `groundtruth`, `orbslam3`, or `vanilla`. |
| `ValueError: Fusion method '...' not recognized` | Typo in `fusion_method`. | Use `clip`, `pe`, or `sam3`. |
| Experiment name has wrong parts | `label` contains underscores. | Labels are used verbatim; underscores are allowed but split the name visually. |
| Config not restored after failure | Unhandled exception skipped `cleanup()`. | The runner calls `cleanup()` in a `finally` block — check for import errors before the loop. |
| Wrong scene file path | `slam_config_path` points to a non-existent file. | Verify `slam_module` and `dataset` match a file under `data/working/configs/slam/`. |

---

## 10. Preview mode

Before running any experiment you can inspect the exact `ovo.yaml` that would be
applied to each one, without touching any real config file:

```bash
python scripts/run_experiments_batch.py --preview
```

One YAML file per experiment is written to `data/working/config_preview/`
(override with `--preview-dir`). The filenames follow the experiment name
convention, so you can immediately see which config corresponds to which run:

```
data/working/config_preview/
  20260305_GT_CLIP_baseline.yaml
  20260305_GTNoise-T0p001-R0p01_PE-Core_noisetest.yaml
  ...
```

Use this to verify before a long batch run that noise, fusion method, and SLAM
module are all wired up correctly.
