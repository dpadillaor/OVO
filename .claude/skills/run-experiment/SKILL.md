---
name: run-experiment
description: Set up and launch an OVO experiment via the batch runner. Use during task verification when you need to run a new experiment to check if an implementation works as expected.
user-invocable: true
disable-model-invocation: false
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
---

# run-experiment — Launch an OVO experiment

Arguments: `$ARGUMENTS`

Parse $ARGUMENTS to extract the experiment config. If critical info is missing and cannot be inferred from task context, ask before proceeding.

---

## Manifest file

Create a new YAML file at `manifests/<Experiment_ID>.yaml` (predict the name from the config before running — see naming convention below). This file serves as both the manifest to execute and the permanent record of what was launched. Never touch `manifests/experiments_manifest.yaml`.

```yaml
default_dataset: Replica

experiments:
  - label: <label>            # required, hyphens only (no underscores)
    scenes_id: <scene>        # single scene name (string) or list
    stages: [run, segment, eval]   # add eval_instances for instance AP metrics
    ovo_config:
      slam:
        slam_module: simulated   # default backend
        close_loops: true
      semantic:
        fusion_method: clip        # default fusion method
        # additional semantic keys depending on fusion_method — see below
      vis:
        stream: true
        type: "rerun"
        rerun_mode: stream         # stream | fusion | loop_closure
        rerun_visual_mode: "off"   # off = no live viewer; change to spawn/serve if needed
        save_rrd: true             # always save recording by default
    # slam_config block only if noise > 0:
    # slam_config:
    #   noise:
    #     translation_noise_std: <float>
    #     rotation_noise_std: <float>
    # restore_pre_fusion_checkpoint: data/checkpoints/Replica/<experiment>/<scene>/pre_fusion.ckpt
    # When set, the run stage skips SLAM entirely and replays only fusion from the checkpoint.
    # experiment_uid: <fixed-uid>   # fix the UID suffix instead of a random one; reuse across
    #                               # entries/runs to write multiple scenes into ONE shared folder.
```

### Experiment name convention
`{DATE}_{SLAM_TOKEN}_{FUSION_TOKEN}_{LABEL}_{UID}` — use this to name the file before running.
The 5-char hex UID is appended automatically by the runner. Predict it as `xxxxx` when naming the manifest file.
Example: `20260407_GT_CLIP_baseline.yaml` (manifest) → `20260407_GT_CLIP_baseline_a3f7c` (actual folder)

Set `experiment_uid: <fixed>` on an entry to override the random suffix — the folder
becomes `{DATE}_{SLAM_TOKEN}_{FUSION_TOKEN}_{LABEL}_{fixed}`. Reuse the same value
across entries to share one folder (see Shared output folder below).

SLAM tokens: `GT`, `GTNoise-T{t}-R{r}`, `GTJump-J{n}`, `ORBSLAM3`, `Vanilla`
Fusion tokens: `CLIP`, `PE-Core`, `PE-Spatial`, `SAM3`, `DINO`

---

## Configuration reference

### `scenes_id` / `scenes_list` — Scene selection

Three mutually exclusive options (priority: `scenes_list` > `scenes_id` > omitted):

| Option | Example | Use when |
|---|---|---|
| `scenes_id: office0` | Single scene (string) | Quick verification |
| `scenes_id: [office0, room0]` | Multiple scenes (list) | Partial run |
| `scenes_list: scenes.txt` | All Replica scenes from file | Full evaluation |
| *(omitted)* | Uses dataset config default | — |

Available Replica scenes (as listed in `scenes.txt`):
```
office0  office1  office2  office3  office4
room0    room1    room2
```
Use `office0` by default for verification — it is the fastest scene.
Use `scenes_list: scenes.txt` when a full evaluation across all scenes is needed.

### `experiment_uid` — Shared output folder (run scenes separately, same experiment)

By default each manifest entry gets a fresh random 5-char UID, so splitting scenes
across entries/runs produces **separate folders**. Set the same `experiment_uid` on
multiple entries to make them resolve to **one** folder; scenes land in their own
subfolders (`{exp}/{scene}/`) and never clobber each other.

```yaml
experiments:
  - label: jump-multi
    scenes_id: office0
    experiment_uid: shared01
    stages: [run, segment, eval]
    ovo_config: { slam: {slam_module: simulated, close_loops: true}, semantic: {fusion_method: clip} }
  - label: jump-multi
    scenes_id: room0
    experiment_uid: shared01
    stages: [run, segment, eval]
    ovo_config: { slam: {slam_module: simulated, close_loops: true}, semantic: {fusion_method: clip} }
```
→ `data/output/Replica/{DATE}_GT_CLIP_jump-multi_shared01/{office0,room0}/`

Notes:
- The folder name is `{DATE}_{SLAM_TOKEN}_{FUSION_TOKEN}_{LABEL}_{experiment_uid}` — keep `label` and tokens identical across the entries you want merged.
- `experiment_meta.json` is written once (first writer wins); keep the experiment-level params the same across the shared entries.
- `date_str` is per-day. Same day → shared folder. Across days the same uid yields different names.
- True parallelism: launch the runner as separate processes, each on a 1-scene manifest with the same `experiment_uid` (entries within one manifest run sequentially).

### `slam_module` — SLAM backends

#### Simulated — clean poses (SimulatedSLAM)
Token: `GT`
```yaml
ovo_config:
  slam:
    slam_module: simulated
    close_loops: true
```

#### Simulated + noise
Token: `GTNoise-T{t}-R{r}` (e.g. `GTNoise-T0p001-R0p01`)
```yaml
ovo_config:
  slam:
    slam_module: simulated
    close_loops: true
slam_config:
  noise:
    translation_noise_std: 0.001   # metres; common values: 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05
    rotation_noise_std: 0.01       # degrees; common value: 0.01
```
The runner merges noise into the per-experiment temp config automatically and sets `noise_enabled: true`.

#### Simulated + jump drift
Token: `GTJump-J{n}` where `n` = number of jumps (e.g. `GTJump-J3`)
```yaml
ovo_config:
  slam:
    slam_module: simulated
    close_loops: true
slam_config:
  noise:
    jump_drift_enabled: true
    jump_seed: 42          # seed for random-direction jumps
    jumps:
      - kf_index: 10
        translation_magnitude: 0.5   # random direction, metres
        rotation_magnitude: 20.0     # random axis, degrees
      - kf_index: 30
        translation_magnitude: 0.5
        rotation_magnitude: 20.0
      # add more entries for more jumps; recommend 3–4 max
```
Each jump entry supports either explicit vectors or random magnitudes:
- `translation: [x, y, z]` — explicit XYZ metres, OR `translation_magnitude: float` — random direction
- `rotation: [rx, ry, rz]` — explicit euler degrees (world-frame, axis-angle via Rodrigues), OR `rotation_magnitude: float` — random axis (world-frame), OR `rotation_jump: {...}` — local-frame yaw/pitch/roll (see below, takes precedence over `rotation`/`rotation_magnitude` if both are present)
- `forward: true` — (optional, only with `translation_magnitude`) applies the translation along the camera's forward (-Z) direction at that keyframe instead of a random world-space direction. Useful for jumps "hacia la cámara".

`noise_enabled` must NOT be set (jump drift uses `jump_drift_enabled`, a separate flag).

##### Local-frame rotation jump (`rotation_jump`)

Models a tracking-loss-style heading flip: the rotation is expressed relative to the
camera's own axes at the trigger instant (not a fixed world-space axis), so it reads
the same regardless of the camera's current pitch/roll. Internally it is conjugated
by the camera's *currently drifted* pose (`offset @ gt_pose` at trigger time), so
chained jumps stay coherent with each other and with every subsequent frame.

```yaml
slam_config:
  noise:
    jump_drift_enabled: true
    jump_seed: 42
    jumps:
      - frame_id: 120
        rotation_jump:
          yaw:   { enabled: true,  std_deg: 40, max_deg: 150 }   # heading, most common
          pitch: { enabled: false, std_deg: 8,  max_deg: 60 }
          roll:  { enabled: false, std_deg: 5,  max_deg: 30 }
```

Each axis (`yaw`/`pitch`/`roll`) is independent:
- `enabled` — if `false` (or the axis block is omitted), that axis contributes 0 and is not sampled
- `std_deg` — standard deviation (degrees) of a zero-mean normal distribution
- `max_deg` — resampled (not clipped) until the sampled value falls within `[-max_deg, max_deg]`, so no single artifact value skews the distribution's tail

Axis convention (OpenCV: X right, Y down, Z forward): `yaw` rotates about Y, `pitch` about X, `roll` about Z. Composed as `Rz(roll) @ Rx(pitch) @ Ry(yaw)` (yaw applied first) — irrelevant when only one axis is active, which is the common case.

Most experiments only need `yaw` (heading is the axis that drifts most in a real tracking loss, since there's no gravity/vanishing-point reference to self-correct it).

The `GTJump-J{n}` token reflects this: `rotation_jump` entries encode the enabled axes' `std_deg` (e.g. `GTJump-J1-T0p5-RY40` for yaw std=40°) instead of `R{rotation_magnitude}`.

##### Pre-fusion checkpoint (optional)
Add `save_pre_fusion_checkpoint: true` to save the geometric + semantic state just before fusion runs. Saved to `data/checkpoints/<experiment>/<scene>/pre_fusion.ckpt`. Enables replaying only the fusion step with different parameters via `scripts/replay_fusion.py`.

```yaml
slam_config:
  noise:
    jump_drift_enabled: true
    save_pre_fusion_checkpoint: true
    jumps:
      - ...
```

Replay with different fusion config:
```bash
python scripts/replay_fusion.py \
  --checkpoint data/checkpoints/<experiment>/<scene>/pre_fusion.ckpt \
  --experiment-name <new_experiment_name> \
  --fusion-config data/working/configs/Replica/new_fusion.yaml
```
Then run `--segment --eval` on the new experiment as normal.

Alternatively, replay via the batch runner manifest (integrates with segment/eval stages):
```yaml
experiments:
  - label: my-replay
    scenes_id: office0
    stages: [run, segment, eval]
    restore_pre_fusion_checkpoint: data/checkpoints/Replica/<experiment>/<scene>/pre_fusion.ckpt
    ovo_config:
      semantic:
        fusion_method: clip
        # override fusion params here
```
When `restore_pre_fusion_checkpoint` is set, the `run` stage skips SLAM entirely and runs only fusion from the checkpoint.

#### ORB-SLAM3 (real SLAM, no noise)
Token: `ORBSLAM3`
```yaml
ovo_config:
  slam:
    slam_module: orbslam3
    close_loops: true
```

#### Vanilla (minimal SLAM, no loop closures)
Token: `Vanilla`
```yaml
ovo_config:
  slam:
    slam_module: vanilla
```

### `fusion_method` — Fusion methods and their required config blocks

#### `clip` (default)
No extra config needed. Uses SigLIP-384 CLIP features.
```yaml
semantic:
  fusion_method: clip
```

#### `pe` — Perception Encoder
Requires a `pe:` block under `semantic:`.
```yaml
semantic:
  fusion_method: pe
  pe:
    model_card: PE-Core-L14-336      # or PE-Spatial-L14-448
```
- `PE-Core-L14-336` → experiment token `PE-Core`
- `PE-Spatial-L14-448` → experiment token `PE-Spatial`

#### `sam3` — SAM3 finetuned perception model
Requires a `sam3:` block under `semantic:`.
```yaml
semantic:
  fusion_method: sam3
  sam3:
    components: vit_only
    load_from_hf: true
```

#### `dino` — DINO features (not fully implemented)
```yaml
semantic:
  fusion_method: dino
```

### `fusion_criteria` — Custom criterion chain (optional)

Override the default criterion chain for any `fusion_method`. Default chains all use `[cooccurrence, centroid, cos_sim, overlap]`.

```yaml
semantic:
  fusion_method: clip
  fusion_criteria: ["centroid", "cos_sim", "overlap"]   # skip cooccurrence veto
```

Available criteria (run in order listed):

| Criterion | What it does |
|---|---|
| `cooccurrence` | Vetoes pairs that co-occurred in > N keyframes (different objects seen together) |
| `centroid` | Rejects pairs with centroid distance > `th_centroid` |
| `aabb` | Rejects pairs whose axis-aligned bounding boxes are > `th_aabb` apart (more robust than centroid for elongated objects) |
| `cos_sim` | Rejects pairs with cosine similarity < `th_cossim` |
| `overlap` | Symmetric: fraction of smaller cloud's points within `th_points` of larger cloud. Accepts if > 0.5 (or > 0.2 if cos_sim > 0.9) |
| `overlap_old` | Asymmetric: fraction of points1 within `th_points` of pcd2 (not normalized by cloud size). Same accept thresholds as `overlap`. Use to compare against legacy behaviour. |

**Note:** the experiment name token only encodes `fusion_method`. Use `label` to distinguish experiments with custom chains (e.g. `clip-no-cooc`).

### Covisibility filter (optional)

Wraps `_fuse_overlapping_instances` with a frustum-overlap pre-filter so only
instance pairs whose keyframes actually see each other get compared. Disabled by
default. When enabled, the runner appends a covisibility token to the
experiment name (see naming above).

```yaml
ovo_config:
  semantic:
    covisibility:
      enabled: true                 # turn the filter on
      min_overlap_ratio: 0.3        # estimator threshold for KF↔KF edges (frustum corners inside)
      max_distance: 5.0             # estimator pre-filter on translation distance (m)
      min_covisibility_overlap: 0.3 # min edge overlap to keep an instance pair
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Build the covisibility graph and use it during fusion |
| `min_overlap_ratio` | `0.3` | Min frustum-corner overlap to add an edge between two KFs |
| `max_distance` | `5.0` | Skip KF pairs farther than this (metres) without computing overlap |
| `min_covisibility_overlap` | `0.3` | Min edge overlap when picking instance pairs to fuse |

When `enabled: false` (or block omitted), the original brute-force pair loop
runs unchanged.

### Contest mechanism (merge/split discriminator)

The contest mechanism is an alternative fusion path. It watches which points one
instance's mask steals from another, then a discriminator emits per-pair verdicts
(MERGE_CONTAINMENT / SPLIT / NO_ACTION) using containment + geometry guards
(seam-normal turn, persistence) instead of the classic criterion chain.

Two top-level `semantic:` keys control whether it acts:

| Key | Default | Values | Meaning |
|---|---|---|---|
| `contest_fusion` | `observe` | `observe` / `only` / `both` | `observe` = classic fusion runs, contest only logs verdicts. `only` = contest drives merges/splits, classic fusion skipped. `both` = contest merges first, classic fusion on the rest. |
| `contest_split_mode` | `off` | `off` / `partial` / `dominance` / `all` | Which SPLIT verdicts actually get applied. `off` = none (merges only). `partial`/`dominance` = only that band. `all` = both bands. |

Tuning lives in a nested `contest:` block under `semantic:` (overrides `ovo.yaml`):

```yaml
ovo_config:
  semantic:
    contest_fusion: only
    contest_split_mode: all
    contest:
      reeval_frontier: true       # phase-2: re-judge "frontera real" with batch union-find
      max_seam_angle: 15.0        # dominance: normal-turn threshold (deg) for object-in-contact
      min_persist_split: 0.1      # dominance/partial: below this persistence -> no split
```

| `contest:` key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch for the contest store/record path |
| `high` | `0.7` | Containment ≥ this → strong band (fragment → merge) |
| `low` | `0.5` | Containment ≥ this and < `high` → partial (ambiguous) band |
| `min_mass` | `50` | Min Σ-KF mass for a pair to be considered (noise gate) |
| `min_split_cont` | `0.05` | Min containment for the dominance band |
| `max_rev_split` | `0.5` | Reverse-containment above this → bidirectional → no split |
| `min_persist_split` | `0.1` | Persistence below this → no split (flicker / grazing) |
| `sim_merge` | `0.81` | Descriptor cos-sim ≥ this in partial band → merge |
| `max_seam_angle` | `15.0` | Dominance: seam normal-turn (deg) above this → object in contact → no split |
| `reeval_frontier` | `false` | Phase-2 re-evaluation: build a union-find from the batch's merges and re-judge `frontera real (>=2 raíces)` verdicts with the real root; collapses cases where N winner-IDs are actually one object (e.g. 95→78). Logs `reason="frontera resuelta por root real"`. |
| `min_count` | `1` | Min per-point claim count to aggregate |
| `prune_every` | `10` | Prune the store every N reports |

Recipes:

```yaml
# Classic fusion, no contest, cooccurrence veto removed
semantic:
  fusion_method: clip
  fusion_criteria: ["centroid", "cos_sim", "overlap"]
  # contest_fusion: observe (default), contest_split_mode: off (default)

# Full contest mechanism (merges + all splits + phase-2 reeval)
semantic:
  fusion_method: clip
  contest_fusion: only
  contest_split_mode: all
  contest:
    reeval_frontier: true
```

Verdicts are written to `<scene>/fusion/contest_verdicts.csv` and the raw per-point store
to `<scene>/fusion/contest.json`. Analyze with the `contest-data` skill
(`Study_seg/contest_csv.py`, `contest_json.py`). Use `label` to encode the contest
config (e.g. `contest-only-reeval`) — the name token only encodes `fusion_method`.

### Key semantic parameters (optional overrides)

These live under `ovo_config.semantic:` and override `ovo.yaml > semantic:`.

| Key | Default | Meaning |
|---|---|---|
| `th_centroid` | `1.5` | Max centroid distance (m) — used by `centroid` criterion |
| `th_aabb` | `0.3` | Max AABB-to-AABB distance (m) — used by `aabb` criterion |
| `th_cossim` | `0.81` | Min cosine similarity for instance fusion |
| `th_points` | `0.1` | Min point overlap ratio for fusion |
| `cooccurrence_veto_threshold` | `5` | Min shared keyframes to veto a fusion (task 17) |
| `track_th` | `100` | Min points for an instance to be tracked |
| `match_distance_th` | `0.05` | Distance threshold for matching |
| `segment_every` | `10` | Run segmentation every N frames |

### `stages`

| Value | What it does |
|---|---|
| `run` | Runs OVO on the scene, produces `ovo_map.ckpt`. If `restore_pre_fusion_checkpoint` is set, skips SLAM and replays only fusion from the checkpoint. |
| `segment` | Runs instance segmentation on the saved map |
| `eval` | Semantic evaluation (mIoU/mAcc) against GT, writes `statistics.txt` |
| `eval_instances` | Instance AP evaluation (class-aware + class-agnostic) using masks from `segment`. Requires instance GT at `data/input/Datasets/{dataset}/instance_gt/{scene}.txt` |

Default: all three (run, segment, eval). For re-evaluation only: `stages: [eval]`. For re-segmentation + eval: `stages: [segment, eval]`. To include instance AP: `stages: [run, segment, eval, eval_instances]`.

### Checkpoint replay

Replay only the fusion step from a pre-fusion checkpoint (saved during a jump-drift run). SLAM, tracking, and segmentation are skipped. Stats from the original run are merged with the new fusion stats.

```yaml
experiments:
  - label: my-replay
    scenes_id: office0
    stages: [run, segment, eval]
    restore_pre_fusion_checkpoint: data/checkpoints/Replica/<experiment>/office0/pre_fusion.ckpt
    ovo_config:
      semantic:
        fusion_method: clip
        th_cossim: 0.75             # new fusion params to test
        fusion_criteria: ["centroid", "cos_sim", "overlap"]
```

Checkpoints are saved at `data/checkpoints/Replica/<experiment>/<scene>/pre_fusion.ckpt` only when both `jump_drift_enabled: true` and `save_pre_fusion_checkpoint: true` are set in the noise config. Non-jump runs do not produce checkpoints.

Naming: use a new `label` that identifies the changed fusion params (e.g. `ckpt-cossim075`). The experiment name will be `{DATE}_{SLAM_TOKEN}_{FUSION_TOKEN}_{label}_{UID}` as usual.

---

## Working directory

By default, run everything from the **main repo root** (`/home/padidavid/repos/OVO`).

If the task specifies a worktree (tasks have their own worktree under `.claude/worktrees/<worktree-name>/`), all paths and commands must be run from that worktree root instead:
```
.claude/worktrees/<worktree-name>/
```

The worktree has its own copy of all configs, manifests, and outputs. Never run the experiment from the main repo when the implementation being tested lives in a worktree — you would be testing the wrong code.

---

## Steps

### 1 — Determine root
- No worktree → `root = /home/padidavid/repos/OVO`
- Worktree → `root = /home/padidavid/repos/OVO/.claude/worktrees/<worktree-name>`

### 2 — Check data symlink (worktree only)
```bash
ls <root>/data/input/Datasets/Replica
```
If missing or empty: **stop and tell the user**. Do not proceed.

### 3 — Create manifest file
Write `<root>/manifests/<Experiment_ID>.yaml` with the full manifest structure (see above). This file is the permanent record — never delete it.

> **Concurrent runs**: the batch runner writes each experiment's config to a temp file `data/working/configs/ovo.<experiment_name>.yaml` and passes it via `--config`. The shared `ovo.yaml` is never modified. Multiple runners can execute simultaneously without conflict.

### 4 — Preview (recommended)
```bash
cd <root> && conda run -n ovo2 python scripts/run_experiments_batch.py --manifest manifests/<Experiment_ID>.yaml --preview
```
Read the generated YAML in `data/working/config_preview/` and verify it looks correct.

### 5 — Run
```bash
cd <root> && conda run -n ovo2 python scripts/run_experiments_batch.py --manifest manifests/<Experiment_ID>.yaml --verbose
```

### 6 — Report
- `Experiment_ID`
- Status (SUCCESS / FAILED)
- Errors if any

Then suggest: `/read-results <new_id> --baseline <baseline_id> --goal <what the task aimed to improve> [--worktree <worktree-name>]`

---

## Experiment output location
`<root>/data/output/Replica/<Experiment_ID>/`

---

## Rerun visualization

Rerun has two independent controls:
- **Live visualization transport** (`rerun_visual_mode`): `off`, `spawn`, or `serve`
- **Recording** (`save_rrd`): `true/false`

Use them independently depending on your workflow.

### Common manifest block
Add to the manifest under `ovo_config:`:
```yaml
ovo_config:
  vis:
    stream: true
    type: "rerun"                 # required
    rerun_mode: loop_closure       # stream | fusion | loop_closure
    rerun_visual_mode: off         # off | spawn | serve
    save_rrd: true                 # independent from visual mode
```

Notes:
- `show_stream` is legacy compatibility only; prefer `rerun_visual_mode`.
- `save_rrd` works with any visual mode.

### Save only (no live viewer)
```yaml
ovo_config:
  vis:
    stream: true
    type: "rerun"
    rerun_mode: loop_closure
    rerun_visual_mode: off
    save_rrd: true
```

This writes `rerun.rrd` inside each scene output folder:
`<root>/data/output/Replica/<Experiment_ID>/<scene_name>/rerun.rrd`

### Live local viewer (same machine)
```yaml
ovo_config:
  vis:
    stream: true
    type: "rerun"
    rerun_mode: fusion
    rerun_visual_mode: spawn
    save_rrd: false
```

### Live remote viewer (experiment on server, viewer on laptop)
OVO arranca un servidor gRPC en el puerto `9877`. Abre el Rerun viewer y conéctate desde tu máquina:
```yaml
ovo_config:
  vis:
    stream: true
    type: "rerun"
    rerun_mode: loop_closure
    rerun_visual_mode: serve
    save_rrd: true
```

En el viewer local:
```
rerun connect grpc://localhost:9877
```
O si accedes via SSH forward: `ssh -L 9877:localhost:9877 <user>@<server>`

### Open recording
```bash
rerun data/output/Replica/<Experiment_ID>/<scene_name>/rerun.rrd
```

### `rerun_mode` options
| Mode | Shows |
|---|---|
| `stream` | Basic point cloud + camera trajectory |
| `fusion` | Instance fusion events (before/after merge) |
| `loop_closure` | Trajectory + point cloud before and after global correction (orange = before, corrected = after) |
