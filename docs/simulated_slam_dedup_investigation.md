# SimulatedSLAM refactor & point-cloud dedup investigation

**Date:** 2026-05-29
**Branch:** `develop`
**Commits:** `2952750` (refactor + jump-drift dedup fix), `ef91155` (manifests relocation + scripts cleanup)

---

## 1. Context

The simulated SLAM backbone injects controllable error models on top of a
ground-truth trajectory to test OVO's semantic fusion. *Jump drift* applies a
large discrete pose offset at configured keyframes; a global geometric
correction at the end of the sequence snaps everything back to GT, which is
meant to trigger a massive re-fusion event.

**Reported symptom:** with `jump_drift_enabled` (2 cm / 0.01° jumps at KF 10 & 30),
a red jump label appeared in Rerun at the right frames, but the point cloud
showed **no visible discontinuity**. Later, after the global correction,
**lines/empty bands crossing the scene** appeared, as if the correction split
the scene into pieces.

---

## 2. Change made

### 2.1 Refactor (no behaviour change)

- Renamed `GroundTruthSLAM` → `SimulatedSLAM`; moved `ovo/slam/groundtruth_slam.py`
  into a package `ovo/slam/simulated/`:
  - `slam.py` — `SimulatedSLAM` orchestrator + `load_trajectory`
  - `tracking.py` — `TrackingStrategy` + `GroundTruthTracking` / `JumpTracking` / `NoisyTracking` (strategy pattern, OCP)
  - `jump_drift.py` — `JumpDriftController` (jump config, accumulated offset, `maybe_trigger`)
  - `keyframes.py` — `KeyframeSelector`
- Decomposed the former god-methods `map()` and `correct_map_globally()` into
  single-responsibility helpers; removed dead loop-closure code.
- Renamed the `slam_module` token `groundtruth` → `simulated` everywhere
  (factory, batch runner, config dir `slam/simulated/`, run-experiment skill).
  **No backwards-compat alias** (by request).

### 2.2 Epoch-aware dedup (behaviour change, jump-only)

`SimulatedSLAM` now owns its mapping instead of calling `VanillaMapper.map`:
`_add_points` → `_dedup_mask` / `_unproject` / `_append`. The dedup matching is
restricted to the **current epoch**: `self.pcd[self._dedup_min_idx:]`.

- `_dedup_min_idx` defaults to `0` → matches the whole map → **identical to
  vanilla** for clean/noise runs.
- A jump bumps `_dedup_min_idx` to the current pcd size, so re-observed
  surfaces after the jump are **not** matched against the pre-jump map and are
  kept as a fresh, displaced "ghost".
- The global correction resets `_dedup_min_idx` to `0`.

`vanilla_mapper.py` was left untouched; real SLAM backbones are unaffected.
Tests updated; epoch behaviour covered by
`tests/integration/test_gt_slam_jump_integration.py::TestJumpOpensDedupEpoch`.

---

## 3. Insights

### 3.1 The dedup was silently absorbing the jump
`match_distance_th = 0.05` (5 cm) > jump translation (2 cm). Post-jump frames
re-observed surfaces, matched the pre-jump points within threshold, and the new
(displaced) points were **suppressed**. Net effect: the jump produced no ghost,
no visible discontinuity — **the experiment was effectively a no-op** at the
mapping stage. The red label fired regardless (pure telemetry, decoupled from
geometry).

### 3.2 The global correction math is exact
For keyframe *k* with accumulated offset `O_k`, points are placed at
`O_k · gt_k · p_cam`. The correction applies `T_k = gt_k · (O_k·gt_k)⁻¹ = O_k⁻¹`,
which maps points exactly to `gt_k · p_cam`. The jump chain (`O = T2·T1`) is
inverted correctly. So a perfectly clean final map is expected **if** point
sampling is coherent — the streaks point to incoherent *sampling*, not to a
wrong correction transform.

### 3.3 The dedup enforces **permanent over-segmentation** (the big one)
Observed: two instances belonging to one sofa — a small corner (seen first) and
the rest of the sofa — that **never fused**. With the epoch-dedup + jump they
**did** fuse.

Mechanism, confirmed by the user (clean run keeps them split; jump run merges
them):

- In a normal run, `VanillaMapper.map` matches each new frame's depth against
  **all** existing points. When the "rest of sofa" frames re-observe the corner
  region already owned by instance A, those pixels are **suppressed** — instance
  B never gets points on A's region.
- Therefore A and B **never share points / never overlap**, and the `overlap`
  fusion criterion can never merge them. First-come-first-served ownership is
  sticky → **once two instances of the same object are split, they can never
  re-merge via overlap.**
- The epoch-dedup, post-jump, locally lifts the suppression → bridging points
  are added in the corner → A grows into B → they overlap → they fuse.

**Key reframing:** the over-segmentation is not fundamentally a "dedup must add
points" problem. It is that the **fusion `overlap` criterion requires shared
points**, while the dedup structurally prevents shared points between
neighbouring instances.

### 3.4 The epoch-dedup is **not semantically neutral**
It adds ghost points that would not exist in a clean run. Those points change
instance shape → change fusion decisions. So a jump run can show fusions
(correct or spurious) a clean run never would. **Evaluating fusion quality on
jump runs is therefore confounded** — an apparent improvement may be
ghost-bridging, not genuine drift handling. (In the sofa case the merge is
semantically *correct* — the corner is sofa — but it was triggered by an
artifact, so it does not generalise.)

### 3.5 What dedup gives us vs. its limitations

| Aspect | Benefit | Limitation |
|---|---|---|
| Bounded map size | one point per surface patch (not ×N frames) | first-view locks geometry; no refinement |
| Speed | fewer points downstream | — |
| Stable point identity | created once | keeps first point regardless of quality |
| Region ownership | no double-counting | **permanent over-segmentation** (no bridging points) |
| Projective match (5 cm) | simple | threshold-sensitive; conflates "same surface" with "within th in current pose"; couples to pose error (drift/jump) |

---

## 4. Future investigation

### 4.1 Visual confirmation (immediate)
Re-run office0 with a large jump (`translation_magnitude: 0.5`, `rotation_magnitude: 20`)
and inspect the RRD: confirm the ghost appears at KF 10/30 and whether the
post-correction streaks disappear (hypothesis: coherent ghost → clean correction).

### 4.2 Fix over-segmentation at the fusion criterion, not the dedup (recommended)
Replace/augment the `overlap` criterion with a **proximity** criterion
(`aabb` / `centroid` nearest-neighbour distance — `aabb` already exists). This
would let adjacent same-object instances (corner + sofa) fuse **without shared
points, without ghost, without touching dedup or inflating the cloud**. Attacks
the root cause exposed in §3.3. Validate on the sofa case (clean run).

### 4.3 Pose-robust / refinable dedup alternatives
- **Voxel-grid occupancy** (world-space, TSDF-style): bound size by space not by
  projective match; pose-robust; any instance can fill an empty voxel.
- **Surfel fusion** (ElasticFusion-style): re-observation *updates* the existing
  point (weighted position/feature) instead of suppressing → refines geometry,
  still bounded, removes the first-view lock (§3.2).
- **Instance-aware dedup**: do not suppress across distinct instances that are
  fusion candidates.

### 4.4 Decouple geometry dedup from instance membership
Keep geometry deduped (bounded) but allow instance membership to be recomputed /
shared so two instances can claim spatially-adjacent deduped points → enables
merge without adding points.

### 4.5 Cost & sensitivity studies
- Quantify point-cloud growth from epoch-dedup vs. number of jumps (VRAM/time).
- `match_distance_th` sensitivity sweep (duplication vs. detail loss).

---

## 5. Open questions
- Are the post-correction streaks actually fixed by the coherent ghost, or is
  there a separate sampling/seam issue? (needs §4.1)
- How often does over-segmentation (§3.3) occur in clean runs across scenes?
  Worth measuring before deciding whether §4.2 is high-value.
- Is jump drift the right tool to stress fusion, given it perturbs the very
  geometry (via ghosts) that fusion consumes? Consider whether a cleaner
  drift-injection that does not alter point sampling is needed.
