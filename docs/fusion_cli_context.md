# OVO Fusion — Context for CLI Development

This document gives a Claude instance working on an external CLI tool everything needed to query and visualize OVO fusion data, without access to the OVO codebase.

---

## What OVO Does (Brief)

OVO is a 3D semantic mapping system. It tracks a camera through a scene, segments objects with SAM2, encodes them with CLIP/DINO/PE, and builds a 3D instance map. When the SLAM backbone detects a map update (loop closure or periodic mapping), OVO runs **fusion**: it checks every pair of existing 3D instances and merges those that represent the same object.

---

## Output Directory Structure

All experiment outputs live under:
```
data/output/<dataset>/<experiment_folder>/<scene>/
```

- `<dataset>`: e.g. `Replica`
- `<experiment_folder>`: named `{DATE}_{SLAM_CONFIG}_{FUSION}_{LABEL}`, e.g. `20260422_GTJump-J1-T0p1-R2p0_CLIP_task1718-jump-cooc`
- `<scene>`: e.g. `room0`, `office1`, `office4` (Replica has 9 scenes total)

Files inside each scene folder:
```
config.yaml            # Full run config (thresholds, model, noise params, etc.)
fusion_decisions.csv   # One row per pair evaluated during fusion
ovo_map.ckpt           # Final map state: 3D points + per-instance features
estimated_c2w.npy      # Camera poses (optional, may not exist)
rerun.rrd              # Rerun visualization file (binary, not useful for queries)
logger/                # Per-frame timing logs (.log files)
```

---

## Fusion Heuristic

Fusion runs after every SLAM map update (`map_every=5` frames by default, but only when `map_updated=True` — i.e. on loop closures). All surviving 3D instances are compared pairwise.

There are **two variants** of the first geometric gate, depending on the branch/experiment:

### Variant A: Centroid gate (develop branch — current production)

```
1. centroid_dist = ||mean(points_i) - mean(points_j)||_2   (metres)
   → if centroid_dist > th_centroid: REJECT (reason="centroid"), stop

2. cos_sim = cosine_similarity(clip_feature_i, clip_feature_j)
   → if cos_sim < th_cossim: REJECT (reason="cos_sim"), stop

3. p_dist = fraction of points in instance_i within th_points metres of instance_j
   → ACCEPT if: p_dist > 0.5  OR  (cos_sim > 0.9 AND p_dist > 0.2)
   → else REJECT (reason="overlap")
```

CSV column for gate metric: `centroid_dist`

**Default thresholds:**
| Threshold | Default | Config key |
|-----------|---------|------------|
| `th_centroid` | 1.5 m | `semantic.th_centroid` |
| `th_cossim` | 0.81 | `semantic.th_cossim` |
| `th_points` | 0.1 m | `semantic.th_points` |

### Variant B: AABB gate (feat/fusion-aabb-distance branch)

Replaces centroid distance with AABB-to-AABB minimum distance. More robust to elongated objects (sofa, table) where centroids are far apart but shapes actually overlap.

```
AABB distance = Euclidean length of the per-axis gap vector between bounding boxes.
Returns 0.0 when AABBs intersect (overlap on all 3 axes).

gap = clamp(max(min_i - max_j, min_j - max_i), min=0)   # per axis
aabb_dist = ||gap||_2

1. if aabb_dist > th_aabb: REJECT (reason="aabb"), stop

2. cos_sim = cosine_similarity(clip_feature_i, clip_feature_j)
   → if cos_sim < th_cossim: REJECT (reason="cos_sim"), stop

3. p_dist (same as Variant A)
   → same ACCEPT rule
```

CSV column for gate metric: `aabb_dist` (replaces `centroid_dist`)

**Default thresholds:**
| Threshold | Default | Config key |
|-----------|---------|------------|
| `th_aabb` | 0.3 m | `semantic.th_aabb` |
| `th_cossim` | 0.81 | `semantic.th_cossim` |
| `th_points` | 0.1 m | `semantic.th_points` |

**The CLI must detect which variant was used** by checking which column is present in the CSV header (`centroid_dist` vs `aabb_dist`). The `reason` column will contain `"centroid"` or `"aabb"` accordingly.

**GeometricOnlyFusion** (fusion_method=geometric): skips cos_sim, uses the geometric gate + overlap only. Both variants apply.

**Cooccurrence veto** (optional feature): before running the geometry/semantic checks, pairs that co-occur too often in the same frame are vetoed early. These show `reason="cooccurrence"` with all metric columns empty. Threshold: `semantic.cooccurrence_veto_threshold` (default 8). When active, cooccurrence rejections dominate (can be >10% of all decisions).

---

## fusion_decisions.csv Format

One row per (frame, instance_pair) evaluation. Written incrementally during the run.

```
frame_id,result,i1,i2,reason,centroid_dist,cos_sim,p_dist
```

| Column | Type | Description |
|--------|------|-------------|
| `frame_id` | int | Frame index when fusion was triggered |
| `result` | str | `ACCEPTED` or `REJECTED` |
| `i1` | int | First instance ID |
| `i2` | int | Second instance ID |
| `reason` | str | Rejection cause: `centroid`, `cos_sim`, `overlap`, `cooccurrence`. Empty if ACCEPTED |
| `centroid_dist` | float | Euclidean centroid distance (metres). Empty for cooccurrence rejections |
| `cos_sim` | float | CLIP cosine similarity [0,1]. Empty if rejected at centroid or cooccurrence stage |
| `p_dist` | float | Point cloud overlap ratio [0,1]. Empty if rejected before overlap stage |

**Detect variant from header:** check which column exists — `centroid_dist` (Variant A) or `aabb_dist` (Variant B). The `reason` field mirrors this: `"centroid"` vs `"aabb"`.

**Typical distribution — Variant A (~8000–15000 rows/scene):**
- ~78% rejected by centroid (cos_sim and p_dist empty)
- ~11% rejected by cooccurrence (all metrics empty)
- ~6% rejected by cos_sim (p_dist empty)
- ~4% rejected by overlap (all metrics present)
- ~0.5% accepted (all metrics present)

**Useful queries on this CSV:**
- History of a pair: filter `(i1==A AND i2==B) OR (i1==B AND i2==A)`
- All accepted fusions: filter `result==ACCEPTED`
- Counterfactual for cos_sim threshold: for rows with reason=`cos_sim`, check if `cos_sim >= new_threshold`
- Counterfactual for centroid: only possible if you also have the ckpt (centroid data not stored for cooc-rejected pairs)
- Counterfactual for overlap accept rule: rows with reason=`overlap` have all three metrics — re-evaluate the accept condition

---

## ovo_map.ckpt Format

PyTorch checkpoint (load with `torch.load(..., weights_only=False)`). Contains **post-fusion final state**.

```python
ckpt = {
    "map_params": {
        "xyz":     Tensor[N, 3],   # 3D point positions (metres)
        "obj_ids": Tensor[N, 1],   # instance ID per point (int)
        "ids":     Tensor[N, 1],   # point index
        "max_id":  int,
        "color":   Tensor[N, 3],   # RGB colour per point
    },
    "ovo_map_params": {
        "ins_3d_ids": ndarray[M],              # list of surviving instance IDs
        "ins3d_0_clip_feature":    Tensor[1, D],  # CLIP embedding for instance 0
        "ins3d_0_clip_feature_kf": Tensor[1, D],  # keyframe-aggregated CLIP
        "ins3d_0_pe_feature":      Tensor[1, D],  # PE embedding (if used)
        "ins3d_0_pe_feature_kf":   Tensor[1, D],
        "ins3d_0_sam3_feature":    Tensor[1, D],  # SAM3 embedding (if used)
        # ... same pattern for ins3d_1, ins3d_2, ...
    }
}
```

- `M` = number of surviving instances (post-fusion), typically 100–200/scene
- `D` = 1152 for SigLIP-384 (CLIP), varies by encoder
- Instance IDs in `ins_3d_ids` match `obj_ids` in map_params and `i1`/`i2` in the CSV

**What you can do with the ckpt:**
- Reconstruct per-instance point clouds: `xyz[obj_ids.squeeze() == instance_id]`
- Get instance centroid: mean of its points
- Get CLIP feature: `ovo_map_params["ins3d_{idx}_clip_feature"]` where idx is position in `ins_3d_ids`
- Re-run fusion heuristic with different thresholds on surviving instances (detects under-fusion)

**What you cannot do:**
- Detect over-fusion (merged instances are gone)
- Replay intermediate fusion states (only final state saved)

---

## config.yaml — Relevant Fields

```yaml
semantic:
  fusion_method: clip          # clip | dino | pe | sam3 | geometric
  th_centroid: 1.5             # metres
  th_cossim: 0.81
  th_points: 0.1               # metres (overlap neighbourhood radius)
  cooccurrence_veto_threshold: 8   # present only when cooc feature active

noise:                         # GT SLAM noise injection (for robustness experiments)
  jump_drift_enabled: true
  jumps:
    - kf_index: 30
      translation_magnitude: 0.1   # metres
      rotation_magnitude: 2.0      # degrees
```

Experiment folder name encodes some of this: `GTJump-J2-T0p1-R2p0` = 2 jumps, 0.1 m translation, 2.0° rotation.

---

## Canonical Queries the CLI Should Support

### Summary (`/summary`)
From `fusion_decisions.csv`:
- Total decisions, accepted count + %, rejected count + %
- Rejection breakdown by reason (count + % of rejected)
- Number of frames with map updates

### Pair history (`/pair <i1> <i2>`)
All rows where `(i1==A AND i2==B) OR (i1==B AND i2==A)`, across all frames. Show frame, result, reason, metrics.

### Instance history (`/instance <id>`)
All rows where `i1==id OR i2==id`. Shows every pair this instance was evaluated against.

### Counterfactual (`/counterfactual cos_sim <new_threshold>`)
From rows with `reason=cos_sim`: how many would flip to ACCEPTED with the new threshold?
From rows with `result=ACCEPTED`: how many would flip to REJECTED?
**Caveat:** cascade effects not captured — this is single-frame, single-criterion only.

### Threshold sweep (`/sweep cos_sim <min> <max> <step>`)
For each threshold value, report: accepted count, rejected-by-cos_sim count that would flip. Useful for finding the elbow.

### Missing criteria (not yet implemented, need ckpt):
- `/rerun <th_centroid> <th_cossim> <th_points>` — re-evaluate fusion on final ckpt state with new thresholds, report what additionally fuses.
- `/pair-geometry <i1> <i2>` — show current centroids and point cloud overlap for two surviving instances (from ckpt).

---

## Ckpt Simulation Modes

When running `/rerun` on the final ckpt state, reconstruct per-instance point clouds from `xyz` + `obj_ids`, then re-apply the heuristic with new thresholds. Match the simulation mode to the variant that produced the experiment (detected from CSV header).

### Mode 1: Centroid (Variant A experiments)

```python
centroid_i = xyz[obj_ids == i].mean(dim=0)
centroid_dist = (centroid_i - centroid_j).norm()
gate = centroid_dist <= th_centroid
```

### Mode 2: AABB (Variant B experiments)

```python
min_i, max_i = xyz[obj_ids == i].min(dim=0).values, xyz[obj_ids == i].max(dim=0).values
gap = torch.clamp(torch.maximum(min_i - max_j, min_j - max_i), min=0)
aabb_dist = gap.norm()
gate = aabb_dist <= th_aabb
```

`aabb_dist == 0.0` means the bounding boxes intersect.

Both modes then apply the same cos_sim and overlap stages if gate passes.
