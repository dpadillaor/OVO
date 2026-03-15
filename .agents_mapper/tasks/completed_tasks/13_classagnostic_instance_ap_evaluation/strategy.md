# Strategy

## Phase 1 — Diagnosis & Visualization ✅ DONE

**Goal:** understand why class-agnostic AP metrics didn't make sense.

### What was done
- Added diagnostic block to `ins_eval_example/class_agnostic_ap_scratch.py` that loads `ovo_map.ckpt` and the GT mesh and compares bounding boxes.
- Added Open3D interactive viewer (`--vis-masks`) with GT and pred side by side in the same window, plus `--clip-z` to cut the ceiling and see inside the room.
- Added GT-oracle sanity check: evaluating with GT masks directly scores 1.0 → eval pipeline is correct.
- Ran class-agnostic AP on experiment `20260305_GT_PE-Core_ComparativaPaper / office0`: **9% mAP**.

### Conclusions from Phase 1
See `conclusions.md` for full details. Summary:
- **Alignment is correct** (bounding boxes of pcd_pred and pcd_gt are nearly identical).
- **Eval pipeline is correct** (GT-oracle scores 1.0).
- **The low AP is caused by over-segmentation**: the predicted map is fragmented into many small instances. Large objects (sofa, chairs, table) are segmented well individually, but each object is split into multiple predicted instances, each with low IoU against the single GT instance — causing most to fall below AP thresholds.

### Script flags added
```bash
# Verify alignment (compare bounding boxes + Open3D viewer)
python ins_eval_example/class_agnostic_ap_scratch.py

# GT-oracle + OVO AP evaluation
python ins_eval_example/class_agnostic_ap_scratch.py --eval --gt_path ins_eval_example/replica_gt/

# Save colored PLY files (open in MeshLab/CloudCompare)
python ins_eval_example/class_agnostic_ap_scratch.py --save-ply --gt_path ins_eval_example/replica_gt/

# Interactive Open3D viewer (GT left, pred right, ceiling clipped)
python ins_eval_example/class_agnostic_ap_scratch.py --vis-masks --gt_path ins_eval_example/replica_gt/ --clip-z 1.3
```

## Phase 2 — Root Cause of Over-Segmentation (current)

The AP is low because predicted instances are fragmented. Need to understand why.

Questions to answer:
- Is over-segmentation happening at the SLAM/OVO level (too many raw instances created) or at the `match_labels_to_vtx` projection step?
- How many GT instances are there vs predicted instances? (GT: ~N unique IDs in office0.txt, Pred: 145 instances)
- What is the distribution of per-instance IoU? How many pred instances achieve IoU > 0.25 with any GT instance?
- Is merging/fusion working correctly during map building?

## Phase 3 — Fix & Clean Script

Once root cause is identified and fixed, consolidate into a clean reusable evaluation script.

## Phase 4 — Integration & Tests

- Integrate instance AP into the main pipeline (`run_eval.py`).
- Write unit/integration tests.
