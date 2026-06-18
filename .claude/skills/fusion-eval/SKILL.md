---
name: fusion-eval
description: Interpret and cross-reference the fusion-evaluation artifacts (fusion_decisions_eval.csv, fusion_eval_summary.json, fusion_instance_stats.csv) produced by studies/fusion_metrics. Use when analyzing why fusion helped/hurt a run, tracing a worsened instance to the decision that caused it, or reading the agnostic-AP impact.
user-invocable: true
disable-model-invocation: false
allowed-tools:
  - Read
  - Bash
---

# fusion-eval — Interpret & cross-reference fusion-evaluation artifacts

The module `studies/fusion_metrics/` grades a run's **real** fusion merge/split
decisions against GT. Run it with the `ovo` env:

```bash
cd studies/fusion_metrics
/home/padidavid/anaconda3/envs/ovo/bin/python -m scripts.eval_fusion_decisions \
  --exp_path data/output/Replica/<EXP_ID> --scene office0     # writes 3 files next to the CSV
```
Needs a `pre_fusion.ckpt` (saved only when `jump_drift_enabled AND
save_pre_fusion_checkpoint`), mirrored under `data/checkpoints/.../<scene>/`.

Interactive debugger (GUI, run yourself): `python -m scripts.debug_merge_decisions
--exp_path <run> --scene office0 --z_max 1.5`. Commands: `gts` (list GT
instances), `gt <id>` (matched green + spurious red shades), `show <obj_id>`
(raw OVO points on GT — see drift), `missing` (instances lost in projection).

## The 3 artifacts (all under `data/output/Replica/<EXP_ID>/<scene>/`)

| File | Grain | What it answers |
|---|---|---|
| `fusion_decisions_eval.csv` | per **pair** | did each merge/split decision agree with GT? |
| `fusion_eval_summary.json` | per **scene** | aggregate quality + agnostic-AP impact |
| `fusion_instance_stats.csv` | per **GT instance** | which objects fusion helped/hurt |

### fusion_decisions_eval.csv
Original `fusion_decisions.csv` rows + `same_object` + `verdict`. Columns:
`frame_id,result,i1,i2,reason,centroid_dist,aabb_dist,cos_sim,p_dist,shared_kfs,same_object,verdict`
- `result` = fusion's call: `ACCEPTED` (merged) / `REJECTED`. `reason` = which gate rejected.
- `same_object` = GT truth (do i1,i2 share their dominant GT instance?).
- `verdict` (positive = MERGE): **TP** correct merge, **FP** over-merge, **FN** over-split, **TN** correct separation.

### fusion_eval_summary.json
- `counts` {TP,FP,FN,TN} and `rates` {precision,recall,f1} over **evaluated** pairs.
- `by_group` = verdicts split by ACCEPTED vs REJECTED/<reason> (see where FN/FP come from).
- `n_instances_pre/post` = real instance count before/after (post from `ovo_map.ckpt`).
- `n_skipped`/`skipped_obj_ids` = pairs dropped because an obj_id didn't project to GT (drift noise, NOT fusion signal).
- `agnostic_impact` = class-agnostic instance AP **pre vs post** fusion:
  - `delta_ap50` etc.: net effect on segmentation quality. ~0 or negative = fusion didn't help / hurt.
  - `delta_spurious50` (negative = consolidated over-split fragments — good), `delta_matched50` (negative = lost a matched object — an over-merge), `delta_missed50`.

### fusion_instance_stats.csv  (one row per GT instance)
`gt_id,name,iou_pre,iou_post,acc_pre,acc_post,matched_obj_id,n_spurious,spurious_obj_ids,status`
- `iou_pre/post` = best IoU of this GT with a prediction, before/after fusion. **IoU is the honest metric** (penalizes over-merge spill).
- `acc_pre/post` = coverage = intersection/|GT|. High acc + low IoU = prediction covers the object but **spills beyond** (over-merge / object swallowed by a big blob).
- `matched_obj_id` = prediction the **agnostic AP** credits to this GT (greedy 1:1, IoU≥0.5); `None` if unmatched.
- `n_spurious`/`spurious_obj_ids` = over-split fragments whose dominant GT is this object.
- `status` = improved / worsened / unchanged (iou_post vs iou_pre).
- Rows named `classN` (e.g. class0/31/40/93) are **background** (class not in `valid_ins_class_ids`) — not real objects; ignore for object-level analysis.

## Cross-referencing — the causal chain

The artifacts are at different grains; join by **obj_id** (`matched_obj_id` ↔
`i1`/`i2`). They share the **pre-fusion obj_id space**, so the keys match directly.

```
fusion_instance_stats.csv   WHICH object worsened?     (sofa 0.84 -> 0.78, matched_obj_id=7)
        | matched_obj_id
fusion_decisions_eval.csv   WHICH decision caused it?  (ACCEPTED 4<->7, same_object=False -> FP)
        | obj_id
debug `gt <id>` / `show`     SEE it in 3D              (the drift, the fragments)
```

Recipe — find the merge(s) that hurt a worsened instance (its `matched_obj_id`):
```bash
D=data/output/Replica/<EXP_ID>/<scene>
# ACCEPTED merges touching obj 7 (the sofa's matched prediction)
awk -F, 'NR>1 && $2=="ACCEPTED" && ($3==7||$4==7)' $D/fusion_decisions_eval.csv
# -> if same_object=False / verdict=FP, that merge over-merged a different object onto it
```
The partner obj_id usually appears in **another** instance's `spurious_obj_ids`
or as its `matched_obj_id` — that tells you which GT object got glued on.

## Interpreting (what the numbers mean)

- **recall low + huge FN** → fusion is too conservative (over-splits). Under drift, fragments of one object are displaced → `centroid`/`aabb` gates reject correct merges → FN piles up in `by_group.REJECTED/centroid`.
- **FP (over-merge)** → typically two **distinct same-class** objects, close (low `centroid_dist`) and semantically similar (high `cos_sim`); one bad merge can worsen **two** instances at once.
- **agnostic AP flat despite -spurious** → AP is recall-bound; cleaning over-split fragments doesn't raise AP if the matched/missed count doesn't move.
- **small objects** (camera, wall-plug): acc≈1 but IoU≈0 → swallowed by a much larger prediction.

## Gotchas
- Drift breaks the GT projection (`match_labels_to_vtx` assumes shared frame). On jump-drift scenes, `skipped_obj_ids` and some low IoUs are **projection noise**, not fusion error. Use `show <obj_id>` to see the displacement. A clean (no-drift) scene isolates the fusion effect.
- `n_instances_pre - n_instances_post` (real merges) can differ from summary `n_accepted` (evaluated-only): one universe is the full run, the other is GT-projected pairs.
