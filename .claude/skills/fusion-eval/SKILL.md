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
decisions against GT. Run it with the `ovo2` env:

```bash
cd studies/fusion_metrics
/home/padidavid/anaconda3/envs/ovo2/bin/python -m scripts.eval_fusion_decisions \
  --exp_path data/output/Replica/<EXP_ID> --scene office0     # writes 3 files next to the CSV
```
Needs a `pre_fusion.ckpt` (saved only when `jump_drift_enabled AND
save_pre_fusion_checkpoint`). Resolution when `--ckpt` is omitted: the run's own
(mirror `data/output` -> `data/checkpoints/.../<scene>/`), else the one its
`config.yaml` borrowed (`restore_pre_fusion_checkpoint`, `{scene}` placeholder —
a run launched from another run's shared checkpoint), else error. The post map
(`ovo_map.ckpt`) is always the run's own.

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
Four blocks (two universes: the real run vs the GT-projected pairs we could score):
- `run` = the real fusion: `instances_pre`, `instances_post` (raw unique obj_ids from `ovo_map.ckpt` points, no projection), `merges_applied` (= pre - post). Plus the projection cross-check: `instances_post_reprojected` = post instances that win ≥1 GT vertex (reproject `match_labels_to_vtx`, same count production's `instance_pred/<scene>.txt` writes), and `post_unmatched_gt` = obj_ids present in the raw post map but absent from the GT projection (have points, reach no vertex). `instances_post - instances_post_reprojected = len(post_unmatched_gt)`; non-empty means our raw count and production's evaluated count diverge — names exactly which.
- `evaluation` = what we could score: `pairs_total`/`pairs_scored`/`pairs_skipped`, `merges_scored`/`merges_skipped`, and `skipped_objects` = per dropped object `{obj_id, pairs, merges}` (how many decisions it was in, how many were real ACCEPTED merges). An obj is skipped when it doesn't project to GT (drift noise, NOT fusion signal).
  - Reconciliation: `run.merges_applied = merges_scored + merges_skipped`; `pairs_total = pairs_scored + pairs_skipped`. A `skipped_objects[i].merges > 0` is exactly why `merges_applied` exceeds `merges_scored`.
- `verdicts` (scored pairs only): `counts` {TP,FP,FN,TN}, `rates` {precision,recall,f1}, `by_group` (verdicts split by ACCEPTED vs REJECTED/<reason> — where FN/FP come from).
- `agnostic_impact` = class-agnostic instance AP **pre vs post** fusion, computed **two ways**:
  - `all` = every GT instance counts (background included). The raw view.
  - `objects` = production's full handling (`ins_eval_utils.evaluate` / `valid_ins_class_ids` + `min_region_size=100`): background-class GT instances are not targets; sub-100-vertex predictions **and** GT are dropped; the forgiven *ignore region* = background **plus** sub-min GT, and an unmatched prediction mostly on it (ignore fraction > IoU threshold) is not counted FP. IoU stays raw — predictions are kept whole, NOT carved (carving would inflate IoU and reward sloppy object+background blobs). This is the mode to compare against production / paper numbers (validated to match `instance_ap_<scene>.txt` `AP_agnostic*` within ~0.006, the residual being project-then-merge vs production's merge-then-project order); `all` runs higher spurious because background blobs count.
  - Each mode has `delta_ap50` etc. (net effect on segmentation quality; ~0 or negative = fusion didn't help / hurt), `delta_spurious50` (negative = consolidated over-split — good), `delta_matched50` (negative = lost a matched object via over-merge), `delta_missed50`, plus full `pre`/`post` per-threshold.

### fusion_instance_stats.csv  (one row per GT instance)
`gt_id,name,iou_pre,iou_post,acc_pre,acc_post,matched_obj_id_pre,n_spurious,spurious_obj_ids,status,match_status`
- `iou_pre/post` = best IoU of this GT with a prediction, before/after fusion. **IoU is the honest metric** (penalizes over-merge spill).
- `acc_pre/post` = coverage = intersection/|GT|. High acc + low IoU = prediction covers the object but **spills beyond** (over-merge / object swallowed by a big blob).
- `matched_obj_id_pre` = prediction the **agnostic AP** credited to this GT **pre-fusion** (greedy 1:1, IoU≥0.5); `None` if unmatched. This is the join key to `fusion_decisions_eval.csv` — even when the match was later lost, it points at the prediction that got over-merged.
- `n_spurious`/`spurious_obj_ids` = over-split fragments whose dominant GT is this object.
- `status` = improved / worsened / unchanged — **IoU magnitude** direction (continuous).
- `match_status` = the **AP match transition** @0.5 (discrete): `KEPT` (matched pre & post), `LOST` (matched pre, missed post — over-merge killed it), `GAINED` (missed pre, matched post — recovered), `UNMATCHED` (missed both). Pair it with `status`: e.g. `UNMATCHED`+`improved` = "fusion improved its IoU but not enough to cross 0.5"; `LOST`+`worsened` = over-merge dropped it below the bar; `KEPT`+`worsened` = degraded but still matched.
- Rows named `classN` (e.g. class0/31/40/93) are **background** (class not in `valid_ins_class_ids`) — not real objects; ignore for object-level analysis.
- Distribution cross-check: `KEPT+LOST = matched_pre`, `KEPT+GAINED = matched_post` (= summary `agnostic_impact.pre/post.per_threshold[0.5].matched`).

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

- **recall low + huge FN** → fusion is too conservative (over-splits). Under drift, fragments of one object are displaced → `centroid`/`aabb` gates reject correct merges → FN piles up in `verdicts.by_group.REJECTED/centroid`.
- **FP (over-merge)** → typically two **distinct same-class** objects, close (low `centroid_dist`) and semantically similar (high `cos_sim`); one bad merge can worsen **two** instances at once.
- **agnostic AP flat despite -spurious** → AP is recall-bound; cleaning over-split fragments doesn't raise AP if the matched/missed count doesn't move.
- **small objects** (camera, wall-plug): acc≈1 but IoU≈0 → swallowed by a much larger prediction.

## Gotchas
- Drift breaks the GT projection (`match_labels_to_vtx` assumes shared frame). On jump-drift scenes, `evaluation.skipped_objects` and some low IoUs are **projection noise**, not fusion error. Use `show <obj_id>` to see the displacement. A clean (no-drift) scene isolates the fusion effect.
- `run.merges_applied` (real merges) can exceed `evaluation.merges_scored` (GT-projected pairs only): the gap is `merges_skipped`, attributable per object in `skipped_objects` (one universe is the full run, the other is scored pairs).
