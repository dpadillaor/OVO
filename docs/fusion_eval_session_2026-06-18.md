# Fusion evaluation tooling — session 2026-06-18

Work on `studies/fusion_metrics/`: a tool that grades a run's **real** fusion
merge/split decisions against GT. Branch `study/point-instance-assignment`.

## What the tool produces

Run: `cd studies/fusion_metrics && python -m scripts.eval_fusion_decisions
--exp_path data/output/Replica/<EXP> --scene office0` (env `ovo`). Needs a
`pre_fusion.ckpt` (saved only on jump-drift runs), mirrored under
`data/checkpoints/...`. Writes three artifacts next to `fusion_decisions.csv`:

| File | Grain | What |
|---|---|---|
| `fusion_decisions_eval.csv` | per pair | each merge/split decision + GT verdict (TP/FP/FN/TN) |
| `fusion_eval_summary.json` | per scene | aggregate verdicts + agnostic-AP impact |
| `fusion_instance_stats.csv` | per GT instance | which objects fusion helped/hurt |

Interactive debugger (GUI): `python -m scripts.debug_merge_decisions
--exp_path <run> --scene office0 --z_max 1.5`. Commands: `gts` (list GT
instances, decoded), `gt <id>` (matched green + spurious red shades), `show
<obj_id>` (raw OVO points over GT → see drift), `missing`.

Skill `fusion-eval` documents how to read and cross-reference the artifacts.

## Changes this session (commits, oldest first)

- **`84026ed`** — 3rd artifact `fusion_instance_stats.csv` + perf fix + debug commands + skill.
  - Per GT instance: `iou_pre/post`, `acc_pre/post`, `matched_obj_id_pre`, `n_spurious`, `spurious_obj_ids`, `status`.
  - **Perf:** `_iou_matrix` / `_intersection` int64→**float32** matmul (BLAS, ~50× faster, exact below 2²⁴). The debug `gt` command was blocking the REPL because int64 matmul over ~500k vertices took ~30-60s; now <1s.
  - **Debug:** load instances from `pre_fusion.ckpt` (not the final map); new `show/gt/gts/missing`; GT id accepts full id or unique instance suffix; decoded GT listing (object vs background).
  - **Viz:** `build_instance_overlay_cloud` (raw OVO points on GT to reveal drift); spurious fragments share one hue with per-fragment shades.
- **`0669d1a`** — `match_status` column + rename `matched_obj_id`→`matched_obj_id_pre`.
  - `match_status` = AP-match transition @0.5: `KEPT` / `LOST` / `GAINED` / `UNMATCHED` (greedy 1:1, consistent with agnostic AP). Separates the discrete match change from `status` (IoU direction). E.g. `UNMATCHED`+`improved` = "improved but not enough"; `LOST`+`worsened` = over-merge dropped it below 0.5.
  - `matched_obj_id_pre`: it is the **pre-fusion** match (post masks are merged groups without obj_ids); kept even when LOST since it is the join key to the merge that over-merged it. `None` when unmatched.
- **`aa59a85`** — restructure `fusion_eval_summary.json` into 4 blocks.
  - `run` (instances_pre/post, merges_applied), `evaluation` (pairs/merges scored vs skipped + `skipped_objects` per object {obj_id, pairs, merges}), `verdicts` (counts/rates/by_group), `agnostic_impact`.
  - Reconciles on its face: `merges_applied = merges_scored + merges_skipped`; `pairs_total = pairs_scored + pairs_skipped`. The old flat layout hid the 51-vs-50 merge gap.
  - `summarize()` is now pure (returns only `verdicts`); the orchestrator assembles `run`/`evaluation`.
- **`596ca6b`** — production-consistent agnostic AP (`objects` mode) alongside `all`.
  - `_gt_instance_masks(gt_ids, valid_classes)`: drop background-class GT targets.
  - `_void_fraction` + `_match_at` void forgiveness: an unmatched prediction that is mostly background (void fraction > IoU threshold) is **not** counted as FP. IoU stays **raw** — predictions are kept whole, **NOT carved** (carving inflates IoU and rewards sloppy object+background blobs; production discounts void in the FP accounting instead).
  - `summary.json` `agnostic_impact` now reports both `{all, objects}` side by side.

## Key findings / analysis (on `20260613_GTJump-J2-T0p03-R0p0_CLIP_light-baseline_82bfd`, office0)

- **Cross-data works.** Causal chain: `fusion_instance_stats` (which object worsened) → `matched_obj_id_pre` → `fusion_decisions_eval` (which merge, via i1/i2) → debug `gt`/`show` (see it in 3D). Same pre-fusion obj_id space joins them.
- **Concrete over-merges found:** panel(20)+switch(30) and sofa(7)+frag-of-other-sofa(4), both `ACCEPTED, same_object=False → FP`. One bad merge can worsen two instances. Cause: same-class objects, close (low `centroid_dist`) + semantically similar (high `cos_sim`) under drift.
- **AP impact ≈ flat** (`delta_ap50 -0.0007`): fusion consolidated 49 spurious fragments but a single over-merge cost 1 matched object (switch 79050 → LOST). AP is recall-bound, so cleaning spurious barely moves it.
- **51 vs 50 merges:** `run.merges_applied = 51`, `evaluation.merges_scored = 50`. The gap is obj 25 (didn't project to GT → 21 skipped pairs, 1 of them the `24↔25` ACCEPTED merge). Now explicit in `evaluation.skipped_objects`.
- **Two universes are internally consistent:** `n_accepted == columns removed == -(Δmatched + Δspurious)`; the single matched→missed flip (switch) matches `delta_matched50 = -1`.

## Open / unresolved

1. **Agnostic AP differs from production (`instance_ap_office0.txt`).** On the *same* input (`instance_pred`), ours `objects` = AP50 0.301 vs production 0.243 (we are ~0.05-0.06 higher across thresholds). Input is NOT the cause (our checkpoint reconstruction ≈ instance_pred result). Causes are algorithmic:
   - **Confidence / PR curve (the big one).** Production ranks predictions by their **score** and integrates a precision-recall curve; we assume **uniform** confidence → a single operating point (optimistic, ignores ranking). The score comes from `classify_instances` (`ovo/entities/ovo.py:855`) = **CLIP semantic confidence** of the class assignment (similarity of the instance descriptor to its best-matching class). Our fusion masks (from the checkpoint, by obj_id) carry no score — but the checkpoint **does** store per-instance descriptors, so the score is **recoverable** via `classify_instances` if we want exact parity.
   - Production's `num_ignore` also forgives small/group GT (we only forgive background void); and production drops predictions <100 pts (`min_region_size`); we do neither.
2. **Conceptual question raised (important):** production's "AP_agnostic" relabels the class to generic for *matching* but still **ranks by the CLIP class confidence**. So it is agnostic in matching, NOT in ranking — a geometrically perfect mask that CLIP fails to recognize is penalized. Our uniform AP is arguably **more** purely agnostic (geometry only). For a fusion study (which changes geometry, not recognition), uniform may be the *better* choice, not a bug to fix. **Decision pending:** keep uniform (more agnostic, isolates fusion's geometric effect) vs add CLIP-score ranking (exact production parity but less agnostic). The pre/post **delta** — what the study cares about — is largely independent of the absolute offset.
3. **PR-curve intuition** still not fully internalized by the user — worth revisiting. Short version: AP rewards a detector that puts its correct predictions at high confidence (top of the ranked list) and wrong ones at low confidence; ordering by score traces the precision/recall curve, AP = area under it. Uniform confidence collapses it to one point.
4. **Drift caveat:** projecting a drifted map to GT mixes fusion error with projection error. `skipped_objects` and some low IoUs on jump-drift scenes are projection noise, not fusion signal. A clean (no-drift) scene would isolate the fusion effect — not yet run.
5. **Housekeeping:** `studies/fusion_metrics/RenderOption_*.json` (Open3D GUI dump) left untracked — gitignore or delete.

## Files touched
- `core/fusion_agnostic_impact.py` — float32 matmul, `InstanceStat`+`MatchStatus`, `per_instance_stats`, `valid_classes` (GT filter + void forgiveness).
- `core/merge_decision_eval.py` — `Verdict` enum, `EvaluatedPair`, `evaluate_decision`, `summarize` (now pure verdicts block).
- `core/loaders.py` — `FusionDecision`, `load_fusion_decisions`, `parse_fusion_decision`, `load_pre_fusion_scene`, `load_pre_fusion_map`.
- `core/writers.py` — `write_instance_stats_csv`, selective-inline JSON formatter.
- `scripts/eval_fusion_decisions.py` — orchestrator: 3 artifacts, run/evaluation assembly, both AP modes.
- `scripts/debug_merge_decisions.py` — pre-fusion load, `show/gt/gts/missing`.
- `viz/debug_viz.py` — overlay cloud, GT inspection, spurious shades.
- `.claude/skills/fusion-eval/SKILL.md` — analysis guide.
