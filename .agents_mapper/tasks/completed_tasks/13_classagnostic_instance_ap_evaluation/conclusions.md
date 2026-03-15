# Conclusions — Class-Agnostic Instance AP

Experiment: `data/output/Replica/20260305_GT_PE-Core_ComparativaPaper`
Scene: `office0`

---

## 1. Alignment: OK

Bounding boxes of `pcd_pred` (OVO map, ~2M points) and `pcd_gt` (Replica mesh, ~589k vertices) are essentially identical:

```
Predicted: min=[-2.009, -3.155, -1.152]  max=[2.397, 1.804, 1.782]
GT:        min=[-2.006, -3.154, -1.169]  max=[2.394, 1.856, 1.823]
```

Both clouds live in the same coordinate frame. `match_labels_to_vtx` is geometrically correct.

## 2. Eval pipeline: OK

GT-oracle sanity check (evaluating with the GT masks directly) scores **mAP = 1.000**. The `ins_eval_utils.evaluate` pipeline and the `instance_pred/office0.txt` loading logic are correct.

## 3. OVO class-agnostic mAP: 9%

This is genuinely low segmentation quality, not an artifact of the eval code.

## 4. Visual inspection (Open3D viewer)

The predicted map is **heavily over-segmented**. Each physical object is split into many small predicted instances, each with a low IoU against the single GT instance it corresponds to — causing most predictions to fall below the AP thresholds (0.25, 0.5, 0.75).

However, the shapes of individual large objects (sofa, chairs, table) are **well reconstructed geometrically** — the boundary of the object is correctly identified, it's just fragmented into multiple instances.

---

## 5. IoU distribution (office0, 20260305_GT_PE-Core_ComparativaPaper)

```
GT instances:   68
Pred instances: 145
Best-IoU distribution:
  mean=0.164  median=0.035  max=0.931
  IoU >= 0.10: 45/145  (31%)
  IoU >= 0.25: 29/145  (20%)
  IoU >= 0.50: 20/145  (14%)
  IoU >= 0.75: 13/145   (9%)  ← matches mAP = 9%
```

**Interpretation:**
- 100/145 predictions have IoU < 0.10 → they don't correspond to any GT instance
- 13 predictions match well (IoU > 0.75) → these are the recognizable objects (sofa, chairs, table)
- The problem is not just over-segmentation: most predictions are spurious/phantom instances

## 6. Root cause hypothesis: no instance fusion + wall fragments

The experiment `20260305_GT_PE-Core_ComparativaPaper` does **not use instance fusion/merging**. This means:
- Every small region OVO detects becomes a separate instance
- Large surfaces (walls, floor, ceiling) get split into many tiny instances
- These fragments have very low IoU with any GT instance → 100 phantom FPs → precision collapses → low AP

The GT has only 68 instances (recognizable objects). OVO produces 145, many of which are wall/floor/ceiling fragments that don't correspond to any GT object.

**Consequence:** this experiment is not the right one to evaluate instance AP fairly. A run with proper instance fusion/merging is needed to get a meaningful AP score.

## Open questions

- Does running with instance fusion reduce the number of phantom instances?
- What experiment config has fusion enabled? Compare AP with vs without fusion.
- Are the 13 well-matched instances (IoU > 0.75) consistently the same object types across scenes?
