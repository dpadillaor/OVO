---
name: contest-eval
description: Evaluate contest merge/split decision QUALITY against ground truth, query the discriminator live, and study the impact of each gate. Use when tuning/simplifying the contest discriminator, checking whether a threshold change helps, or grading how good a run's decisions are vs GT.
user-invocable: true
disable-model-invocation: false
allowed-tools:
  - Read
  - Bash
  - Edit
---

# contest-eval — Grade contest decisions vs GT + study gates

Unified module `studies/contest_metrics/` (run from repo root, env `ovo`). Three domains
under one CLI: **`eval`** (grade decisions vs GT), **`query`** (interrogate the discriminator
live), **`viz`** (Tier2 telemetry figures). Sits on `pre_fusion.ckpt` (the substrate the
contest actually saw) via `ContestProbe`, which recomputes verdicts under ANY config →
**counterfactual: change a gate and re-grade WITHOUT re-running the experiment**.

```bash
conda run -n ovo python -m studies.contest_metrics <domain> ...
```

## eval — grade decisions against ground truth (the main tool for tuning)

Projects GT instance labels onto every map point (KDTree, `eval/gt.py`) and grades each
decision with one point-set primitive that unifies merge and split (`eval/grade.py`):

> A decision reassigns a point-set `S` from defender `D` to challenger `C`.
> `belongs = g(S) == g(C)` where `g(X)` = dominant GT label over `X`. `moved` = MERGE or SPLIT.
> **moved&belongs=TP · moved&¬belongs=FP(over-merge/theft) · ¬moved&belongs=FN · ¬moved&¬belongs=TN.**
> MERGE: `S` = whole defender. SPLIT: `S` = `split_points`. No GT measurable → SKIP.

```bash
# grade the tuning scenes at the current default (firm_tau=0.30)
python -m studies.contest_metrics eval --exp <id> --scenes tuning
# counterfactual: grade a different gate value WITHOUT re-running the experiment
python -m studies.contest_metrics eval --exp <id> --scenes tuning --set firm_tau=0.4 --set contest_split_mode=all
# one scene, or held-out set, or a comma list; JSON for machine use
python -m studies.contest_metrics eval --exp <id> --scene office4 --json
python -m studies.contest_metrics eval --exp <id> --scenes held
```

- `--scenes` groups: **`tuning`** = office0/office3/office4/room1 · **`held`** = office1/office2/room0/room2 · **`all`** = 8. Or a comma list. (Tuning/held split: refine the discriminator on `tuning`, prove generalization on `held` — never tune on held.)
- `--set K=V` overrides any contest threshold (`firm_tau`, `min_grabs`, `high`, `low`, `min_mass`, `max_seam_angle`, `contest_split_mode=all`, …). Repeatable. This is the counterfactual knob.

Output (per scene + pooled):
- **Confusion** TP/FP/FN/TN/SKIP + precision/recall/f1 (positive = reassign).
- **Per-gate table** (by `reason` collapsed to a canonical gate): which gate produces the FP/wrong decisions. **This is what you read to decide which gate to cut/fix.**
- **Missed transfers**: firm chunks whose GT-dominant == challenger (≠ defender) that were NOT moved — the recall blind spot the winner-take-all verdict level can't show. High count = contest too conservative on splits (correlates with flat AP_50).

Reading it: a gate with FP≈TP is a coin flip → cut or redesign. Many missed transfers = the split machinery is too timid. Grade the SAME exp under two `--set` values and diff the per-gate confusion to see a gate's exact impact.

**Requires** an `--exp` whose `pre_fusion.ckpt` exists (a `raw-baseline` generator or any run; check with `query resolve`). GT lives at `data/input/Datasets/Replica/{scene}_mesh.ply` + `instance_gt/{scene}.txt`. Reliable on clean (no-drift) runs; GT projection degrades under drift.

## query — interrogate the discriminator live (why a decision, what the features are)

```bash
python -m studies.contest_metrics query pair   --exp <id> --scene office4 27 2   # what classify decides A↔B, why
python -m studies.contest_metrics query branch --exp <id> --scene office4 --decision SPLIT
python -m studies.contest_metrics query list   --exp <id> --scene office4 --by persistence -n 20
python -m studies.contest_metrics query point  --exp <id> --scene office4 123456 # raw store counters (P4/loyalty)
python -m studies.contest_metrics query resolve --exp <id> --scene office4        # which ckpt resolved + exists
python -m studies.contest_metrics query history --exp <id> --scene office4 27 2   # what it DECIDED over the run (verdicts.csv)
```
All accept `--set K=V` (same counterfactual overrides) and `--json`. Use `query pair` to
investigate a specific FP the `eval` per-gate table flagged.

## viz — Tier2 telemetry figures (per-KF timing/signals)

```bash
python -m studies.contest_metrics viz scene --exp <id> --scene office4 [--derivative]
python -m studies.contest_metrics viz exp   --exp <id>
```

## tui — interactive query (needs `textual`, installed in env `ovo2`)

```bash
python -m studies.contest_metrics tui --exp <id> --scene office4   # `set k=v` re-tunes without reload
```

## Firmness gate (the atom every feature is built on)

A point counts for a pair (firm) iff the challenger grabbed it **≥ `min_grabs` (5) KFs AND
`grabs/claims` ≥ `firm_tau` (0.30)** — evidence floor AND commitment floor (persistence at the
point). `firm_tau=0.30` is the tuned default (fixes room1/room2 over-merge regressions,
+7.9% global AP_agnostic vs raw). `firm_tau=0` → pure count (legacy). Lives in
`ovo/entities/contest/aggregator.py::_is_firm`. Renamed from `min_count`→`min_grabs`.

## Typical tuning loop

1. `eval --exp <raw-baseline> --scenes tuning` → read the per-gate confusion + missed count.
2. Spot the worst gate (FP≈TP) → `query pair` a couple of its FP defenders to see the features.
3. Counterfactual: `eval --set <threshold>=<new>` → does the FP drop without killing TP?
4. If a code change is needed, edit the discriminator, re-run the replay experiment, re-`eval`.
5. Confirm on `--scenes held` that the improvement generalizes.
