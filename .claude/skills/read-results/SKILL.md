---
name: read-results
description: Read, parse and compare OVO experiment results. Use after running an experiment to evaluate if a task implementation actually improves metrics vs a baseline.
user-invocable: true
disable-model-invocation: true
allowed-tools:
  - Read
  - Bash
---

# read-results — Compare OVO experiment results

Arguments: `$ARGUMENTS`

Expected arguments:
```
<new_experiment_id> --baseline <baseline_experiment_id> [--goal <description>]
```

- `<new_experiment_id>`: Experiment_ID of the new run (e.g. `20260407_GT_CLIP_task17-cooc-graph`)
- `--baseline <id>`: Experiment_ID to compare against (e.g. `20260305_GT_CLIP_ComparativaPaper`)
- `--goal <description>`: Optional. What improvement this task aimed to achieve (e.g. "reduce over-merging while maintaining mIoU"). Used to frame the conclusion.

If baseline is not provided, list available experiments and ask.

---

## Working directory

By default, results live in the **main repo**: `data/output/`.

If `--worktree <name>` is specified, the new experiment's output is in:
`.claude/worktrees/<name>/data/output/`

The baseline is always read from the main repo's `data/output/` (baselines are pre-existing runs from the main repo). Load them from separate paths if needed.

---

## Step 1 — Load both experiments

Run the following Python snippet. Adjust `new_output_dir` if the new experiment ran in a worktree:

```python
import sys
sys.path.insert(0, '.')
from pathlib import Path
from ovo.utils.results_utils import load_experiments, filter_experiments

main_output = Path('data/output')
# If worktree: new_output = Path('.claude/worktrees/<name>/data/output')
new_output = main_output  # change if worktree

new_id = '<new_experiment_id>'
baseline_id = '<baseline_experiment_id>'

df_new, _ = load_experiments(new_output)
df_base, _ = load_experiments(main_output)

new_row = filter_experiments(df_new, experiment_ids=[new_id])
baseline_row = filter_experiments(df_base, experiment_ids=[baseline_id])

cols = ['Experiment_ID', 'mIoU', 'mAcc', 'Head_mIoU', 'Common_mIoU', 'Tail_mIoU', 'Num_Instances']
print("=== BASELINE ===")
print(baseline_row[cols].to_string(index=False))
print("\n=== NEW ===")
print(new_row[cols].to_string(index=False))

# Deltas
for col in ['mIoU', 'mAcc', 'Num_Instances']:
    delta = new_row[col].values[0] - baseline_row[col].values[0]
    print(f"  Δ {col}: {delta:+.4f}")
```

Execute with: `conda run -n ovo2 python -c "<snippet above>"`

---

## Step 2 — Per-scene breakdown (if needed)

If aggregate results are ambiguous, load per-scene stats:

```python
from ovo.utils.results_utils import load_scene_results, filter_scene_results

df_scene, _ = load_scene_results(output_dir)
new_scenes = filter_scene_results(df_scene, experiment_ids=[new_id] )  # note: filter_scene_results uses same kwargs pattern
# or filter manually: df_scene[df_scene['Experiment_ID'] == new_id]
```

---

## Step 3 — Instance count check

`Num_Instances` in the DataFrame is parsed from `instance_pred/<scene>.txt` (one line = one predicted instance). Compare:

- If the task aimed to **reduce over-merging**: expect fewer instances, mIoU stable or improved.
- If the task aimed to **improve recall**: expect more instances, higher mIoU.
- If the task aimed to **improve classification**: instance count similar, mIoU improved.

---

## Step 4 — Report

Write a structured verdict:

```
## Verification results

| Metric         | Baseline | New    | Delta   |
|----------------|----------|--------|---------|
| mIoU           | X.XXX    | X.XXX  | +/-X.XX |
| mAcc           | X.XXX    | X.XXX  | +/-X.XX |
| Num_Instances  | NNN      | NNN    | +/-NNN  |
| Head_mIoU      | X.XXX    | X.XXX  | +/-X.XX |
| Common_mIoU    | X.XXX    | X.XXX  | +/-X.XX |
| Tail_mIoU      | X.XXX    | X.XXX  | +/-X.XX |

**Goal**: <what the task aimed to improve>
**Verdict**: PASS / FAIL / INCONCLUSIVE
**Reasoning**: <brief explanation linking metrics to goal>
```

Verdict criteria:
- **PASS**: metrics move in the expected direction for the stated goal, no significant regression elsewhere.
- **FAIL**: metrics move against the goal, or there is a significant regression (>2% mIoU drop).
- **INCONCLUSIVE**: deltas are within noise (< ±0.5% mIoU) or results are contradictory across scenes.

---

## Available baselines

Existing experiments in `data/output/Replica/` that are useful as baselines:

| Experiment_ID | Config | Notes |
|---|---|---|
| `20260305_GT_CLIP_ComparativaPaper` | GT + CLIP, no noise | Clean baseline |
| `20260305_GT_PE-Core_ComparativaPaper` | GT + PE-Core, no noise | PE baseline |
| `20260305_GT_SAM3_ComparativaPaper` | GT + SAM3, no noise | SAM3 baseline |
| `20260310_GTNoise-T0p001-R0p01_CLIP_ComparativaPaper` | GT + CLIP + light noise | Noise baseline |
| `20260310_GTNoise-T0p05-R0p01_CLIP_ComparativaPaper` | GT + CLIP + heavy noise | Stress baseline |

Run `ls data/output/Replica/` for the full list.
