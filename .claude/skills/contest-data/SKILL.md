---
name: contest-data
description: Extract and analyze contest mechanism outputs (contest_verdicts.csv and contest.json) from an OVO experiment. Use when inspecting merge/split decisions, comparing two runs, or digging into raw per-point dispute data.
user-invocable: true
disable-model-invocation: false
allowed-tools:
  - Read
  - Bash
---

# contest-data — Analyze contest outputs (CSV verdicts + JSON store)

Two stdlib-only tools live in `Study_seg/`. No deps, run with plain `python3`.
Each contest experiment writes both files per scene under
`data/output/Replica/<EXP_ID>/<scene>/fusion/` (older runs: the scene root — the
tools take an explicit path, so point them at wherever the file actually is):

| File | What it is | Tool |
|---|---|---|
| `contest_verdicts.csv` | one row per decided defender: decision + features | `Study_seg/contest_csv.py` |
| `contest.json` | raw counters: `{"grabs": {point_id: {grabber: nº_KFs}}, "claims": {point_id: n}, "sightings": {point_id: n}}` | `Study_seg/contest_json.py` |
| `ovo_map.ckpt` (`map_params.normals`) | per-point surface normals | `Study_seg/seam_normals_viz.py` |

## When to use which

- **"why did pair X get decision Y"** → `contest_csv.py pair`
- **"what changed between two runs"** → `contest_csv.py compare`
- **"distribution of decisions / feature stats"** → `contest_csv.py summary`
- **"which instance is hoarding points / raw dispute counts"** → `contest_json.py`
- **"how many KFs saw point P under instance W"** → `contest_json.py point`
- **"is this dominance split a real fragment or a touching object"** → `seam_normals_viz.py` (geometric normal-turn signal)

## contest_csv.py — the verdicts (decisions + features)

```bash
python3 Study_seg/contest_csv.py summary  CSV
python3 Study_seg/contest_csv.py pair     CSV 63 151
python3 Study_seg/contest_csv.py filter   CSV --decision SPLIT --sort containment --top 10
python3 Study_seg/contest_csv.py compare  BASELINE_CSV NEW_CSV
```

- `summary` — counts per decision, reason families, per-decision feature medians/min/max.
- `pair ID...` — every row touching those instances (as defender or challenger). The go-to for "what happened to instance N".
- `filter --decision {MERGE_CONTAINMENT|SPLIT|NO_ACTION|DEFER_TO_FUSION}` — sort by any feature (`containment`, `reverse_containment`, `firm_points`, `total_grabs`, `persistence`, `focus`), `--top N`, `--asc`.
- `compare A B` — decision counts delta + the exact list of `(defender->challenger)` whose verdict changed. **Use this to validate a threshold change**: run baseline vs new and read which verdicts flipped.

## contest_json.py — the raw store (per-point dispute counts)

```bash
python3 Study_seg/contest_json.py summary  JSON --top 15
python3 Study_seg/contest_json.py winner   JSON 63 151
python3 Study_seg/contest_json.py point    JSON 172656
python3 Study_seg/contest_json.py hot      JSON --by winners --top 15
python3 Study_seg/contest_json.py pairmass JSON 151 63
```

- `summary` — total contested points, distinct winners, how many points have 1 / 2 / 3+ claimants (≥2 = real dispute), KF-count distribution, top winners by points claimed.
- `winner ID...` — points claimed by an instance + KF/point stats. Spot blobs hoarding points.
- `point PID...` — which instances saw a given point, in how many KFs.
- `hot --by {winners|count}` — most contested points (most claimants) or most-observed.

## seam_normals_viz.py — geometric normal-turn signal (needs normals)

Separates a **real fragment** (a mis-segmented piece of W → transfer it) from an
**object in contact** (a distinct object touching W → do NOT transfer) for the
dominance band, where 2D-mask features and the CLIP descriptor can't tell them apart.

For each dominance pair `A→W`: takes the chunk (A's disputed points), finds each
chunk point's K-neighbour patch in W (kd-tree), averages W's normals (sign-aligned),
and measures the `|cos|` angle between the chunk's normal and W's patch normal.

- **low angle (~<10°)** = surface continuous across the seam = same object → transfer.
- **high angle (~>20°)** = a kink/groove = two objects touching → keep split / no-action.

Validated on office3: BUENO mesa `89→103` ≈ 8°, MALO cojín `18→26` ≈ 62°, objects
on table `131/134→103` ≈ 26–36°. Clean gap between ~10° and ~22°; threshold ~15°.

```bash
python Study_seg/seam_normals_viz.py [RUN_OFFICE_DIR] [--kw 15] [--dmax 0.05] [--out path.rrd]
# needs the ovo2 env: conda run -n ovo2 python Study_seg/seam_normals_viz.py ...
```

- **Requires** a run with normals (`map_params.normals`) AND intact owners — i.e.
  `contest_split_mode: partial` (dominance is NOT applied, so the chunk still belongs
  to A). A split-applied run moves the chunk to W and chunks come out empty.
- No arg → auto-picks latest `*contest-normals-partial-pia*` run.
- Prints the per-pair angle table and writes a `.rrd` to inspect in Rerun: per pair,
  `A_loser`(grey) `W_winner`(blue) `seam_chunk_A`(red) `W_patch_usado`(yellow=the W
  points averaged) `normal_chunk`(orange) vs `normal_W_parche`(cyan). Parallel
  orange/cyan = continuous; diverging = kink. In Replica (GT depth) k=1 ≈ patch since
  there's no sensor noise; the patch matters with real (noisy) depth.

## Feature cheat-sheet (CSV)

| Feature | Meaning |
|---|---|
| `containment` | firm_points / \|defender\| — fraction of the defender inside the challenger (directional) |
| `reverse_containment` | same, other direction (challenger→defender). ≈0 = unidirectional |
| `firm_points` | nº of the defender's POINTS grabbed firmly by the challenger (grabs ≥ min_count) |
| `total_grabs` | Σ grabs over those points (raw weight, noise gate ≥ min_mass) |
| `persistence` | mean(grabs_challenger / claims) per point — temporal stability |
| `focus` | firm_points / defender's total disputed points — is the dispute concentrated on one rival |

## Typical workflow

1. Run experiment (skill `run-experiment`), note the new `EXP_ID`.
2. `contest_csv.py compare <baseline>/fusion/contest_verdicts.csv <new>/fusion/contest_verdicts.csv` → see what flipped.
3. For suspicious flips: `contest_csv.py pair <new>/fusion/contest_verdicts.csv <ids>` → read features + reason.
4. To ground a decision in raw data: `contest_json.py point <new>/fusion/contest.json <point_id>` or `winner <ins_id>`.
