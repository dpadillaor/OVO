---
name: contest-eval
description: Interpret and cross-reference the contest Tier2 telemetry artifacts (logger/contest/*.log per-KF signals, their figures under fusion/contest/figures) produced by studies/contest_metrics, plus the raw contest.json counters. Use when reading how a run reacted to SLAM drift, spotting when/where OVO started mixing instances, or interpreting the per-KF n_matched/n_pre_assign/n_orphans/n_births/n_robos signals and their derivatives.
user-invocable: true
disable-model-invocation: false
allowed-tools:
  - Read
  - Bash
---

# contest-eval — Interpret & cross-reference contest Tier2 telemetry

The module `studies/contest_metrics/` turns the contest's **per-KF Tier2
telemetry** (`logger/contest/*.log`) into per-signal figures. This skill reads
those signals + the raw per-point counters in `contest.json` to answer *"how did
this run react to drift, and where did OVO start mixing instances?"*.

> Sibling skills: **contest-data** = the merge/split *verdicts* + raw dispute
> store (`contest_verdicts.csv`, `contest.json`). **fusion-eval** = fusion
> decisions graded vs GT. This skill = the **per-KF telemetry** (the hot-path
> health signals), not graded against GT — it is a *behaviour* view, not a
> *quality* view.

## The CLI — one figure per signal

Any env with matplotlib/numpy/pyyaml (the `ovo2` env works):

```bash
# one scene -> figures into <exp>/<scene>/fusion/contest/figures/
/home/padidavid/anaconda3/envs/ovo2/bin/python -m studies.contest_metrics.viz scene \
  --exp <EXP_ID> --scene office0 [--derivative] [--ext svg|png|pdf] [--no-derived]
# whole run (every scene with contest logs):
/home/padidavid/anaconda3/envs/ovo2/bin/python -m studies.contest_metrics.viz exp \
  --exp <EXP_ID> [--derivative]
```
Run from repo root (`studies` must be importable). `--exp` accepts a full path,
a short ID, or a unique prefix (globbed under `data/output/Replica/`). Jump
keyframes are read from `<scene>/config.yaml` (`noise.jumps`) and drawn as dashed
vlines on every figure — the drift is marked for you.

`--derivative` adds a two-panel `contest_<signal>_deriv.svg` (signal + Δ/KF). The
discrete derivative is the sharper drift detector (see below).

## The artifacts

| Path | Grain | What it is |
|---|---|---|
| `<scene>/logger/contest/*.log` | per **KF** | Tier2 telemetry, one number per line per keyframe |
| `<scene>/fusion/contest/figures/contest_*.svg` | per **KF** | the figures this module writes |
| `<scene>/fusion/contest/contest.json` | per **point** | raw cumulative counters (`grabs`/`claims`/`sightings`) |
| `<scene>/logger/frame_id.log` | per **KF** | KF idx → source frame id (maps `frame_id` jumps to a KF) |

**Preconditions** (else there is nothing to plot):
- `semantic.log: true` in the run config, AND the run was produced with the
  Tier2-telemetry code (routes to `logger/contest/`). Older runs have no
  `logger/contest/` dir and no `n_*` per-KF logs — re-run to generate them.
- `contest.json` is written regardless of `contest_fusion` (hot path always
  records). `contest_verdicts.csv` needs `contest_fusion != off` (see contest-data).

## The 6 Tier2 signals (per KF)

Each is a leaf of the partition of *matched* points (see `docs/refactor_contest_signals.md §5`).

| Signal | Counts (this KF) | Reads as |
|---|---|---|
| `n_matched` | points that reprojected well (frustum + reprojection ok) | how much of the map is in view |
| `n_pre_assign` | of matched, those already owning an instance (`ins > -1`) | **map maturity** — and the **drift detector** |
| `n_used` | SAM masks that assigned ≥1 point | how many masks actually landed |
| `n_orphans` | matched points under **no** used mask (`n_matched − n_covered`) | **SAM coverage gap** |
| `n_births` | new instances created | growth / re-exploration |
| `n_robos` | grab events (contested points, grabber ≠ owner) | **dispute intensity** |

Derived (written unless `--no-derived`):

| Derived | Formula | Reads as |
|---|---|---|
| `robo_rate` | `n_robos / n_pre_assign` | fraction of owned points being stolen |
| `orphan_rate` | `n_orphans / n_matched` | fraction of visible points SAM missed |

> These per-KF **counts** are the flow; `contest.json`'s `grabs`/`claims`/
> `sightings` are the per-point **cumulative stock** (Σ over KFs). `n_robos`
> accumulates into `grabs`, `n_pre_assign`-under-mask into `claims`, `n_matched`
> into `sightings`. Invariant on the stock: `sightings ⊇ claims ⊇ Σ grabs`.

## Interpreting — the drift signature

Validated on office0 with a single rotation jump (std 40°, `max_deg 150`) at kf 78,
both **yaw** and **roll**:

- **`n_pre_assign` derivative is the cleanest drift detector.** Its Δ/KF collapses
  to a sharp **negative** spike ~**−16× to −19×** the MAD noise band, at **kf 81**
  (~3 KF after the jump). The raw signal only dips modestly and is easy to miss;
  the derivative makes it unambiguous. Use `--derivative` and read the bottom panel.
- **The sign disambiguates drift from loop closure.** A sudden **negative** Δ =
  drift (OVO abruptly stops recognizing what it had → mixing starts). A large
  **positive** Δ (e.g. ~kf 175 in these runs) = rapid recovery / re-densification,
  the loop-closure correction — *not* a problem. Magnitude alone can't tell them
  apart; the signed derivative can.
- **Detection lag ≈ 3 KF.** The collapse manifests at kf 81 for a jump at kf 78 —
  by the time it fires, a few KF of mixing already happened. This is the ceiling
  on "act in time".
- **Cross-axis, deterministic.** yaw and roll give the same spike at the same KF
  (roll slightly stronger, −224k vs −206k). Pre-jump Δ values are *identical*
  across axes (same seed/trajectory before kf 78) — a good "is this real or noise"
  check: a real drift signature diverges only *after* the jump KF.

Other signals at the jump:
- **`orphan_rate` peaks** (points still visible but SAM masks land elsewhere → matched-but-unmasked). Reacts, but it is **noisy across the whole run** — on its own it does not separate the jump from baseline. Corroborating, not primary.
- **`n_robos` collapses for one KF** right at the jump (almost nothing lands on owned points under a mask), then recovers.
- **`n_births` spikes** a few KF later (the reoriented view spawns new instances) — the mixing/duplication starting.

## Interpreting — steady-state (no drift)

- **`orphan_rate` high** = SAM under-segmenting the visible scene (coverage gap); persistent high orphan_rate means many points are seen but never claimed.
- **`robo_rate` high** = lots of contested ownership — instances fighting over points; sustained high rate flags unstable boundaries (candidates the contest/fusion will act on).
- **`n_used` low vs `n_births` high** = the view keeps producing new masks that don't align with existing instances → fragmentation.
- **`n_pre_assign` rising smoothly** = healthy map maturation (more of the view is already owned).

## contest.json — the per-point stock (cross-ref)

```bash
D=data/output/Replica/<EXP_ID>/<scene>/fusion/contest
python3 -c "import json; d=json.load(open('$D/contest.json'));
print({k: len(d[k]) for k in d})"   # {'grabs':.., 'claims':.., 'sightings':..}
```
Per point: `persistence(grabber) = grabs[point][grabber] / claims[point]` ∈ [0,1]
(fraction of a point's claims that were steals by that grabber — the temporal
stability of a dispute). Loyalty of a point = `claims − Σ grabs` (times it fell
under its own owner's mask). Analyze verdicts/disputes with the **contest-data**
skill; use this skill's telemetry to see *when* the disputes happened.

## Gotchas

- **No `logger/contest/` → old run.** The Tier2 telemetry is recent; a run without
  it must be re-launched with `semantic.log: true` and current code. `config.yaml`
  saying `log: true` is necessary but not sufficient — the code version matters.
- **Detection lag is real** (~3 KF). Don't trust the exact jump KF from the signal;
  read the marked vline (from config) as ground truth and the spike as the *effect*.
- **`orphan_rate` alone is not a drift detector** — too noisy. Use `n_pre_assign`'s
  derivative; corroborate with orphan_rate/n_robos.
- **Loop closure looks like an event too.** A big *positive* `n_pre_assign` Δ is a
  correction, not drift. Cross-check with `logger/t_loop_closure_refusion.log` and
  the frame_id of the trajectory revisit before calling it drift.
- **Per-KF counts, not per-point.** These logs are flow (Σ over the KF), not the
  cumulative store. For per-point history read `contest.json`.
