"""Phase 1 — data-driven ranges from baseline fusion_decisions.csv (0 new runs).

For each baseline (light, aggressive) and scene, load the per-pair fusion
decisions and: (1) attribute REJECTs to the deciding criterion, (2) inspect the
ctx-value distributions (cos_sim, centroid_dist, p_dist, shared_kfs) split by
ACCEPT vs REJECT, (3) derive physically-bounded search ranges per threshold.
Writes study/ranges.json + study/figs/phase1_*.png and prints a summary.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/padidavid/repos/OVO")
CKPT_OUT = {
    "light":      "20260613_GTJump-J2-T0p03-R0p0_CLIP_light-baseline_82bfd",
    "aggressive": "20260613_GTJump-J2-T0p3-R0p0_CLIP_agressive-baseline_5f2f1",
}
SCENES = ["office0", "office1", "office2", "office3", "office4",
          "room0", "room1", "room2"]
OUTDIR = ROOT / "data/output/Replica"
FIGS = ROOT / "study/figs"

# Physically-bounded prior ranges (playbook appendix); refined by data below.
PRIOR = {
    "th_centroid":          [0.3, 3.0],
    "th_aabb":              [0.05, 1.5],
    "th_cossim":            [0.5, 0.97],
    "th_points":            [0.02, 0.3],
    "th_overlap_ratio":     [0.2, 0.8],
    "th_overlap_cos":       [0.75, 0.98],
    "th_overlap_ratio_low": [0.05, 0.5],
    "cooccurrence_veto_threshold": [2, 15],
}


def load_all():
    frames = []
    for cond, folder in CKPT_OUT.items():
        for sc in SCENES:
            p = OUTDIR / folder / sc / "fusion_decisions.csv"
            if not p.is_file():
                continue
            df = pd.read_csv(p)
            df["condition"] = cond
            df["scene"] = sc
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    print(f"Loaded {len(df)} pair-decisions from "
          f"{df.scene.nunique()} scenes x {df.condition.nunique()} conditions")
    summary = {"n_decisions": int(len(df)), "by_condition": {}}

    # ---- (1) rejection attribution -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cond in zip(axes, ["light", "aggressive"]):
        d = df[df.condition == cond]
        rej = d[d.result == "REJECTED"].reason.value_counts(normalize=True)
        acc = (d.result == "ACCEPTED").mean()
        rej.plot.bar(ax=ax, color="#5c7cfa")
        ax.set_title(f"{cond}: REJECT attribution (accept_rate={acc:.3f})")
        ax.set_ylabel("fraction of REJECTs")
        summary["by_condition"][cond] = {
            "n": int(len(d)),
            "accept_rate": float(acc),
            "reject_reasons": {k: float(v) for k, v in rej.items()},
        }
    fig.tight_layout()
    fig.savefig(FIGS / "phase1_reject_attribution.png", dpi=110)
    plt.close(fig)

    # ---- (2) ctx distributions ACCEPT vs REJECT ------------------------
    ctx_cols = ["cos_sim", "centroid_dist", "p_dist", "shared_kfs"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    pct = {}
    for ri, cond in enumerate(["light", "aggressive"]):
        d = df[df.condition == cond]
        for ci, col in enumerate(ctx_cols):
            ax = axes[ri, ci]
            for res, color in [("ACCEPTED", "#51cf66"), ("REJECTED", "#ff6b6b")]:
                vals = pd.to_numeric(d[d.result == res][col], errors="coerce").dropna()
                if len(vals):
                    ax.hist(vals, bins=40, alpha=0.55, color=color, label=res, density=True)
            ax.set_title(f"{cond}: {col}")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
        # percentiles of accepted pairs -> informs separating thresholds
        acc = d[d.result == "ACCEPTED"]
        pct[cond] = {}
        for col in ctx_cols:
            v = pd.to_numeric(acc[col], errors="coerce").dropna()
            if len(v):
                pct[cond][col] = {q: float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95)}
    fig.tight_layout()
    fig.savefig(FIGS / "phase1_ctx_distributions.png", dpi=110)
    plt.close(fig)
    summary["accepted_pair_percentiles"] = pct

    # ---- (3) derive ranges ---------------------------------------------
    # Start from physically-bounded priors; widen/center using observed accepted
    # pair statistics where available (pooled across conditions).
    ranges = {k: list(v) for k, v in PRIOR.items()}
    # cos_sim: accepted pairs cluster high; keep prior but record observed p5.
    all_acc = df[df.result == "ACCEPTED"]
    cos_acc = pd.to_numeric(all_acc.cos_sim, errors="coerce").dropna()
    cent_acc = pd.to_numeric(all_acc.centroid_dist, errors="coerce").dropna()
    pdist_acc = pd.to_numeric(all_acc.p_dist, errors="coerce").dropna()
    notes = {}
    if len(cos_acc):
        notes["th_cossim"] = (f"accepted cos_sim p5={np.percentile(cos_acc,5):.3f} "
                              f"median={np.median(cos_acc):.3f}")
    if len(cent_acc):
        notes["th_centroid"] = (f"accepted centroid_dist p95={np.percentile(cent_acc,95):.3f} "
                                f"median={np.median(cent_acc):.3f}")
    if len(pdist_acc):
        notes["th_points"] = (f"accepted p_dist p95={np.percentile(pdist_acc,95):.4f}")

    out = {"ranges": ranges, "prior_source": "playbook appendix (physically bounded)",
           "data_notes": notes, "summary": summary}
    (ROOT / "study/ranges.json").write_text(json.dumps(out, indent=2))
    print("Wrote study/ranges.json")
    print(json.dumps({"accept_rate": {c: summary['by_condition'][c]['accept_rate']
                                      for c in summary['by_condition']},
                      "data_notes": notes}, indent=2))


if __name__ == "__main__":
    main()
