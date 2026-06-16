"""Phase 2 analysis — rank chains, Wilcoxon vs baseline, pick winner(s).

Baseline reference chain = [centroid, cos_sim, overlap] (label abl-centroid-cossim-overlap).
Statistical unit = scene (n=8). Paired Wilcoxon signed-rank per scene for each
chain vs baseline, per condition. Winner rule: not significantly worse than the
best in any condition (on mIoU and AP_agnostic).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/padidavid/repos/OVO")
MASTER = ROOT / "study/runs_master.parquet"
FIGS = ROOT / "study/figs"
BASELINE_LABEL = "abl-centroid-cossim-overlap"


def per_scene_frame(row):
    ps = row["per_scene"]
    if isinstance(ps, str):
        ps = json.loads(ps)
    return pd.DataFrame(ps)


def main():
    df = pd.read_parquet(MASTER)
    df = df[df.label.str.startswith("abl-") & (df.n_scenes == 8)].copy()
    print(f"Phase 2 rows: {len(df)} (expect 24)")

    # per-scene dict: {(label,cond): {scene: {metric}}}
    bench = {}
    for _, r in df.iterrows():
        psf = per_scene_frame(r)
        bench[(r.label, r.condition)] = psf.set_index("Scene")

    results = {"by_condition": {}, "winner": None}
    for cond in ["light", "aggressive"]:
        sub = df[df.condition == cond].copy()
        base_key = (BASELINE_LABEL, cond)
        base_ps = bench.get(base_key)
        rows = []
        for _, r in sub.iterrows():
            ps = bench[(r.label, cond)]
            entry = {"label": r.label, "chain": r.fusion_criteria,
                     "mIoU": float(r.mIoU), "AP_agnostic": float(r.AP_agnostic),
                     "mAcc": float(r.mAcc), "Num_Instances": float(r.Num_Instances)}
            if base_ps is not None and r.label != BASELINE_LABEL:
                common = base_ps.index.intersection(ps.index)
                for metric in ["mIoU", "AP_agnostic"]:
                    a = ps.loc[common, metric].astype(float).values
                    b = base_ps.loc[common, metric].astype(float).values
                    d = a - b
                    if np.allclose(d, 0):
                        entry[f"wilcoxon_p_{metric}"] = 1.0
                    else:
                        try:
                            entry[f"wilcoxon_p_{metric}"] = float(
                                st.wilcoxon(a, b, zero_method="wilcox").pvalue)
                        except ValueError:
                            entry[f"wilcoxon_p_{metric}"] = float("nan")
                    entry[f"delta_{metric}"] = float(np.mean(d))
            rows.append(entry)
        rows.sort(key=lambda e: (e["mIoU"] + e["AP_agnostic"]), reverse=True)
        results["by_condition"][cond] = rows
        print(f"\n=== {cond} (ranked by mIoU+AP_agn) ===")
        for e in rows:
            dm = e.get("delta_mIoU"); pv = e.get("wilcoxon_p_mIoU")
            print(f"  {e['label']:38s} mIoU={e['mIoU']:.4f} AP_agn={e['AP_agnostic']:.4f} "
                  f"Ninst={e['Num_Instances']:.0f}"
                  + (f"  dmIoU={dm:+.4f} p={pv:.3f}" if dm is not None else "  (baseline)"))

    # ---- winner selection: best mean(mIoU,AP_agn) averaged across conditions,
    # not significantly worse than per-condition best on either metric ----
    labels = sorted(df.label.unique())
    agg = {}
    for lab in labels:
        scores = []
        for cond in ["light", "aggressive"]:
            row = next(e for e in results["by_condition"][cond] if e["label"] == lab)
            scores.append((row["mIoU"], row["AP_agnostic"]))
        agg[lab] = {"mean_mIoU": float(np.mean([s[0] for s in scores])),
                    "mean_AP_agnostic": float(np.mean([s[1] for s in scores])),
                    "combined": float(np.mean([s[0] + s[1] for s in scores]))}
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["combined"], reverse=True)
    results["aggregate_ranking"] = [{"label": k, **v} for k, v in ranked]
    results["winner"] = ranked[0][0]
    # second candidate if close
    results["runner_up"] = ranked[1][0] if len(ranked) > 1 else None
    print(f"\nWINNER (mean combined): {results['winner']}")
    print(f"RUNNER-UP: {results['runner_up']}")

    # ---- cooccurrence damage quantification ----
    cooc_damage = {}
    for cond in ["light", "aggressive"]:
        rows = {e["label"]: e for e in results["by_condition"][cond]}
        dmi, dap = [], []
        for lab in labels:
            if lab.startswith("abl-cooccurrence-"):
                off = lab.replace("abl-cooccurrence-", "abl-")
                if off in rows:
                    dmi.append(rows[lab]["mIoU"] - rows[off]["mIoU"])
                    dap.append(rows[lab]["AP_agnostic"] - rows[off]["AP_agnostic"])
        if dmi:
            base_ap = rows["abl-centroid-cossim-overlap"]["AP_agnostic"]
            cooc_damage[cond] = {
                "mean_delta_mIoU_on_vs_off": float(np.mean(dmi)),
                "mean_delta_AP_agnostic_on_vs_off": float(np.mean(dap)),
                "mean_pct_AP_agnostic_gain": float(np.mean(dap) / base_ap * 100),
                "n_pairs": len(dmi)}
    results["cooccurrence_damage"] = cooc_damage
    print(f"\nCooccurrence on-vs-off mean dmIoU: {cooc_damage}")

    # ---- overlap vs overlap_old ----
    ov_cmp = {}
    for cond in ["light", "aggressive"]:
        rows = {e["label"]: e for e in results["by_condition"][cond]}
        dmi, dap = [], []
        for lab in labels:
            if lab.endswith("-overlap"):
                old = lab[:-len("-overlap")] + "-overlapold"
                if old in rows:
                    dmi.append(rows[lab]["mIoU"] - rows[old]["mIoU"])
                    dap.append(rows[lab]["AP_agnostic"] - rows[old]["AP_agnostic"])
        if dmi:
            ov_cmp[cond] = {"mean_delta_mIoU_new_vs_old": float(np.mean(dmi)),
                            "mean_delta_AP_agnostic_new_vs_old": float(np.mean(dap)),
                            "n_pairs": len(dmi)}
    results["overlap_new_vs_old"] = ov_cmp
    print(f"overlap(new) vs overlap_old mean dmIoU: {ov_cmp}")

    # ---- delta heatmap (chain x condition) vs baseline ----
    fig, ax = plt.subplots(figsize=(8, 7))
    mat, ylabels = [], []
    for lab in labels:
        if lab == BASELINE_LABEL:
            continue
        ylabels.append(lab.replace("abl-", ""))
        mat.append([next(e for e in results["by_condition"][c]
                         if e["label"] == lab).get("delta_mIoU", 0.0)
                    for c in ["light", "aggressive"]])
    mat = np.array(mat)
    im = ax.imshow(mat, cmap="RdBu", vmin=-abs(mat).max(), vmax=abs(mat).max(), aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["light", "aggressive"])
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title("Δ mIoU vs baseline [centroid,cos_sim,overlap]")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(FIGS / "phase2_delta_heatmap.png", dpi=110)
    plt.close(fig)

    (ROOT / "study/phase2_results.json").write_text(json.dumps(results, indent=2))
    print("\nWrote study/phase2_results.json + figs/phase2_delta_heatmap.png")


if __name__ == "__main__":
    main()
