"""Analyze the ORB-SLAM external validation: A(baseline) vs B(mechanism) vs C(tuned).

Per-scene paired Wilcoxon (n<=8) of B and C against A on mIoU and AP_agnostic.
Tells us whether the simulated-jump-drift finding (cooccurrence -> AP_agnostic up)
transfers to real ORB-SLAM drift. Writes study/orbslam_results.json + fig.
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
MASTER = ROOT / "study/orbslam_master.parquet"
FIGS = ROOT / "study/figs"
A, B, C, D = "orb-A-baseline", "orb-B-mechanism", "orb-C-tuned", "orb-D-nocooc"


def ps(row):
    p = row["per_scene"]
    return pd.DataFrame(json.loads(p) if isinstance(p, str) else p).set_index("Scene")


def main():
    df = pd.read_parquet(MASTER)
    df = df[df.label.isin([A, B, C, D])].drop_duplicates("label", keep="last")
    by = {r.label: r for _, r in df.iterrows()}
    out = {"configs": {}, "comparisons": {}}
    for lab in [A, B, C, D]:
        if lab in by:
            r = by[lab]
            out["configs"][lab] = {"mIoU": float(r.mIoU), "AP_agnostic": float(r.AP_agnostic),
                                   "mAcc": float(r.mAcc), "Num_Instances": float(r.Num_Instances),
                                   "n_scenes": int(r.n_scenes)}
    if A not in by:
        print("No baseline (A) yet."); (ROOT/"study/orbslam_results.json").write_text(json.dumps(out, indent=2)); return
    def paired(ref, lab):
        r0, r1 = ps(by[ref]), ps(by[lab])
        common = r0.index.intersection(r1.index)
        cmp = {"n": int(len(common))}
        for metric in ["mIoU", "AP_agnostic"]:
            a = r1.loc[common, metric].astype(float).values
            b = r0.loc[common, metric].astype(float).values
            d = a - b
            cmp[f"delta_{metric}_mean"] = float(np.mean(d))
            bm = float(np.mean(b))
            cmp[f"pct_{metric}"] = float(np.mean(d) / bm * 100) if bm else None
            if np.allclose(d, 0):
                cmp[f"wilcoxon_p_{metric}"] = 1.0
            else:
                try:
                    cmp[f"wilcoxon_p_{metric}"] = float(st.wilcoxon(a, b).pvalue)
                except ValueError:
                    cmp[f"wilcoxon_p_{metric}"] = float("nan")
        return cmp

    for lab in [B, C, D]:
        if lab in by:
            out["comparisons"][f"{lab}_vs_A"] = paired(A, lab)
    # controlled isolation of cooccurrence: B (with cooc) vs D (same chain, no cooc)
    if B in by and D in by:
        out["comparisons"]["B_vs_D_cooccurrence_isolation"] = paired(D, B)

    # transfer verdict for the headline finding (mechanism, B)
    bcmp = out["comparisons"].get(f"{B}_vs_A")
    if bcmp:
        out["finding_transfers"] = bool(bcmp["delta_AP_agnostic_mean"] > 0)
        out["finding_summary"] = (
            f"En ORB-SLAM real, el mecanismo cooccurrence cambia AP_agnostic en "
            f"{bcmp['delta_AP_agnostic_mean']:+.4f} ({bcmp['pct_AP_agnostic']:+.1f}%, "
            f"Wilcoxon p={bcmp['wilcoxon_p_AP_agnostic']:.3f}, n={bcmp['n']}) y mIoU en "
            f"{bcmp['delta_mIoU_mean']:+.4f} ({bcmp['pct_mIoU']:+.1f}%). "
            f"{'TRANSFIERE' if bcmp['delta_AP_agnostic_mean']>0 else 'NO transfiere'} "
            f"el hallazgo del estudio sim.")

    (ROOT / "study/orbslam_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # bar plot
    labs = [l for l in [A, B, C] if l in out["configs"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["mIoU", "AP_agnostic"]):
        vals = [out["configs"][l][metric] for l in labs]
        ax.bar([l.replace("orb-", "") for l in labs], vals,
               color=["#7885b0", "#51cf66", "#5c7cfa"][:len(labs)])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"ORB-SLAM real · {metric}")
    fig.tight_layout(); fig.savefig(FIGS / "orbslam_compare.png", dpi=110); plt.close(fig)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
