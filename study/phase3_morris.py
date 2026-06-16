"""Phase 3 — Morris screening (RQ3 qualitative). Which thresholds matter / interact.

Runs on the Phase-2 winning chain. Active params inferred from the chain.
Cost = N*(k+1) per condition; we use a small N (budget-capped, deterministic).
Each evaluation yields BOTH mIoU and AP_agnostic, so we analyze both objectives
from the same runs (no extra cost). mu_star = importance, sigma = interaction.

Outputs study/morris.json + figs/phase3_morris_*.png and the reduced active set.
Resumable: skips labels already in runs_master.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze

from study.launcher import run_fusion_config, MASTER

ROOT = Path("/home/padidavid/repos/OVO")
FIGS = ROOT / "study/figs"
N_TRAJ = 4  # trajectories; budget cap

ALL_PARAMS_FOR = {
    "centroid": ["th_centroid"],
    "aabb": ["th_aabb"],
    "cos_sim": ["th_cossim"],
    "overlap": ["th_points", "th_overlap_ratio", "th_overlap_cos", "th_overlap_ratio_low"],
    "overlap_old": ["th_points", "th_overlap_ratio", "th_overlap_cos", "th_overlap_ratio_low"],
    "cooccurrence": ["cooccurrence_veto_threshold"],
}


def active_params(chain):
    names = []
    for c in chain:
        for p in ALL_PARAMS_FOR.get(c, []):
            if p not in names:
                names.append(p)
    return names


def get_winner_chain():
    if len(sys.argv) > 1:
        return json.loads(sys.argv[1])
    res = json.loads((ROOT / "study/phase2_results.json").read_text())
    winner = res["winner"]
    # recover chain from any condition row
    for e in res["by_condition"]["light"]:
        if e["label"] == winner:
            ch = e["chain"]
            return json.loads(ch) if isinstance(ch, str) else ch
    raise RuntimeError("winner chain not found")


def done_labels():
    if not MASTER.exists():
        return set()
    df = pd.read_parquet(MASTER)
    return set(zip(df.label, df.condition)) if "condition" in df else set()


def main():
    chain = get_winner_chain()
    names = active_params(chain)
    ranges = json.loads((ROOT / "study/ranges.json").read_text())["ranges"]
    bounds = [ranges[n] for n in names]
    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    X = morris_sample(problem, N=N_TRAJ, num_levels=4)
    print(f"Phase 3 Morris: chain={chain}\n active={names}\n {len(X)} points/condition")

    done = done_labels()
    morris_out = {"chain": chain, "active_params": names,
                  "n_points": len(X), "by_condition": {}}

    for cond in ["light", "aggressive"]:
        Ymi, Yap = [], []
        for i, x in enumerate(X):
            theta = {"fusion_criteria": chain}
            for n, v in zip(names, x):
                theta[n] = int(round(v)) if n == "cooccurrence_veto_threshold" else float(v)
            label = f"morris-{cond}-{i}"
            if (label, cond) in done:
                df = pd.read_parquet(MASTER)
                row = df[(df.label == label) & (df.condition == cond)].iloc[0]
                r = {"mIoU": float(row.mIoU), "AP_agnostic": float(row.AP_agnostic)}
                print(f"  SKIP {label}", flush=True)
            else:
                try:
                    r = run_fusion_config(cond, theta, label)
                    print(f"  {label} mIoU={r['mIoU']:.4f} AP_agn={r['AP_agnostic']:.4f}", flush=True)
                except Exception as e:
                    print(f"  !! FAILED {label}: {e}", flush=True)
                    r = {"mIoU": np.nan, "AP_agnostic": np.nan}
            Ymi.append(r["mIoU"]); Yap.append(r["AP_agnostic"])
        Ymi, Yap = np.array(Ymi), np.array(Yap)
        cond_res = {}
        for obj, Y in [("mIoU", Ymi), ("AP_agnostic", Yap)]:
            if np.isnan(Y).any():
                # SALib needs finite Y; impute NaN with column mean
                Y = np.where(np.isnan(Y), np.nanmean(Y), Y)
            res = morris_analyze(problem, X, Y, num_levels=4)
            cond_res[obj] = {"names": names,
                             "mu_star": [float(v) for v in res["mu_star"]],
                             "sigma": [float(v) for v in res["sigma"]],
                             "mu": [float(v) for v in res["mu"]]}
        morris_out["by_condition"][cond] = cond_res

    # ---- reduced active set: params with high mu_star on EITHER objective ----
    importance = {n: 0.0 for n in names}
    for cond in morris_out["by_condition"].values():
        for obj in cond.values():
            ms = np.array(obj["mu_star"])
            ms = ms / (ms.max() + 1e-12)
            for n, v in zip(names, ms):
                importance[n] = max(importance[n], float(v))
    reduced = [n for n in names if importance[n] >= 0.25]  # keep >=25% of max importance
    if not reduced:
        reduced = names
    morris_out["normalized_importance"] = importance
    morris_out["reduced_params"] = reduced
    print(f"\nReduced active params for Optuna: {reduced}")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for j, cond in enumerate(["light", "aggressive"]):
        for k, obj in enumerate(["mIoU", "AP_agnostic"]):
            ax = axes[k, j]
            d = morris_out["by_condition"][cond][obj]
            ax.scatter(d["mu_star"], d["sigma"], color="#5c7cfa")
            for n, x, y in zip(d["names"], d["mu_star"], d["sigma"]):
                ax.annotate(n, (x, y), fontsize=7)
            ax.set_xlabel("mu*  (importance)"); ax.set_ylabel("sigma (interaction)")
            ax.set_title(f"{cond} / {obj}")
    fig.tight_layout(); fig.savefig(FIGS / "phase3_morris.png", dpi=110); plt.close(fig)

    (ROOT / "study/morris.json").write_text(json.dumps(morris_out, indent=2))
    print("Wrote study/morris.json + figs/phase3_morris.png")


if __name__ == "__main__":
    main()
