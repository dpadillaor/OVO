"""Phase 6 — robustness & consistency (RQ4). Minimal extra runs (2 cross-runs).

- Cross light<->aggressive: apply each condition's knee theta to the other
  condition; compare. If same theta leads on both -> robust.
- Per-scene consistency: mean +/- std of mIoU / AP_agn across the 8 scenes for
  each knee theta.
- LOSO: leave-one-scene-out resampling of the stored 8 per-scene values
  (no new runs) -> std of the leave-one-out aggregate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from study.launcher import run_fusion_config, MASTER

ROOT = Path("/home/padidavid/repos/OVO")


def per_scene_of(label, cond):
    df = pd.read_parquet(MASTER)
    row = df[(df.label == label) & (df.condition == cond)]
    if row.empty:
        return None
    ps = row.iloc[0]["per_scene"]
    if isinstance(ps, str):
        ps = json.loads(ps)
    return pd.DataFrame(ps)


def loso(values):
    values = np.asarray(values, float)
    n = len(values)
    aggs = [np.mean(np.delete(values, i)) for i in range(n)]
    return {"mean": float(np.mean(aggs)), "std": float(np.std(aggs)),
            "min": float(np.min(aggs)), "max": float(np.max(aggs))}


def main():
    p4 = json.loads((ROOT / "study/phase4_results.json").read_text())
    out = {"cross_condition": {}, "consistency": {}, "loso": {}}

    knees = {c: p4["by_condition"][c]["knee"] for c in ["light", "aggressive"]}

    # ---- cross-condition application ----
    cross = {"light": "aggressive", "aggressive": "light"}
    for src, tgt in cross.items():
        theta = knees[src]["theta"]
        label = f"robust-{src}knee-on-{tgt}"
        df = pd.read_parquet(MASTER) if MASTER.exists() else pd.DataFrame()
        prev = df[(df.label == label) & (df.condition == tgt)] if len(df) else df
        if len(prev):
            r = prev.iloc[0].to_dict()
            print(f"  SKIP {label}")
        else:
            print(f"  RUN  {label} (theta from {src} knee on {tgt})", flush=True)
            r = run_fusion_config(tgt, theta, label)
        out["cross_condition"][f"{src}_knee_on_{tgt}"] = {
            "mIoU": float(r["mIoU"]), "AP_agnostic": float(r["AP_agnostic"]),
            "Num_Instances": float(r["Num_Instances"])}

    # native knee performance for comparison
    for c in ["light", "aggressive"]:
        out["cross_condition"][f"{c}_knee_on_{c}"] = {
            "mIoU": knees[c]["mIoU"], "AP_agnostic": knees[c]["AP_agnostic"],
            "Num_Instances": knees[c]["Num_Instances"]}

    # robust verdict: does light-knee stay >= within 5% of native on aggressive
    # and vice versa?
    def rel(a, b):
        return (a - b) / b if b else 0.0
    la = out["cross_condition"]["light_knee_on_aggressive"]["mIoU"]
    aa = out["cross_condition"]["aggressive_knee_on_aggressive"]["mIoU"]
    al = out["cross_condition"]["aggressive_knee_on_light"]["mIoU"]
    ll = out["cross_condition"]["light_knee_on_light"]["mIoU"]
    out["cross_condition"]["light_knee_rel_drop_on_aggressive"] = float(rel(la, aa))
    out["cross_condition"]["aggressive_knee_rel_drop_on_light"] = float(rel(al, ll))
    consistent = (abs(rel(la, aa)) < 0.05) and (abs(rel(al, ll)) < 0.05)
    out["consistent_across_conditions"] = bool(consistent)

    # ---- per-scene consistency + LOSO ----
    for c in ["light", "aggressive"]:
        label = f"opt-{c}-{knees[c]['number']}"
        psf = per_scene_of(label, c)
        if psf is None:
            print(f"  !! no per-scene for {label}")
            continue
        for metric in ["mIoU", "AP_agnostic"]:
            vals = psf[metric].astype(float).values
            out["consistency"].setdefault(c, {})[metric] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "per_scene": {s: float(v) for s, v in zip(psf["Scene"], vals)}}
            out["loso"].setdefault(c, {})[metric] = loso(vals)

    (ROOT / "study/phase6_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote study/phase6_results.json")
    print(json.dumps({"consistent_across_conditions": out["consistent_across_conditions"],
                      "light_knee_rel_drop_on_aggressive":
                          out["cross_condition"]["light_knee_rel_drop_on_aggressive"],
                      "aggressive_knee_rel_drop_on_light":
                          out["cross_condition"]["aggressive_knee_rel_drop_on_light"]},
                     indent=2))


if __name__ == "__main__":
    main()
