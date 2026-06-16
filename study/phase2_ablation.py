"""Phase 2 — mechanism ablation (RQ1). Thresholds = defaults.

Factorial: geometry {centroid, aabb, centroid+aabb} x overlap {overlap,
overlap_old} x cooccurrence {off, on} = 12 chains. cos_sim always present.
Order: [cooccurrence?, geometry..., cos_sim, overlap_variant]. Each chain x
{light, aggressive} x 8 scenes via the single launcher. Covisibility axis
dropped: not implemented in current code (silently ignored) -> declared.

Resumable: skips (label, condition) pairs already present in runs_master.
"""
import itertools
from pathlib import Path

import pandas as pd

from study.launcher import run_fusion_config, MASTER

ROOT = Path("/home/padidavid/repos/OVO")

GEOM = [["centroid"], ["aabb"], ["centroid", "aabb"]]
OVERLAP = ["overlap", "overlap_old"]
COOC = [False, True]


def chain_label(chain):
    return "abl-" + "-".join(chain).replace("_", "")


def already_done(label, cond):
    if not MASTER.exists():
        return False
    df = pd.read_parquet(MASTER)
    if "label" not in df or "condition" not in df:
        return False
    return not df[(df.label == label) & (df.condition == cond)].empty


def main():
    configs = []
    for g, o, c in itertools.product(GEOM, OVERLAP, COOC):
        chain = (["cooccurrence"] if c else []) + g + ["cos_sim", o]
        configs.append(chain)
    print(f"Phase 2: {len(configs)} chains x 2 conditions = {len(configs)*2} runs")

    for chain in configs:
        label = chain_label(chain)
        theta = {"fusion_criteria": chain}
        for cond in ["light", "aggressive"]:
            if already_done(label, cond):
                print(f"  SKIP {cond:10s} {label} (already in master)")
                continue
            print(f"  RUN  {cond:10s} {label}  chain={chain}", flush=True)
            try:
                r = run_fusion_config(cond, theta, label)
                print(f"       -> mIoU={r['mIoU']:.4f} AP_agn={r['AP_agnostic']} "
                      f"Ninst={r['Num_Instances']} ({r['elapsed_s']:.0f}s)", flush=True)
            except Exception as e:
                print(f"       !! FAILED {cond} {label}: {e}", flush=True)
    print("Phase 2 done.")


if __name__ == "__main__":
    main()
