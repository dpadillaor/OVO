"""External validation on ORB-SLAM2 (real drift, NO checkpoints, full pipeline).

Tests whether the simulated-jump-drift study findings transfer to real SLAM.
Three configs x 8 scenes (full SLAM+SAM+CLIP+fusion+eval each, ~18 min/scene):
  A) baseline   [centroid, cos_sim, overlap]            defaults
  B) mechanism  [cooccurrence, aabb, cos_sim, overlap]  defaults  (pure test of finding)
  C) tuned      recommended_overall theta from the study (does synthetic tuning transfer?)

ORB-SLAM is multithreaded => NOT deterministic; one sample per (config, scene),
paired Wilcoxon across the 8 scenes. Resumable: skips configs already in master.

Usage:
  python -m study.orbslam_validate            # full: 3 configs x 8 scenes
  python -m study.orbslam_validate --smoke    # config A on office0 only
"""
import json, subprocess, datetime as dt, sys
from pathlib import Path

import yaml
import pandas as pd

from ovo.utils import results_utils as ru

ROOT = Path("/home/padidavid/repos/OVO")
OUTPUT_ROOT = ROOT / "data/output"
OUT_DATASET = OUTPUT_ROOT / "Replica"
COMMIT = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
MASTER = ROOT / "study/orbslam_master.parquet"

_TH_KEYS = ["th_centroid", "th_aabb", "th_cossim", "th_points",
            "th_overlap_ratio", "th_overlap_cos", "th_overlap_ratio_low",
            "cooccurrence_veto_threshold"]


def _recommended_theta():
    c = json.loads((ROOT / "study/conclusions.json").read_text())
    return c["best_config"]["recommended_overall"]["theta"]


def configs():
    return [
        ("orb-A-baseline", {"fusion_criteria": ["centroid", "cos_sim", "overlap"]}),
        ("orb-B-mechanism", {"fusion_criteria": ["cooccurrence", "aabb", "cos_sim", "overlap"]}),
        ("orb-C-tuned", _recommended_theta()),
    ]


def build_manifest(label, theta, scenes=None):
    sem = {"fusion_method": "clip",
           "fusion_criteria": list(theta["fusion_criteria"])}
    for k in _TH_KEYS:
        if k in theta:
            sem[k] = theta[k]
    exp = {
        "label": label,
        "stages": ["run", "segment", "eval", "eval_instances"],
        "ovo_config": {
            "slam": {"slam_module": "orbslam2", "close_loops": True},
            "semantic": sem,
            "vis": {"stream": True, "type": "rerun",
                    "rerun_visual_mode": "off", "save_rrd": False},
        },
    }
    if scenes is None:
        exp["scenes_list"] = "scenes.txt"
    else:
        exp["scenes_id"] = list(scenes)
    mani = {"default_dataset": "Replica", "experiments": [exp]}
    p = ROOT / f"manifests/{dt.date.today():%Y%m%d}_orbslam_{label}.yaml"
    p.write_text(yaml.safe_dump(mani, sort_keys=False))
    return p


def _existing():
    return {d.name for d in OUT_DATASET.iterdir() if d.is_dir()} if OUT_DATASET.exists() else set()


def _append(row):
    flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
    df = pd.read_parquet(MASTER) if MASTER.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([flat])], ignore_index=True)
    df.to_parquet(MASTER); df.to_csv(MASTER.with_suffix(".csv"), index=False)


def run_config(label, theta, scenes=None, timeout=43200):
    mani = build_manifest(label, theta, scenes)
    before = _existing()
    t0 = dt.datetime.now()
    proc = subprocess.run(
        ["conda", "run", "-n", "ovo2", "python", "scripts/run_experiments_batch.py",
         "--manifest", str(mani.relative_to(ROOT)), "--verbose"],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    new = sorted(_existing() - before)
    match = [n for n in new if f"_{label}_" in n]
    if not match:
        raise RuntimeError(f"No output for {label}. rc={proc.returncode}\n{proc.stderr[-1500:]}")
    exp_id = match[0]
    exp_df, _ = ru.load_experiments(str(OUTPUT_ROOT))
    sc_df, _ = ru.load_scene_results(str(OUTPUT_ROOT))
    r = exp_df[exp_df.Experiment_ID == exp_id].iloc[0]
    scl = sc_df[sc_df.Experiment_ID == exp_id] if "Experiment_ID" in sc_df else sc_df.iloc[0:0]

    def g(c):
        try:
            return float(r.get(c))
        except (TypeError, ValueError):
            return None
    per_scene = [{c: (float(s[c]) if c != "Scene" else s[c])
                  for c in ["Scene", "mIoU", "mAcc", "AP_agnostic", "Num_Instances"] if c in scl.columns}
                 for _, s in scl.iterrows()]
    row = {"label": label, "experiment_id": exp_id, "commit": COMMIT,
           "ts": t0.isoformat(), "elapsed_s": (dt.datetime.now() - t0).total_seconds(),
           "fusion_criteria": list(theta["fusion_criteria"]),
           **{k: theta[k] for k in _TH_KEYS if k in theta},
           "mIoU": g("mIoU"), "mAcc": g("mAcc"), "AP": g("AP"),
           "AP_agnostic": g("AP_agnostic"), "Num_Instances": g("Num_Instances"),
           "n_scenes": int(len(scl)), "per_scene": per_scene}
    _append(row)
    return row


def done(label):
    if not MASTER.exists():
        return False
    df = pd.read_parquet(MASTER)
    return "label" in df and not df[df.label == label].empty


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        lbl, theta = configs()[0]
        print(f"SMOKE ORB-SLAM: {lbl} on office0", flush=True)
        r = run_config(lbl + "-smoke", theta, scenes=["office0"])
        print(json.dumps({k: r[k] for k in ["mIoU", "AP_agnostic", "Num_Instances", "elapsed_s"]}, default=str))
        return
    for lbl, theta in configs():
        if done(lbl):
            print(f"SKIP {lbl}", flush=True); continue
        print(f"RUN {lbl} (8 scenes) theta={theta}", flush=True)
        try:
            r = run_config(lbl, theta)
            print(f"  -> mIoU={r['mIoU']:.4f} AP_agn={r['AP_agnostic']} "
                  f"Ninst={r['Num_Instances']} n={r['n_scenes']} ({r['elapsed_s']/60:.0f}min)", flush=True)
        except Exception as e:
            print(f"  !! FAILED {lbl}: {e}", flush=True)
    print("ORB-SLAM validation done.")


if __name__ == "__main__":
    main()
