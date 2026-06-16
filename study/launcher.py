"""OVO Fusion Study — central launcher.

Single entry point `run_fusion_config(condition, theta, label)` used by ALL phases.
Builds a replay manifest from a baseline pre_fusion checkpoint, runs only
fusion+segment+eval+eval_instances (SLAM/SAM skipped), parses metrics via
results_utils, appends a row to the master table, and returns the metrics.

Deterministic replay -> no seeds / repetitions. Statistical unit = scene (n=8).
"""
import subprocess, json, datetime as dt
from pathlib import Path

import yaml
import pandas as pd

from ovo.utils import results_utils as ru

ROOT = Path("/home/padidavid/repos/OVO")
OUTPUT_ROOT = ROOT / "data/output"           # results_utils scans output_dir/{dataset}/{exp}
OUT_DATASET = OUTPUT_ROOT / "Replica"
DATASET = "Replica"
COMMIT = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()

SCENES = ["office0", "office1", "office2", "office3", "office4",
          "room0", "room1", "room2"]

# Exact baseline checkpoint folders (Phase 0.2 verified).
CKPT = {
    "light":      "20260613_GTJump-J2-T0p03-R0p0_CLIP_light-baseline_82bfd",
    "aggressive": "20260613_GTJump-J2-T0p3-R0p0_CLIP_agressive-baseline_5f2f1",
}

MASTER = ROOT / "study/runs_master.parquet"
MANIFEST_INDEX = ROOT / "study/manifest_index.csv"

# Threshold keys passed through to ovo_config.semantic when present in theta.
_TH_KEYS = ["th_centroid", "th_aabb", "th_cossim", "th_points",
            "th_overlap_ratio", "th_overlap_cos", "th_overlap_ratio_low",
            "cooccurrence_veto_threshold"]


def build_manifest(condition, theta, label, scenes=None):
    base = CKPT[condition]
    sem = {"fusion_method": "clip",
           "fusion_criteria": list(theta["fusion_criteria"])}
    for k in _TH_KEYS:
        if k in theta:
            sem[k] = theta[k]
    exp = {
        "label": label,
        "stages": ["run", "segment", "eval", "eval_instances"],
        # {scene} substituted per-scene by run_eval.run_scene
        "restore_pre_fusion_checkpoint":
            f"data/checkpoints/Replica/{base}/{{scene}}/pre_fusion.ckpt",
        "ovo_config": {
            "slam": {"slam_module": "simulated", "close_loops": True},
            "semantic": sem,
            "vis": {"stream": True, "type": "rerun",
                    "rerun_visual_mode": "off", "save_rrd": False},
        },
    }
    if scenes is None:
        exp["scenes_list"] = "scenes.txt"
    else:
        exp["scenes_id"] = list(scenes)
    mani = {"default_dataset": DATASET, "experiments": [exp]}
    mdir = ROOT / "manifests"
    p = mdir / f"{dt.date.today():%Y%m%d}_replay_{condition}_{label}.yaml"
    p.write_text(yaml.safe_dump(mani, sort_keys=False))
    return p


def _existing_folders():
    if not OUT_DATASET.exists():
        return set()
    return {d.name for d in OUT_DATASET.iterdir() if d.is_dir()}


def _append_master(row):
    flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
            for k, v in row.items()}
    df = pd.read_parquet(MASTER) if MASTER.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([flat])], ignore_index=True)
    df.to_parquet(MASTER)
    df.to_csv(MASTER.with_suffix(".csv"), index=False)


def _append_manifest_index(label, condition, theta, manifest, exp_id):
    row = {"timestamp": dt.datetime.now().isoformat(), "label": label,
           "condition": condition, "experiment_id": exp_id,
           "manifest": str(manifest.relative_to(ROOT)),
           "theta": json.dumps(theta)}
    df = pd.read_csv(MANIFEST_INDEX) if MANIFEST_INDEX.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(MANIFEST_INDEX, index=False)


def run_fusion_config(condition, theta, label, scenes=None, timeout=14400):
    """Run one replay config across the given scenes (default all 8).

    Robust run-folder detection: snapshot output folders before, diff after.
    """
    assert condition in CKPT
    mani = build_manifest(condition, theta, label, scenes=scenes)
    before = _existing_folders()
    t0 = dt.datetime.now()
    proc = subprocess.run(
        ["conda", "run", "-n", "ovo2", "python",
         "scripts/run_experiments_batch.py",
         "--manifest", str(mani.relative_to(ROOT)), "--verbose"],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    after = _existing_folders()
    new = sorted(after - before)
    if not new:
        # fall back: maybe folder reused; match by label substring, newest mtime
        cands = [d for d in OUT_DATASET.iterdir()
                 if d.is_dir() and f"_{label}_" in d.name]
        if not cands:
            raise RuntimeError(
                f"No output folder for label={label}. "
                f"rc={proc.returncode}\nSTDERR tail:\n{proc.stderr[-2000:]}")
        exp_id = max(cands, key=lambda d: d.stat().st_mtime).name
    else:
        # there should be exactly one new folder for this label
        match = [n for n in new if f"_{label}_" in n]
        exp_id = (match[0] if match else new[-1])

    exp_df, _ = ru.load_experiments(str(OUTPUT_ROOT))
    sc_df, _ = ru.load_scene_results(str(OUTPUT_ROOT))
    row_exp = exp_df[exp_df["Experiment_ID"] == exp_id]
    if row_exp.empty:
        raise RuntimeError(f"load_experiments found no row for {exp_id}")
    r = row_exp.iloc[0]
    scl = sc_df[sc_df["Experiment_ID"] == exp_id] if "Experiment_ID" in sc_df else sc_df.iloc[0:0]

    def g(col):
        v = r.get(col)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    per_scene = []
    for _, s in scl.iterrows():
        per_scene.append({c: (float(s[c]) if c != "Scene" else s[c])
                          for c in ["Scene", "mIoU", "mAcc", "AP_agnostic", "Num_Instances"]
                          if c in scl.columns})

    metrics = {
        "condition": condition, "label": label, "experiment_id": exp_id,
        "commit": COMMIT, "ts": t0.isoformat(),
        "elapsed_s": (dt.datetime.now() - t0).total_seconds(),
        "fusion_criteria": list(theta["fusion_criteria"]),
        **{k: theta[k] for k in _TH_KEYS if k in theta},
        "mIoU": g("mIoU"), "mAcc": g("mAcc"),
        "AP": g("AP"), "AP_agnostic": g("AP_agnostic"),
        "Num_Instances": g("Num_Instances"),
        "n_scenes": int(len(scl)),
        "per_scene": per_scene,
    }
    _append_master(metrics)
    _append_manifest_index(label, condition, theta, mani, exp_id)
    return metrics


if __name__ == "__main__":
    import sys
    cond = sys.argv[1] if len(sys.argv) > 1 else "light"
    lbl = sys.argv[2] if len(sys.argv) > 2 else "smoke-default"
    theta = {"fusion_criteria": ["centroid", "cos_sim", "overlap"]}
    print(json.dumps(run_fusion_config(cond, theta, lbl), indent=2, default=str))
