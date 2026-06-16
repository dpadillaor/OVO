"""Phase 4 — multi-objective BO (RQ2). Two independent studies (light, aggressive).

Maximize [mIoU, AP_agnostic] over the reduced param set (from Morris) on the
winning chain. NSGA-II sampler, SQLite-persisted (resumable). No pruning
(each trial is one full fusion, no intermediate values). Knee point = Pareto
point closest to ideal (1,1) after per-front [0,1] normalization.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler

from study.launcher import run_fusion_config, MASTER

ROOT = Path("/home/padidavid/repos/OVO")
STUDY_DIR = ROOT / "study"
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
INT_PARAMS = {"cooccurrence_veto_threshold"}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_context():
    morris = json.loads((ROOT / "study/morris.json").read_text())
    chain = morris["chain"]
    active = morris["reduced_params"]
    ranges = json.loads((ROOT / "study/ranges.json").read_text())["ranges"]
    return chain, active, ranges


def make_objective(cond, chain, active, ranges):
    def objective(trial):
        theta = {"fusion_criteria": list(chain)}
        for n in active:
            lo, hi = ranges[n]
            if n in INT_PARAMS:
                theta[n] = trial.suggest_int(n, int(lo), int(hi))
            else:
                theta[n] = trial.suggest_float(n, float(lo), float(hi))
        label = f"opt-{cond}-{trial.number}"
        r = run_fusion_config(cond, theta, label)
        mi = r["mIoU"] if r["mIoU"] is not None else 0.0
        ap = r["AP_agnostic"] if r["AP_agnostic"] is not None else 0.0
        trial.set_user_attr("Num_Instances", r["Num_Instances"])
        trial.set_user_attr("mAcc", r["mAcc"])
        return mi, ap
    return objective


def knee(trials):
    pts = np.array([[t.values[0], t.values[1]] for t in trials])
    mn, mx = pts.min(0), pts.max(0)
    span = np.where(mx - mn == 0, 1, mx - mn)
    norm = (pts - mn) / span
    d = np.linalg.norm(norm - np.array([1.0, 1.0]), axis=1)
    return trials[int(np.argmin(d))]


def main():
    chain, active, ranges = load_context()
    print(f"Phase 4 Optuna: chain={chain} active={active} n_trials={N_TRIALS}")
    out = {"chain": chain, "active_params": active, "n_trials_target": N_TRIALS,
           "by_condition": {}}

    for cond in ["light", "aggressive"]:
        storage = f"sqlite:///{STUDY_DIR}/optuna_{cond}.db"
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=NSGAIISampler(seed=42, population_size=18),
            storage=storage, study_name=f"fusion_{cond}", load_if_exists=True)
        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, N_TRIALS - done)
        print(f"\n[{cond}] {done} done, running {remaining} more")
        if remaining:
            study.optimize(make_objective(cond, chain, active, ranges),
                           n_trials=remaining)
        pareto = study.best_trials
        kn = knee(pareto) if pareto else None
        out["by_condition"][cond] = {
            "n_complete": len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE]),
            "pareto": [{"number": t.number, "mIoU": t.values[0],
                        "AP_agnostic": t.values[1], "params": t.params,
                        "Num_Instances": t.user_attrs.get("Num_Instances")}
                       for t in pareto],
            "knee": ({"number": kn.number, "mIoU": kn.values[0],
                      "AP_agnostic": kn.values[1],
                      "theta": {"fusion_criteria": chain, **kn.params},
                      "Num_Instances": kn.user_attrs.get("Num_Instances")}
                     if kn else None),
        }
        print(f"[{cond}] pareto size={len(pareto)} "
              f"knee={out['by_condition'][cond]['knee']}")

    (ROOT / "study/phase4_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote study/phase4_results.json")


if __name__ == "__main__":
    main()
