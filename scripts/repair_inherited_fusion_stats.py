"""Strip inherited source-run fusion stats from restore-from-checkpoint experiment logs.

Background: before the logger fix, ``Logger.load_stats_from`` copied ALL stats from the
checkpoint's source run, including fusion-owned ones (sc_*, t_crit_*, t_fusion, ...). A
restore run then appended its own single fusion event, so those logs ended up as
``[source_values] + [this_run_values]``. This one-off repair removes the inherited prefix
so existing runs match the post-fix behaviour.

Dry-run by default (reports only). Pass --apply to rewrite logs in place.

Usage:
    python scripts/repair_inherited_fusion_stats.py --dataset Replica
    python scripts/repair_inherited_fusion_stats.py --dataset Replica --apply
"""
from __future__ import annotations

import sys
import pathlib
import argparse

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from ovo.entities.logger import _is_fusion_owned_stat  # noqa: E402

# Factory default chain for clip/dino/pe/sam3 (ovo/entities/fusion/factory.py).
STANDARD_CHAIN = ["cooccurrence", "centroid", "cos_sim", "overlap"]


def _read_lines(path: pathlib.Path) -> list[str]:
    """Non-empty stripped lines of a .log; [] if missing."""
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _scene_dirs(exp_dir: pathlib.Path) -> list[pathlib.Path]:
    """Scene subdirs that hold a config.yaml."""
    return sorted(d for d in exp_dir.iterdir() if d.is_dir() and (d / "config.yaml").is_file())


def _source_name(restore_path: str) -> str:
    """Experiment name embedded in a restore_pre_fusion_checkpoint path."""
    parts = pathlib.Path(restore_path).parts
    return parts[parts.index("Replica") + 1]


def repair_scene(scene_dir: pathlib.Path, dataset_root: pathlib.Path, apply: bool) -> dict:
    """Strip the source prefix from one scene's fusion logs. Returns a report dict."""
    cfg = yaml.safe_load((scene_dir / "config.yaml").read_text())
    criteria = cfg.get("semantic", {}).get("fusion_criteria")
    restore = cfg.get("restore_pre_fusion_checkpoint")
    scene = scene_dir.name

    report = {
        "scene": scene,
        "criteria": criteria,
        "standard": criteria == STANDARD_CHAIN,
        "restored": bool(restore),
        "action": "skip (no restore)",
        "stripped": {},
    }
    if not restore:
        return report

    src_logger = dataset_root / _source_name(restore) / scene / "logger"
    run_logger = scene_dir / "logger"

    # Gate: a polluted restore run has the source's event + its own = 2 values.
    npe = _read_lines(run_logger / "n_pairs_evaluated.log")
    if len(npe) < 2:
        report["action"] = f"skip (not polluted: n_pairs_evaluated={len(npe)})"
        return report

    report["action"] = "repaired" if apply else "would repair"
    for log_file in sorted(run_logger.glob("*.log")):
        key = log_file.stem
        if not _is_fusion_owned_stat(key):
            continue
        n_inherited = len(_read_lines(src_logger / f"{key}.log"))
        if n_inherited == 0:
            continue  # run-only criterion, nothing inherited
        run_lines = _read_lines(log_file)
        kept = run_lines[n_inherited:]
        report["stripped"][key] = (len(run_lines), n_inherited, len(kept))
        if apply:
            log_file.write_text("\n".join(kept))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Replica", help="Dataset under data/output/")
    parser.add_argument("--apply", action="store_true", help="Rewrite logs (default: dry-run)")
    args = parser.parse_args(argv)

    dataset_root = REPO_ROOT / "data" / "output" / args.dataset
    exps = sorted(d for d in dataset_root.iterdir() if d.is_dir())

    n_repaired = 0
    for exp in exps:
        rows = [repair_scene(s, dataset_root, args.apply) for s in _scene_dirs(exp)]
        if not rows:
            continue
        restored = rows[0]["restored"]
        std = "standard" if rows[0]["standard"] else f"custom {rows[0]['criteria']}"
        touched = [r for r in rows if r["stripped"]]
        n_repaired += len(touched)
        tag = f"restore" if restored else "no-restore"
        print(f"\n{exp.name}\n  config: {std} | {tag}")
        for r in touched:
            strip = ", ".join(f"{k}:{a}->{c}" for k, (a, _, c) in r["stripped"].items())
            print(f"    {r['scene']}: {r['action']} | {strip}")
        if restored and not touched:
            print(f"    (no scenes needed repair)")

    mode = "APPLIED" if args.apply else "DRY-RUN (use --apply to write)"
    print(f"\n{mode}: {n_repaired} scene-logs {'repaired' if args.apply else 'to repair'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
