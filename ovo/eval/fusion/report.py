"""Report: where each evaluated fusion spent its time, next to what each step decided.

    python -m ovo eval report --dataset Replica --experiment my_exp [--scenes office0 [--frame 1990]]
                              [--out report.csv]

It only reads: the fusions `eval fusion` evaluated ({scene}/fusion/eval_{frame}/) and the run's
step times ({scene}/fusion/fusion_timings.csv, ovo.entities.fusion.timing). One row per evaluated
fusion and step, then a TOTAL per step over all of them:

    scene,frame_id,step,in,reject,FN,TN,prepare_s,pairs_s,ms_pair
    office0,1990,centroid,248,180,2,178,0.002,0.010,0.04

    in, reject           judged pairs that reached the step / that it stopped; all of them, since
                         they all cost time (decisions_eval.csv)
    FN, TN               of the scored pairs it stopped (summary.json cascade: pairs with a mask)
    prepare_s, pairs_s   once per fusion / summed over the pairs (fusion_timings.csv)
    ms_pair              pairs_s / in, in milliseconds

extract and apply judge no pairs: their pair columns are empty. The table is printed; with
`--out`, it is also written as CSV, with every decimal.
"""

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ovo import layout
from . import decisions

COLUMNS = ("scene", "frame_id", "step", "in", "reject", "FN", "TN", "prepare_s", "pairs_s", "ms_pair")
TOTAL = "TOTAL"


def report_experiment(dataset: str, experiment: str, scenes: Optional[list[str]] = None,
                      frame: Optional[int] = None) -> list[dict]:
    """Rows of every evaluated fusion of the experiment (or of `scenes`; `frame`: that one only,
    one scene), then the TOTAL rows."""
    exp = layout.ExperimentLayout.named(dataset, experiment)
    if frame is not None and (scenes is None or len(scenes) != 1):
        raise ValueError("--frame needs exactly one scene: each scene fuses at its own frames")
    if scenes is None:
        found = exp.root.iterdir() if exp.root.is_dir() else []
        scenes = sorted(p.name for p in found if p.is_dir() and _evaluated(p))
    jobs = [(scene, f) for scene in scenes for f in _frames(exp.scene(scene), scene, frame)]
    if not jobs:
        raise ValueError(f"{exp.root}: no evaluated fusion; run `python -m ovo eval fusion` first")
    rows = [row for scene, f in jobs for row in fusion_rows(scene, exp.scene(scene), f)]
    return rows + total_rows(rows)


def fusion_rows(scene: str, scene_dir: Path, frame_id: int) -> list[dict]:
    """The rows of one evaluated fusion: extract, its pipeline steps, apply."""
    summary = json.loads(layout.fusion_eval_summary(scene_dir, frame_id).read_text())
    cascade = {c["step"]: c for c in summary["cascade"]}
    steps = list(cascade)
    reached, stopped = _counts(layout.fusion_eval_decisions(scene_dir, frame_id), steps)
    times = _times(scene, scene_dir, frame_id)
    if list(times) != ["extract", *steps, "apply"]:
        raise ValueError(f"{scene}, frame {frame_id}: timed steps {list(times)} are not the evaluated "
                         f"{steps}: the timings do not belong to this fusion")
    rows = []
    for step, (prepare_s, pairs_s) in times.items():
        row = {"scene": scene, "frame_id": frame_id, "step": step, "prepare_s": prepare_s, "pairs_s": pairs_s}
        if step in cascade:
            row.update({"in": reached[step], "reject": stopped[step], "FN": cascade[step]["FN"],
                        "TN": cascade[step]["TN"], "ms_pair": _ms(pairs_s, reached[step])})
        rows.append(row)
    return rows


def total_rows(rows: list[dict]) -> list[dict]:
    """One row per step (in first-seen order), summed over every fusion."""
    sums: dict[str, dict] = {}
    for row in rows:
        total = sums.setdefault(row["step"], defaultdict(float))
        for key in ("in", "reject", "FN", "TN", "prepare_s", "pairs_s"):
            if row.get(key) is not None:
                total[key] += row[key]
    out = []
    for step, total in sums.items():
        row = {"scene": TOTAL, "frame_id": None, "step": step, "prepare_s": total["prepare_s"]}
        if "in" in total:
            row.update({k: int(total[k]) for k in ("in", "reject", "FN", "TN")})
            row.update({"pairs_s": total["pairs_s"], "ms_pair": _ms(total["pairs_s"], row["in"])})
        out.append(row)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict]) -> str:
    """The table for the terminal, rounded."""
    def cell(key: str, value) -> str:
        if value is None:
            return ""
        if key in ("prepare_s", "pairs_s"):
            return f"{value:.4f}"
        if key == "ms_pair":
            return f"{value:.3f}"
        return str(value)

    table = [list(COLUMNS)] + [[cell(k, r.get(k)) for k in COLUMNS] for r in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(COLUMNS))]
    return "\n".join("  ".join(c.ljust(w) if i < 3 else c.rjust(w) for i, (c, w) in enumerate(zip(line, widths)))
                     for line in table)


def _evaluated(scene_dir: Path) -> bool:
    return any(layout.fusion_dir(scene_dir).glob("eval_*/summary.json"))


def _frames(scene_dir: Path, scene: str, frame: Optional[int]) -> list[int]:
    """Evaluated fusion frames of a scene, ascending."""
    frames = sorted(json.loads(p.read_text())["frame_id"]
                    for p in layout.fusion_dir(scene_dir).glob("eval_*/summary.json"))
    if frame is None:
        return frames
    if frame not in frames:
        raise ValueError(f"{scene}: the fusion at frame {frame} is not evaluated; evaluated: {frames}")
    return [frame]


def _counts(path: Path, steps: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Per step, judged pairs that reached it and that it stopped (an accepted pair passes all)."""
    rows = decisions.parse(path.read_text()).rows
    index = {name: i for i, name in enumerate(steps)}
    left_at = [len(steps) if r.accepted else index[r.reason] for r in rows]
    reached = {s: sum(at >= i for at in left_at) for s, i in index.items()}
    stopped = {s: sum(at == i for at in left_at) for s, i in index.items()}
    return reached, stopped


def _times(scene: str, scene_dir: Path, frame_id: int) -> dict[str, tuple[float, Optional[float]]]:
    path = layout.fusion_timings(scene_dir)
    if not path.is_file():
        raise FileNotFoundError(f"{scene}: no step times at {path} (a run older than them?)")
    times = {r["step"]: (float(r["prepare_s"]), float(r["pairs_s"]) if r["pairs_s"] else None)
             for r in csv.DictReader(io.StringIO(path.read_text())) if int(r["frame_id"]) == frame_id}
    if not times:
        raise ValueError(f"{scene}: no step times for the fusion at frame {frame_id} in {path}")
    return times


def _ms(seconds: Optional[float], pairs: int) -> Optional[float]:
    return None if seconds is None or pairs == 0 else 1000 * seconds / pairs
