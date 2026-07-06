"""Load contest Tier2 per-KF telemetry (logger/contest/*.log) at scene / experiment level."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

try:
    import yaml
except ImportError:  # yaml is optional: only needed to mark jumps
    yaml = None

OUTPUT_ROOT = pathlib.Path(__file__).resolve().parents[3] / "data" / "output" / "Replica"

# Canonical Tier2 signals (mirror ovo/entities/logger.py::_CONTEST_KF_STATS), in read order.
KF_SIGNALS = ["n_matched", "n_pre_assign", "n_used", "n_orphans", "n_births", "n_robos"]

# Per-signal display metadata: nice title + y-axis label.
SIGNAL_META = {
    "n_matched":    ("Matched points / KF", "points"),
    "n_pre_assign": ("Pre-assigned matched points / KF", "points"),
    "n_used":       ("Masks that assigned points / KF", "masks"),
    "n_orphans":    ("Orphan matched points / KF", "points"),
    "n_births":     ("New instances born / KF", "instances"),
    "n_robos":      ("Grab events (robos) / KF", "grabs"),
}


@dataclass
class SceneContestData:
    """Contest Tier2 telemetry for a single scene."""

    exp_id: str
    scene: str
    series: dict[str, np.ndarray]          # signal -> per-KF values
    frame_ids: np.ndarray | None = None    # KF idx -> source frame id (from logger/frame_id.log)
    jumps: list[dict] = field(default_factory=list)  # [{kf: int, label: str}] from config.yaml

    @property
    def n_kf(self) -> int:
        return max((len(v) for v in self.series.values()), default=0)

    @property
    def signals(self) -> list[str]:
        """Present raw signals, canonical order first then any extras (e.g. t_contest_*)."""
        present = [s for s in KF_SIGNALS if s in self.series]
        extra = sorted(s for s in self.series if s not in KF_SIGNALS)
        return present + extra

    @property
    def derived(self) -> dict[str, np.ndarray]:
        """Rate signals derived on the fly (not stored). Empty if inputs missing."""
        out: dict[str, np.ndarray] = {}
        s = self.series
        if "n_robos" in s and "n_pre_assign" in s:
            out["robo_rate"] = s["n_robos"] / np.maximum(s["n_pre_assign"], 1)
        if "n_orphans" in s and "n_matched" in s:
            out["orphan_rate"] = s["n_orphans"] / np.maximum(s["n_matched"], 1)
        return out


def _read_log(path: pathlib.Path) -> np.ndarray:
    """Read a per-KF .log (one number per line) into a float array."""
    txt = path.read_text().strip()
    if not txt:
        return np.array([])
    toks = txt.replace("\n", ",").split(",")
    return np.array([float(t) for t in toks if t.strip() != ""])


def contest_log_dir(exp_path: str | pathlib.Path, scene: str) -> pathlib.Path:
    return pathlib.Path(exp_path) / scene / "logger" / "contest"


def _load_jumps(exp_path: pathlib.Path, scene: str, frame_ids: np.ndarray | None) -> list[dict]:
    """Read jump keyframes from the scene config so figures can mark them."""
    cfg_path = pathlib.Path(exp_path) / scene / "config.yaml"
    if yaml is None or not cfg_path.exists():
        return []
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
    except Exception:
        return []
    jumps_cfg = (((cfg or {}).get("noise") or {}).get("jumps")) or []
    out: list[dict] = []
    for j in jumps_cfg:
        if "kf_index" in j:
            kf = int(j["kf_index"])
        elif "frame_id" in j and frame_ids is not None and len(frame_ids):
            # map the trigger frame to its KF position (nearest logged frame)
            kf = int(np.argmin(np.abs(frame_ids - int(j["frame_id"]))))
        else:
            continue
        axes = [a for a in ("yaw", "pitch", "roll")
                if isinstance(j.get("rotation_jump"), dict)
                and j["rotation_jump"].get(a, {}).get("enabled")]
        label = f"jump@{kf}" + (f" ({'+'.join(axes)})" if axes else "")
        out.append({"kf": kf, "label": label})
    return out


def discover_scenes(exp_path: str | pathlib.Path) -> list[str]:
    p = pathlib.Path(exp_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {p}")
    scenes = []
    for entry in sorted(p.iterdir()):
        d = contest_log_dir(entry, entry.name)
        if entry.is_dir() and d.is_dir() and any(d.glob("*.log")):
            scenes.append(entry.name)
    return scenes


def load_scene(exp_path: str | pathlib.Path, scene: str) -> SceneContestData:
    log_dir = contest_log_dir(exp_path, scene)
    if not log_dir.is_dir():
        raise FileNotFoundError(f"No contest logs for {exp_path}/{scene}: {log_dir} missing")

    series: dict[str, np.ndarray] = {}
    for f in sorted(log_dir.glob("*.log")):
        arr = _read_log(f)
        if len(arr):
            series[f.stem] = arr
    if not series:
        raise FileNotFoundError(f"No non-empty contest logs in {log_dir}")

    fid_path = pathlib.Path(exp_path) / scene / "logger" / "frame_id.log"
    frame_ids = _read_log(fid_path) if fid_path.exists() else None
    jumps = _load_jumps(pathlib.Path(exp_path), scene, frame_ids)

    return SceneContestData(
        exp_id=pathlib.Path(exp_path).name,
        scene=scene,
        series=series,
        frame_ids=frame_ids,
        jumps=jumps,
    )


def load_experiment(exp_path: str | pathlib.Path) -> list[SceneContestData]:
    scenes = discover_scenes(exp_path)
    if not scenes:
        raise FileNotFoundError(f"No scenes with contest logs in {exp_path}")
    return [load_scene(exp_path, s) for s in scenes]
