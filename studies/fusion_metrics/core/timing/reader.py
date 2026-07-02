"""I/O: read per-criterion fusion timing/count logs from a scene's ``logger/`` dir.

Per criterion: ``sc_<name>.log`` = pairs short-circuited (count), ``t_crit_<name>.log``
= accumulated seconds. Both are single-value files with no trailing newline, and exist
only if the criterion ran (dynamic keys).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

# Default fusion chain order. aabb is off in standard runs; overlap is the final criterion.
CRITERIA = ("cooccurrence", "centroid", "aabb", "cos_sim", "overlap")


@dataclass
class CriterionTiming:
    """One fusion criterion: name, pairs decided (count), accumulated time (s)."""

    name: str
    count: int
    time_s: float

    @property
    def mean_s(self) -> float:
        """Mean seconds per decided pair; 0.0 if the criterion never fired."""
        return self.time_s / self.count if self.count else 0.0


def logger_dir(exp_path: pathlib.Path, scene: str) -> pathlib.Path:
    """``{exp_path}/{scene}/logger``."""
    return exp_path / scene / "logger"


def _read_scalar(path: pathlib.Path) -> float | None:
    """Read a single-value .log (no trailing newline). None if missing or empty."""
    if not path.is_file():
        return None
    text = path.read_text().strip()
    return float(text) if text else None


def load_criterion_timings(exp_path: pathlib.Path, scene: str,
                           criteria: tuple[str, ...] | list[str] | None = None
                           ) -> list[CriterionTiming]:
    """Load count+time for every criterion whose sc_*/t_crit_* logs are present and non-empty.

    Parameters
    ----------
    exp_path : pathlib.Path
        Experiment output directory.
    scene : str
        Scene name.
    criteria : tuple or list of str, optional
        Criterion names to look for. If None, uses the default CRITERIA tuple.
    """
    if criteria is None:
        criteria = CRITERIA
    logs = logger_dir(exp_path, scene)
    timings: list[CriterionTiming] = []
    for name in criteria:
        count = _read_scalar(logs / f"sc_{name}.log")
        time_s = _read_scalar(logs / f"t_crit_{name}.log")
        if count is None or time_s is None:
            continue  # criterion did not run in this scene
        timings.append(CriterionTiming(name=name, count=int(count), time_s=time_s))
    return timings
