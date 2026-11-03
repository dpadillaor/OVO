"""SimulatedSLAM config: the YAML section `slam.simulated`, validated. Every key is optional.

Each piece owns its parameters and defaults (keyframes.py, drift.py, jumps.py); this module puts
them together and rejects unknown keys at startup instead of silently ignoring them.

    slam:
      close_loops: True        # corrections on: always one at the end, plus `correction.at_frames`
      simulated:
        keyframes: {dist: 0.1, rot_deg: 5.0}
        drift: {translation_std: 0.01, rotation_std_deg: 0.5, seed: 42}   # absent = no drift
        jumps: [{at_frame: 300, translation: [0, 0, -0.1]}]              # absent = no jumps
        jump_seed: 42
        correction: {at_frames: [500]}
"""

from dataclasses import dataclass, field, fields
from typing import Optional

from .drift import DriftCfg
from .jumps import JumpCfg
from .keyframes import KeyframeCfg


@dataclass(frozen=True)
class CorrectionCfg:
    at_frames: tuple[int, ...] = ()


@dataclass(frozen=True)
class SimulatedCfg:
    keyframes: KeyframeCfg = field(default_factory=KeyframeCfg)
    drift: Optional[DriftCfg] = None
    jumps: tuple[JumpCfg, ...] = ()
    jump_seed: int = 42
    correction: CorrectionCfg = field(default_factory=CorrectionCfg)

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "SimulatedCfg":
        raw = dict(raw or {})
        _check_keys(cls, raw, "slam.simulated")
        kwargs = {}
        if "keyframes" in raw:
            kwargs["keyframes"] = _build(KeyframeCfg, raw["keyframes"], "slam.simulated.keyframes")
        if raw.get("drift") is not None:
            kwargs["drift"] = _build(DriftCfg, raw["drift"], "slam.simulated.drift")
        if "jumps" in raw:
            for i, j in enumerate(raw["jumps"] or []):
                if not isinstance(j, dict) or "at_frame" not in j:
                    raise ValueError(f"slam.simulated.jumps[{i}]: each jump needs `at_frame`, got {j!r}")
            kwargs["jumps"] = tuple(_build(JumpCfg, j, f"slam.simulated.jumps[{i}]")
                                    for i, j in enumerate(raw["jumps"] or []))
        if "jump_seed" in raw:
            kwargs["jump_seed"] = int(raw["jump_seed"])
        if "correction" in raw:
            correction = dict(raw["correction"] or {})
            _check_keys(CorrectionCfg, correction, "slam.simulated.correction")
            at_frames = tuple(int(f) for f in correction.get("at_frames") or ())
            if any(f < 0 for f in at_frames):
                raise ValueError(f"slam.simulated.correction.at_frames: must be >= 0, got {list(at_frames)}")
            kwargs["correction"] = CorrectionCfg(at_frames)
        return cls(**kwargs)


def _check_keys(cls, raw: dict, path: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; valid: {sorted(known)}")


def _build(cls, raw: Optional[dict], path: str):
    raw = dict(raw or {})
    _check_keys(cls, raw, path)
    return cls(**raw)
