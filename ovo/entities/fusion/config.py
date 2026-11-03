"""Fusion config: the YAML section `semantic.instance_fusion`, validated. Every key is optional.

Each piece owns its parameters and defaults (overlap.py, rule.py, gates.py); this module only
puts them together and rejects unknown keys at startup instead of silently ignoring them.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from .overlap import OverlapCfg
from .rule import AcceptCfg


@dataclass(frozen=True)
class GateSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionCfg:
    gates: tuple[GateSpec, ...] = ()
    overlap: OverlapCfg = field(default_factory=OverlapCfg)
    accept: AcceptCfg = field(default_factory=AcceptCfg)
    apply: str = "greedy"  # greedy | chained

    @classmethod
    def from_dict(cls, raw: dict | None) -> "FusionCfg":
        raw = dict(raw or {})
        gates = raw.pop("gates", None)
        cfg = _build(cls, raw, "instance_fusion")
        if gates is not None:
            cfg = _with(cfg, gates=tuple(_gate_spec(g) for g in gates))
        return cfg


def _build(cls, raw: dict, path: str):
    """Instantiate dataclass `cls` from `raw`, recursing into nested dataclasses.
    Missing keys keep their default; unknown keys raise."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; valid: {sorted(known)}")
    kwargs = {}
    for name, value in raw.items():
        factory = known[name].default_factory
        default = factory() if callable(factory) else known[name].default
        if is_dataclass(default) and isinstance(value, dict):
            value = _build(type(default), value, f"{path}.{name}")
        elif isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def _gate_spec(entry) -> GateSpec:
    """`centroid` or `{centroid: {max_dist: 1.5}}` -> GateSpec."""
    if isinstance(entry, str):
        return GateSpec(entry)
    if isinstance(entry, dict) and len(entry) == 1:
        (name, params), = entry.items()
        return GateSpec(name, dict(params or {}))
    raise ValueError(f"instance_fusion.gates: each entry must be a name or {{name: {{params}}}}, got {entry!r}")


def _with(cfg, **changes):
    return type(cfg)(**{**{f.name: getattr(cfg, f.name) for f in fields(cfg)}, **changes})
