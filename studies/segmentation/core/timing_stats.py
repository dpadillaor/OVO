"""Agregación pura de perfiles de coste: List[FrameProfile] -> estadística (mean/std/min/max)."""
from __future__ import annotations

from dataclasses import dataclass

from studies.segmentation.core.profiling import FrameProfile


@dataclass(frozen=True)
class Stat:
    """Resumen de una magnitud sobre varias medidas."""
    mean: float
    std: float
    min: float
    max: float


def _stat(xs: list[float]) -> Stat:
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return Stat(mean=m, std=var ** 0.5, min=min(xs), max=max(xs))


@dataclass(frozen=True)
class TimingStats:
    """Coste agregado sobre N perfiles (reps o frames). total = encoder+decode+post+overhead."""
    n: int
    total_ms: Stat
    encoder_ms: Stat
    decode_ms: Stat
    post_ms: Stat
    overhead_ms: Stat
    peak_vram_mb: Stat
    n_masks: Stat
    n_crops: int


def aggregate(profiles: list[FrameProfile]) -> TimingStats:
    """List[FrameProfile] -> TimingStats. Asume misma config (nº de crops constante)."""
    assert profiles, "sin perfiles que agregar"
    return TimingStats(
        n=len(profiles),
        total_ms=_stat([p.total_ms for p in profiles]),
        encoder_ms=_stat([p.encoder_ms for p in profiles]),
        decode_ms=_stat([p.decode_ms for p in profiles]),
        post_ms=_stat([p.post_ms for p in profiles]),
        overhead_ms=_stat([p.overhead_ms for p in profiles]),
        peak_vram_mb=_stat([p.peak_vram_mb for p in profiles]),
        n_masks=_stat([float(p.n_masks) for p in profiles]),
        n_crops=profiles[0].n_crops,
    )
