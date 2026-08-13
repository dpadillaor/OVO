"""Medición de coste del AMG por crop, con las cautelas de GPU (sync, warmup, VRAM pico).

Capa de I/O de medida: instrumenta el predictor del AMG (encoder=set_image, decode=_predict).
La mugre del monkeypatch queda encerrada aquí; agregación y pintado viven fuera y son puros.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now() -> float:
    _sync()  # CUDA es asíncrono: sincroniza antes de leer el reloj
    return time.perf_counter()


@dataclass
class CropTiming:
    """Coste de un crop: total, encoder (set_image), decode (_predict) y VRAM pico.
    El post-proceso del crop (stability, RLE, box-NMS...) = total - encoder - decode."""
    total_ms: float = 0.0
    encoder_ms: float = 0.0
    decode_ms: float = 0.0
    peak_vram_mb: float = 0.0

    @property
    def post_ms(self) -> float:
        return max(self.total_ms - self.encoder_ms - self.decode_ms, 0.0)


@dataclass
class FrameProfile:
    """Perfil de coste de un frame: total, nº de máscaras crudas y desglose por crop."""
    total_ms: float
    n_masks: int
    crops: list[CropTiming] = field(default_factory=list)

    @property
    def n_crops(self) -> int:
        return len(self.crops)

    @property
    def encoder_ms(self) -> float:
        return sum(c.encoder_ms for c in self.crops)

    @property
    def decode_ms(self) -> float:
        return sum(c.decode_ms for c in self.crops)

    @property
    def post_ms(self) -> float:
        """Post-proceso dentro de los crops: stability, umbral, box-NMS, RLE, uncrop."""
        return sum(c.post_ms for c in self.crops)

    @property
    def overhead_ms(self) -> float:
        """Glue a nivel generate: cajas de crop, cross-crop NMS, ensamblado final."""
        return max(self.total_ms - sum(c.total_ms for c in self.crops), 0.0)

    @property
    def peak_vram_mb(self) -> float:
        return max((c.peak_vram_mb for c in self.crops), default=0.0)


def profile_frame(segmenter, image: np.ndarray, warmup: int = 1) -> FrameProfile:
    """Mide el coste de segmentar un frame. `segmenter` debe exponer `_amg` y `segment`."""
    pred = segmenter._amg.predictor
    for _ in range(warmup):  # primeras pasadas: compilación/autotuning, se descartan
        segmenter.segment(image)

    amg = segmenter._amg
    crops: list[CropTiming] = []
    orig_set, orig_pred, orig_crop = pred.set_image, pred._predict, amg._process_crop

    def timed_set(img, *a, **k):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t = _now()
        r = orig_set(img, *a, **k)
        crops.append(CropTiming(encoder_ms=(_now() - t) * 1000))
        return r

    def timed_pred(*a, **k):
        t = _now()
        r = orig_pred(*a, **k)
        c = crops[-1]
        c.decode_ms += (_now() - t) * 1000
        if torch.cuda.is_available():
            c.peak_vram_mb = max(c.peak_vram_mb, torch.cuda.max_memory_allocated() / 1024 ** 2)
        return r

    def timed_crop(*a, **k):
        t = _now()
        r = orig_crop(*a, **k)  # dentro llama a set_image (crea el crop) y _predict
        crops[-1].total_ms = (_now() - t) * 1000
        return r

    pred.set_image, pred._predict, amg._process_crop = timed_set, timed_pred, timed_crop
    try:
        t0 = _now()
        records = segmenter.segment(image)
        total = (_now() - t0) * 1000
    finally:
        pred.set_image, pred._predict, amg._process_crop = orig_set, orig_pred, orig_crop

    return FrameProfile(total_ms=total, n_masks=len(records), crops=crops)
