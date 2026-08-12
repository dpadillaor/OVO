"""NMS externo de OVO (mask_nms) como cálculo puro que además explica cada descarte.

Replica la semántica exacta de `ovo.utils.segment_utils.mask_nms` (mismos umbrales,
misma red de seguridad top-3) y anota, por máscara, qué regla la eliminó y quién la mató.
Sin modelo, sin disco, sin pintado: solo tensores -> DecisionBreakdown.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import torch

# Regla que descarta una máscara. Nombres desde la perspectiva de la máscara eliminada:
#   iou             -> se solapa demasiado (IoU) con otra mejor puntuada
#   inner_contained -> está contenida dentro de una mejor puntuada
#   inner_container -> contiene (o queda dentro de) una peor puntuada que gana el pulso
#   low_score       -> su score no supera el umbral
ReasonKind = Literal["iou", "inner_contained", "inner_container", "low_score"]


@dataclass(frozen=True)
class Kill:
    """Un motivo de descarte: la regla, su valor medido y la máscara culpable (None si low_score)."""
    kind: ReasonKind
    value: float
    killer_index: Optional[int]


@dataclass(frozen=True)
class MaskVerdict:
    """Veredicto de una máscara: sus datos, si sobrevive y, si no, por qué."""
    index: int          # índice en el orden de entrada
    area: int
    score: float
    kept: bool
    kills: tuple[Kill, ...] = ()


@dataclass(frozen=True)
class DecisionBreakdown:
    """Resultado del NMS: un veredicto por máscara + los umbrales usados."""
    verdicts: tuple[MaskVerdict, ...]
    iou_thr: float
    score_thr: float
    inner_thr: float

    @property
    def kept(self) -> tuple[MaskVerdict, ...]:
        return tuple(v for v in self.verdicts if v.kept)

    @property
    def removed(self) -> tuple[MaskVerdict, ...]:
        return tuple(v for v in self.verdicts if not v.kept)


def scores_from_records(records: Sequence[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extrae (masks NHW bool, scores N) de la lista de dicts que devuelve un AMG (SAM2/SAM3)."""
    masks = torch.from_numpy(np.stack([r["segmentation"] for r in records], axis=0)).bool()
    stability = np.asarray([r["stability_score"] for r in records], dtype=np.float32)
    pred_iou = np.asarray([r["predicted_iou"] for r in records], dtype=np.float32)
    scores = torch.from_numpy(stability * pred_iou)
    return masks, scores


def evaluate(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.8,
    score_thr: float = 0.7,
    inner_thr: float = 0.5,
) -> DecisionBreakdown:
    """masks (N,H,W) bool + scores (N,) -> DecisionBreakdown. Mismo keep-set que ovo.mask_nms."""
    scores_sorted, idx = scores.sort(0, descending=True)
    n = idx.shape[0]
    order = idx.view(-1)
    masks_ord = masks[order]
    area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_mat = torch.zeros((n, n), dtype=torch.float)
    inner_mat = torch.zeros((n, n), dtype=torch.float)
    for i in range(n):
        for j in range(i, n):
            inter = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou_mat[i, j] = inter / union if union > 0 else 0.0
            r_i = inter / area[i] if area[i] > 0 else 0.0
            r_j = inter / area[j] if area[j] > 0 else 0.0
            if r_i < 0.5 and r_j >= 0.85:
                inner_mat[i, j] = 1 - r_j * r_i
            if r_i >= 0.85 and r_j < 0.5:
                inner_mat[j, i] = 1 - r_i * r_j

    iou_u = torch.triu(iou_mat, diagonal=1)
    iou_max, iou_arg = iou_u.max(dim=0)
    inner_u = torch.triu(inner_mat, diagonal=1)
    inner_u_max, inner_u_arg = inner_u.max(dim=0)
    inner_l = torch.tril(inner_mat, diagonal=1)
    inner_l_max, inner_l_arg = inner_l.max(dim=0)

    keep_iou = iou_max <= iou_thr
    keep_conf = scores_sorted.squeeze(-1) > score_thr if scores_sorted.ndim > 1 else scores_sorted > score_thr
    keep_inner_u = inner_u_max <= 1 - inner_thr
    keep_inner_l = inner_l_max <= 1 - inner_thr

    # Red de seguridad de mask_nms: si un criterio deja 0, rescata las top-3 por score.
    top3 = scores_sorted.reshape(-1).topk(min(3, n)).indices
    if keep_conf.sum() == 0:
        keep_conf[top3] = True
    if keep_inner_u.sum() == 0:
        keep_inner_u[top3] = True
    if keep_inner_l.sum() == 0:
        keep_inner_l[top3] = True

    keep = keep_iou & keep_conf & keep_inner_u & keep_inner_l

    verdicts: list[MaskVerdict] = []
    for p in range(n):
        orig = int(idx[p].item())
        kept = bool(keep[p].item())
        kills: list[Kill] = []
        if not kept:
            if not keep_iou[p]:
                kills.append(Kill("iou", float(iou_max[p]), int(idx[iou_arg[p]].item())))
            if not keep_inner_u[p]:
                kills.append(Kill("inner_contained", float(inner_u_max[p]), int(idx[inner_u_arg[p]].item())))
            if not keep_inner_l[p]:
                kills.append(Kill("inner_container", float(inner_l_max[p]), int(idx[inner_l_arg[p]].item())))
            if not keep_conf[p]:
                kills.append(Kill("low_score", float(scores_sorted.reshape(-1)[p]), None))
        verdicts.append(MaskVerdict(
            index=orig,
            area=int(area[p].item()),
            score=float(scores_sorted.reshape(-1)[p].item()),
            kept=kept,
            kills=tuple(kills),
        ))

    # Orden de salida = orden de entrada, para que index case con la lista original de máscaras.
    verdicts.sort(key=lambda v: v.index)
    return DecisionBreakdown(tuple(verdicts), iou_thr, score_thr, inner_thr)
