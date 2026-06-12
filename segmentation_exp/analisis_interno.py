"""
Para cada máscara eliminada, muestra qué y cómo la eliminó.
"""

import os, sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "thirdParty", "segment-anything-2"))

from ovo.utils.segment_utils import load_sam

FRAME_ID = 70
DATASET_PATH = os.path.join(PROJECT_ROOT, "data/input/Datasets/Replica/office0")
SAM_CKPT = os.path.join(PROJECT_ROOT, "data/input/sam_ckpts/sam2.1_hiera_large.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
H, W = 680, 1200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

frame_path = os.path.join(DATASET_PATH, "results", f"frame{FRAME_ID:06d}.jpg")
color_data = cv2.imread(frame_path)
color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
image_rgb = cv2.cvtColor(color_data.astype(np.uint8), cv2.COLOR_BGR2RGB)

sam_config = {"sam_version":"2.1","sam_encoder":"hiera_l","sam_ckpt_path":os.path.dirname(SAM_CKPT),
    "points_per_side":16,"crop_n_layers":0,"nms_iou_th":0.8,"stability_score_th":0.95,
    "min_mask_region_area":0,"use_m2m":False}
mask_generator = load_sam(sam_config, device=DEVICE)
with torch.no_grad():
    mask_generator.generate(np.random.rand(512,512,3).astype(np.uint8))
with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
    masks_raw = mask_generator.generate(image_rgb)

masks_sorted = sorted(masks_raw, key=lambda x: x['stability_score'], reverse=True)
masks_t = torch.from_numpy(np.stack([m['segmentation'] for m in masks_sorted], axis=0))
scores = torch.from_numpy(np.stack([m['stability_score']*m['predicted_iou'] for m in masks_sorted], axis=0))

# ─── Replicate mask_nms logic exactly ──────────────────────────────────
scores_s, idx_s = scores.sort(0, descending=True)
masks_ord = masks_t[idx_s.view(-1)]
masks_area = torch.sum(masks_ord, dim=(1,2), dtype=torch.float)
num = len(masks_sorted)

iou_mat = torch.zeros((num,num), dtype=torch.float)
inner_mat = torch.zeros((num,num), dtype=torch.float)
for i in range(num):
    for j in range(i, num):
        inter = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
        union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
        iou_mat[i,j] = inter / union if union > 0 else 0
        if inter / masks_area[i] < 0.5 and inter / masks_area[j] >= 0.85:
            inner_mat[i,j] = 1 - (inter / masks_area[j]) * (inter / masks_area[i])
        if inter / masks_area[i] >= 0.85 and inter / masks_area[j] < 0.5:
            inner_mat[j,i] = 1 - (inter / masks_area[i]) * (inter / masks_area[j])

# Apply criteria
iou_mat.triu_(diagonal=1)
iou_max, _ = iou_mat.max(dim=0)

inner_u = torch.triu(inner_mat, diagonal=1)
inner_max_u, _ = inner_u.max(dim=0)
inner_l = torch.tril(inner_mat, diagonal=0)
inner_max_l, _ = inner_l.max(dim=0)

keep_iou = iou_max <= 0.8
keep_score = scores_s.squeeze() > 0.7
keep_inner_u = inner_max_u <= 0.5
keep_inner_l = inner_max_l <= 0.5

# Safety: at least top 3
if keep_score.sum() == 0: keep_score[:3] = True
if keep_inner_u.sum() == 0: keep_inner_u[:3] = True
if keep_inner_l.sum() == 0: keep_inner_l[:3] = True

keep = keep_iou & keep_score & keep_inner_u & keep_inner_l
kept_sorted = set(keep.nonzero(as_tuple=True)[0].tolist())
removed_sorted = set(range(num)) - kept_sorted

# Map sorted position → original index
removed_orig = sorted([idx_s[sp].item() for sp in removed_sorted])
print(f"Kept (sorted pos): {sorted(kept_sorted)}")
print(f"Removed (sorted pos): {sorted(removed_sorted)} → orig IDs: {removed_orig}")


# Para cada removed, encontrar qué kept (con score mayor) lo elimina
print("\n--- Análisis por máscara eliminada ---")
for rp in sorted(removed_sorted):
    rid = idx_s[rp].item()
    reasons = []
    # Score check
    if scores_s[rp].item() <= 0.7:
        reasons.append(f"score_thr: score={scores_s[rp].item():.3f} <= 0.7")

    # IoU and inner with higher-scored masks
    for kp in sorted(kept_sorted):
        if kp >= rp:
            continue  # only higher-scored masks
        kid = idx_s[kp].item()

        if iou_mat[kp, rp] > 0.8:
            reasons.append(f"IoU={iou_mat[kp,rp]:.3f} con máscara #{kid} (orig ID {kid})")

        if inner_mat[kp, rp] > 0.5:
            # Lower-scored mask (rp) is inside higher-scored (kp)
            inter = torch.sum(torch.logical_and(masks_ord[kp], masks_ord[rp]), dtype=torch.float)
            pct_lower = (inter / masks_area[rp] * 100).item()
            pct_higher = (inter / masks_area[kp] * 100).item()
            reasons.append(f"inner={inner_mat[kp,rp]:.3f}: máscara #{kid} cubre {pct_lower:.0f}% de la eliminada, "
                          f"pero eliminada cubre solo {pct_higher:.0f}% de la #{kid}")

        if inner_mat[rp, kp] > 0.5:
            # Higher-scored mask (kp) is inside lower-scored (rp) - rare
            inter = torch.sum(torch.logical_and(masks_ord[kp], masks_ord[rp]), dtype=torch.float)
            pct_lower = (inter / masks_area[rp] * 100).item()
            pct_higher = (inter / masks_area[kp] * 100).item()
            reasons.append(f"inner={inner_mat[rp,kp]:.3f}: máscara #{kid} está dentro de la eliminada "
                          f"({pct_higher:.0f}% de la #{kid} solapada)")

    print(f"\n  Eliminada orig ID {rid} (sorted pos {rp}):")
    print(f"    area={masks_area[rp].item():.0f}, score={scores_s[rp].item():.3f}")
    if reasons:
        for r in reasons:
            print(f"    → {r}")
    else:
        print(f"    → ¡sin razón clara!")
