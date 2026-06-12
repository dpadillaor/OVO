"""
OVO post-processing pipeline visualized step by step.

Output (6 images):
  00_original.png          - Frame original
  01_sam2_raw.png          - 23 máscaras SAM2 con IDs
  02_masks_update.png      - 19 OK (color) + 4 eliminadas (✗ gris)
  03_solo_removidas.png    - Solo las 4 máscaras eliminadas
  04_mask2segmap.png       - 19 máscaras finales con solapamientos resueltos
  05_pipeline.png          - Tríptico: raw | masks_update | segmap

Usage:
    conda run -n ovo2 python segmentation_exp/debug_sam_pipeline.py
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

from ovo.utils.segment_utils import masks_update, mask2segmap, load_sam, mask_nms

# ─── Config ─────────────────────────────────────────────────────────────
FRAME_ID = 70
DATASET_PATH = os.path.join(PROJECT_ROOT, "data/input/Datasets/Replica/office0")
SAM_CKPT = os.path.join(PROJECT_ROOT, "data/input/sam_ckpts/sam2.1_hiera_large.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

H, W = 680, 1200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 1. Load image ─────────────────────────────────────────────────────
frame_path = os.path.join(DATASET_PATH, "results", f"frame{FRAME_ID:06d}.jpg")
color_data = cv2.imread(frame_path)
color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
image_rgb = cv2.cvtColor(color_data.astype(np.uint8), cv2.COLOR_BGR2RGB)
print(f"[1] Loaded frame {FRAME_ID}")


# ─── 2. Run SAM2 (stock, crop_n_layers=0) ──────────────────────────────
sam_config = {
    "sam_version": "2.1", "sam_encoder": "hiera_l",
    "sam_ckpt_path": os.path.dirname(SAM_CKPT),
    "points_per_side": 16, "crop_n_layers": 0,
    "nms_iou_th": 0.8, "stability_score_th": 0.95,
    "min_mask_region_area": 0, "use_m2m": False,
}
mask_generator = load_sam(sam_config, device=DEVICE)
with torch.no_grad():
    mask_generator.generate(np.random.rand(512, 512, 3).astype(np.uint8))

with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
    masks_raw = mask_generator.generate(image_rgb)

masks_sorted = sorted(masks_raw, key=lambda x: x['stability_score'], reverse=True)
print(f"[2] SAM2: {len(masks_sorted)} masks")


# ─── 3. Apply masks_update (OVO NMS) ───────────────────────────────────
masks_updated, = masks_update(masks_sorted, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)

masks_tensor = torch.from_numpy(np.stack([m['segmentation'] for m in masks_sorted], axis=0))
scores = torch.from_numpy(np.stack([m['stability_score'] * m['predicted_iou'] for m in masks_sorted], axis=0))
I = mask_nms(masks_tensor, scores, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
kept = set(I.tolist())
removed = set(range(len(masks_sorted))) - kept
print(f"[3] masks_update: {len(kept)} kept, {len(removed)} removed → IDs: {sorted(removed)}")


# ─── 4. Apply mask2segmap ──────────────────────────────────────────────
seg_map, binary_maps = mask2segmap(masks_updated, image_rgb, sort=True)
# Also apply mask2segmap to raw to compare
seg_map_raw, binary_maps_raw = mask2segmap(masks_sorted, image_rgb, sort=True)
print(f"[4] mask2segmap: {binary_maps.shape[0]} final layers")


# ─── Helper: draw masks with numbered IDs ──────────────────────────────
def _numbered_img(image, masks_list, highlight_removed=None):
    H, W = image.shape[:2]
    n = len(masks_list)
    colors = (plt.cm.tab20(np.linspace(0, 1, max(n, 1)))[:, :3] * 255).astype(np.uint8)

    img = image.copy().astype(np.float32) * 0.3
    for i, m in enumerate(masks_list):
        seg = m["segmentation"] if isinstance(m, dict) else m
        if seg.dtype != bool:
            seg = seg > 0
        if highlight_removed is not None and i in highlight_removed:
            for c in range(3):
                img[seg, c] = img[seg, c] * 0.3 + np.array([200, 50, 50], dtype=np.float32)[c] * 0.1
        else:
            color = colors[i % len(colors)]
            for c in range(3):
                img[seg, c] = img[seg, c] * 0.7 + color[c] * 0.3
    for i, m in enumerate(masks_list):
        seg = m["segmentation"] if isinstance(m, dict) else m
        if seg.dtype != bool:
            seg = seg > 0
        kernel = np.ones((3, 3), np.uint8)
        border = seg.astype(np.uint8) - cv2.erode(seg.astype(np.uint8), kernel)
        img[border > 0] = 0

    pil = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(18, H//35))
    except:
        font = ImageFont.load_default()
    for i, m in enumerate(masks_list):
        seg = m["segmentation"] if isinstance(m, dict) else m
        ys, xs = np.where(seg)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        label = "✗" if (highlight_removed is not None and i in highlight_removed) else str(i)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.ellipse([cx-tw//2-4, cy-th//2-4, cx+tw//2+4, cy+th//2+4], fill=(0,0,0))
        draw.text((cx-tw//2, cy-th//2), label, fill=(255,255,255), font=font)
    return np.array(pil)


def _segmap_numbered(image, binary_maps):
    H, W = image.shape[:2]
    n = binary_maps.shape[0]
    colors = (plt.cm.tab20(np.linspace(0, 1, max(n, 1)))[:, :3] * 255).astype(np.uint8)
    img = image.copy().astype(np.float32) * 0.3
    for i in range(n):
        seg = binary_maps[i]
        color = colors[i % len(colors)]
        for c in range(3):
            img[seg, c] = img[seg, c] * 0.7 + color[c] * 0.3
    for i in range(n):
        seg = binary_maps[i]
        kernel = np.ones((3, 3), np.uint8)
        border = seg.astype(np.uint8) - cv2.erode(seg.astype(np.uint8), kernel)
        img[border > 0] = 0
    pil = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(18, H//35))
    except:
        font = ImageFont.load_default()
    for i in range(n):
        seg = binary_maps[i]
        ys, xs = np.where(seg)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        label = str(i)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.ellipse([cx-tw//2-4, cy-th//2-4, cx+tw//2+4, cy+th//2+4], fill=(0,0,0))
        draw.text((cx-tw//2, cy-th//2), label, fill=(255,255,255), font=font)
    return np.array(pil)


# ─── 5. Generate images ────────────────────────────────────────────────
SAVE = lambda name: os.path.join(OUTPUT_DIR, name)

# 00: Original
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image_rgb); ax.set_title(f"Frame {FRAME_ID} - original", fontsize=13); ax.axis("off")
plt.tight_layout(); plt.savefig(SAVE("00_original.png"), dpi=150, bbox_inches="tight"); plt.close()

# 01: SAM2 raw - 23 masks
img01 = _numbered_img(image_rgb, masks_sorted)
fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(img01); ax.set_title(f"SAM2 raw: {len(masks_sorted)} máscaras", fontsize=14); ax.axis("off")
plt.tight_layout(); plt.savefig(SAVE("01_sam2_raw.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved 01_sam2_raw.png")

# 02: After masks_update - 19 kept (color) + 4 removed (✗)
img02 = _numbered_img(image_rgb, masks_sorted, highlight_removed=removed)
fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(img02)
ax.set_title(f"masks_update: {len(kept)} OK + {len(removed)} eliminadas ✗", fontsize=14)
ax.axis("off")
plt.tight_layout(); plt.savefig(SAVE("02_masks_update.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved 02_masks_update.png")

# 03: Only the removed masks
removed_masks = [masks_sorted[i] for i in sorted(removed)]
img03 = _numbered_img(image_rgb, removed_masks)
fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(img03); ax.set_title(f"Solo las {len(removed_masks)} eliminadas (IDs {sorted(removed)})", fontsize=14); ax.axis("off")
plt.tight_layout(); plt.savefig(SAVE("03_solo_removidas.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved 03_solo_removidas.png")

# 04: After mask2segmap - final 19 layers (overlaps resolved)
img04 = _segmap_numbered(image_rgb, binary_maps)
fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(img04); ax.set_title(f"mask2segmap: {binary_maps.shape[0]} capas finales (solapamientos resueltos)", fontsize=14); ax.axis("off")
plt.tight_layout(); plt.savefig(SAVE("04_mask2segmap.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved 04_mask2segmap.png")

# 05: Pipeline triptych
fig, axes = plt.subplots(1, 3, figsize=(32, 9))
titles = [
    f"1) SAM2 raw ({len(masks_sorted)})",
    f"2) masks_update ({len(kept)} kept, ✗={len(removed)})",
    f"3) mask2segmap ({binary_maps.shape[0]} capas)"
]
for ax, img, title in zip(axes, [img01, img02, img04], titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
plt.tight_layout(); plt.savefig(SAVE("05_pipeline.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved 05_pipeline.png")


print(f"\nDone. {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")
