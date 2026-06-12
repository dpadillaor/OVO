"""
Para cada máscara eliminada por masks_update, muestra:
  - Contorno rojo: máscara eliminada
  - Contorno azul: máscara(s) que causan la eliminación
  - Relleno amarillo: zona de solapamiento
  - Métricas: IoU, inner overlap

Output: images/06_detalle_eliminada_ID.png
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


# ─── 1. Load image ─────────────────────────────────────────────────────
frame_path = os.path.join(DATASET_PATH, "results", f"frame{FRAME_ID:06d}.jpg")
color_data = cv2.imread(frame_path)
color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
image_rgb = cv2.cvtColor(color_data.astype(np.uint8), cv2.COLOR_BGR2RGB)

# ─── 2. Run SAM2 ───────────────────────────────────────────────────────
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

# ─── 3. Compute mask_nms and get kept/removed ──────────────────────────
masks_tensor = torch.from_numpy(np.stack([m['segmentation'] for m in masks_sorted], axis=0))
scores = torch.from_numpy(np.stack([m['stability_score'] * m['predicted_iou'] for m in masks_sorted], axis=0))

I = mask_nms(masks_tensor, scores, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
kept = set(I.tolist())
removed = sorted(set(range(len(masks_sorted))) - kept)

# ─── 4. For each removed mask, find WHY ────────────────────────────────
# Build the same matrices as mask_nms
scores_s, idx_s = scores.sort(0, descending=True)
masks_ord = masks_tensor[idx_s.view(-1)]
masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)
num_masks = len(masks_sorted)

# Find for each removed mask which kept mask triggered it
def find_killer(removed_idx):
    """Return list of (killer_original_idx, reason, iou, inner)"""
    ri = removed_idx
    # Get position in sorted order
    sorted_pos = (idx_s == ri).nonzero(as_tuple=True)[0].item()
    results = []
    for j in range(num_masks):
        kj = idx_s[j].item()  # original index
        if kj == ri or kj not in kept:
            continue
        intersection = torch.logical_and(masks_tensor[ri], masks_tensor[kj]).sum().float()
        union = torch.logical_or(masks_tensor[ri], masks_tensor[kj]).sum().float()
        iou = (intersection / union).item() if union > 0 else 0

        inner_val = None
        area_ri = masks_tensor[ri].sum().float()
        area_kj = masks_tensor[kj].sum().float()
        intersect_ratio_ri = intersection / area_ri if area_ri > 0 else 0
        intersect_ratio_kj = intersection / area_kj if area_kj > 0 else 0

        reason = None
        if iou > 0.8:
            reason = "iou"
        elif intersect_ratio_ri >= 0.85 and intersect_ratio_kj < 0.5:
            inner_val = (1 - intersect_ratio_ri * intersect_ratio_kj).item()
            if inner_val <= 0.5:
                reason = "inner_contained"
        elif intersect_ratio_kj >= 0.85 and intersect_ratio_ri < 0.5:
            inner_val = (1 - intersect_ratio_ri * intersect_ratio_kj).item()
            if inner_val <= 0.5:
                reason = "inner_contains"

        if reason is not None:
            results.append((kj, reason, iou, inner_val, intersection.item(),
                            intersect_ratio_ri.item(), intersect_ratio_kj.item()))
    return results


# ─── 5. Draw detailed view per removed mask ────────────────────────────
def draw_removed_detail(image, masks_list, removed_i, killers, idx_to_i):
    """
    Draw:
      - Full image with:
        - Removed mask in red outline
        - Killer mask(s) in blue outline  
        - Overlap in yellow
      - Inset zoom of the overlap region
    """
    H, W = image.shape[:2]
    rem_mask = masks_list[removed_i]['segmentation']
    if rem_mask.dtype != bool:
        rem_mask = rem_mask > 0

    # Start with a copy
    img = image.copy().astype(np.float32)

    # Draw masks: removed in red overlay, killers in blue overlay
    overlap_all = np.zeros_like(rem_mask)
    for kj, reason, iou, inner_val, inter_size, ratio_ri, ratio_kj in killers:
        killer_mask = masks_list[kj]['segmentation']
        if killer_mask.dtype != bool:
            killer_mask = killer_mask > 0
        # Overlap between this killer and removed
        overlap = rem_mask & killer_mask
        overlap_all |= overlap

        # Blue tint for killer
        for c in range(3):
            img[killer_mask & ~overlap, c] = img[killer_mask & ~overlap, c] * 0.5 + np.array([50, 50, 200])[c] * 0.5

    # Red tint for removed (non-overlapping parts)
    for c in range(3):
        img[rem_mask & ~overlap_all, c] = img[rem_mask & ~overlap_all, c] * 0.5 + np.array([200, 50, 50])[c] * 0.5

    # Yellow for overlap
    for c in range(3):
        img[overlap_all, c] = img[overlap_all, c] * 0.3 + np.array([255, 255, 50])[c] * 0.7

    # Borders
    kernel = np.ones((3, 3), np.uint8)
    for label, mask, color in [("removed", rem_mask, (0,0,255)), ("killer", overlap_all, (255,0,0))]:
        border = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), kernel)
        if label == "removed":
            img[border > 0] = np.array([255, 0, 0], dtype=np.float32)
        elif label == "killer" and False:  # don't draw killer border separately
            pass
    # Overlap border
    overlap_border = overlap_all.astype(np.uint8) - cv2.erode(overlap_all.astype(np.uint8), kernel)
    img[overlap_border > 0] = np.array([255, 255, 0], dtype=np.float32)

    # Also draw killer border in blue
    for kj, reason, iou, inner_val, inter_size, ratio_ri, ratio_kj in killers:
        killer_mask = masks_list[kj]['segmentation']
        if killer_mask.dtype != bool:
            killer_mask = killer_mask > 0
        killer_border = killer_mask.astype(np.uint8) - cv2.erode(killer_mask.astype(np.uint8), kernel)
        # Only where not overlapping
        non_overlap_border = killer_border & ~overlap_all.astype(np.uint8)
        img[non_overlap_border > 0] = np.array([0, 100, 255], dtype=np.float32)

    img_out = img.astype(np.uint8)

    # Add labels with PIL
    pil = Image.fromarray(img_out)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(14, H//45))
    except:
        font = ImageFont.load_default()

    # Label removed mask at its centroid
    ys, xs = np.where(rem_mask)
    if len(ys) > 0:
        cy, cx = int(ys.mean()), int(xs.mean())
        label = f"ID {removed_i} (removed)"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([cx-tw//2-5, cy-th//2-5, cx+tw//2+5, cy+th//2+5], fill=(200,0,0))
        draw.text((cx-tw//2, cy-th//2), label, fill=(255,255,255), font=font)

    # Label killers
    for kj, reason, iou, inner_val, inter_size, ratio_ri, ratio_kj in killers:
        killer_mask = masks_list[kj]['segmentation']
        if killer_mask.dtype != bool:
            killer_mask = killer_mask > 0
        ys, xs = np.where(killer_mask)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        # Offset label so it doesn't overlap with the removed label
        label = f"ID {kj} (killer)"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([cx-tw//2-5, cy-th//2-5, cx+tw//2+5, cy+th//2+5], fill=(0,0,200))
        draw.text((cx-tw//2, cy-th//2), label, fill=(255,255,255), font=font)

    np_img = np.array(pil)

    # Also create a zoomed inset showing just the overlap region
    ys, xs = np.where(overlap_all)
    if len(ys) > 0:
        y_min, y_max = max(0, ys.min()-30), min(H, ys.max()+30)
        x_min, x_max = max(0, xs.min()-30), min(W, xs.max()+30)
        zoom = np_img[y_min:y_max, x_min:x_max].copy()
        # Resize for visibility
        zoom_h, zoom_w = zoom.shape[:2]
        scale = min(300/zoom_w, 300/zoom_h)
        new_w, new_h = int(zoom_w*scale), int(zoom_h*scale)
        zoom = cv2.resize(zoom, (max(new_w, 1), max(new_h, 1)))

    # Build annotation text
    text_lines = [f"Máscara eliminada: ID {removed_i}"]
    for kj, reason, iou, inner_val, inter_size, ratio_ri, ratio_kj in killers:
        text_lines.append(f"  vs ID {kj}:")
        text_lines.append(f"    IoU = {iou:.3f}  (threshold 0.8)")
        text_lines.append(f"    intersección = {inter_size} px")
        text_lines.append(f"    % de removed dentro de killer = {ratio_ri*100:.1f}%")
        text_lines.append(f"    % de killer dentro de removed = {ratio_kj*100:.1f}%")
        if inner_val is not None:
            text_lines.append(f"    inner_score = {inner_val:.3f}  (threshold 0.5)")
        text_lines.append(f"    razón: {reason}")

    return np_img, text_lines


# Generate detail images
print("\nAnálisis de máscaras eliminadas:")
for ri in removed:
    killers = find_killer(ri)
    if not killers:
        # Check if it was removed by score_thr
        score_val = scores[ri].item()
        if score_val <= 0.7:
            print(f"\n  ID {ri}: eliminada por score_thr (score={score_val:.3f} ≤ 0.7)")
        else:
            # Find best matching kept mask even if below threshold
            best_match = None
            best_iou = 0
            for kj in kept:
                inter = torch.logical_and(masks_tensor[ri], masks_tensor[kj]).sum().float()
                union = torch.logical_or(masks_tensor[ri], masks_tensor[kj]).sum().float()
                iou_val = (inter / union).item() if union > 0 else 0
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_match = kj
            if best_match is not None:
                killers = [(best_match, "best_match", best_iou, None, 0, 0, 0)]
                print(f"\n  ID {ri}: eliminada (mejor match: ID {best_match}, IoU={best_iou:.3f})")
    else:
        print(f"\n  ID {ri}: {'; '.join([f'vs ID {kj} ({reason}, IoU={iou:.3f})' for kj, reason, iou, *_ in killers])}")

    img_det, text = draw_removed_detail(image_rgb, masks_sorted, ri, killers, None)

    # Save full detailed view
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.imshow(img_det)
    ax.set_title("\n".join(text[:3]), fontsize=12, loc="left")
    ax.axis("off")
    # Add annotation box at bottom
    plt.figtext(0.02, 0.02, "\n".join(text), fontsize=10,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
                color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"06_detalle_eliminada_{ri}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved 06_detalle_eliminada_{ri}.png")
