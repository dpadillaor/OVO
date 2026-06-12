"""
Compare raw SAM2 output from:
  A) Stock SAM2AutomaticMaskGenerator (what OVO uses internally)
  B) SAM2AutomaticMaskGeneratorModified (user's script)

NO postprocessing. Just generate() -> compare.

Usage:
    conda run -n ovo2 python segmentation_exp/compare_generators.py
"""

import os, sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "thirdParty", "segment-anything-2"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Import user's modified generator
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "sam2_exploration"))
from sam2_amg_modified import SAM2AutomaticMaskGeneratorModified

# ─── Config ─────────────────────────────────────────────────────────────
FRAME_ID = 70
DATASET_PATH = os.path.join(PROJECT_ROOT, "data/input/Datasets/Replica/office0")
SAM_CKPT = os.path.join(PROJECT_ROOT, "data/input/sam_ckpts/sam2.1_hiera_large.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

H, W = 680, 1200
SAM_ENCODER = "hiera_l"
SAM_VERSION = "2.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. Load image (exactly as OVO) ────────────────────────────────────
frame_path = os.path.join(DATASET_PATH, "results", f"frame{FRAME_ID:06d}.jpg")
color_data = cv2.imread(frame_path)
color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
color_data = color_data.astype(np.uint8)
image_rgb = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
print(f"[1] Loaded frame {FRAME_ID}: {image_rgb.shape}")


# ─── 2. Build model once, share between both generators ─────────────────
model_cfg = os.path.join("configs", f"sam{SAM_VERSION}", f"sam{SAM_VERSION}_{SAM_ENCODER}.yaml")
sam_model = build_sam2(model_cfg, SAM_CKPT, device=DEVICE, mode="eval", apply_postprocessing=False)
sam_model.eval()
print(f"[2] SAM2 model loaded on {DEVICE}")


# ─── 3. Create both generators with IDENTICAL params ────────────────────
# Using the same points_per_side=16 as OVO config
common_params = dict(
    points_per_side=16,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    box_nms_thresh=0.7,
    crop_n_layers=0,
    min_mask_region_area=0,
    use_m2m=False,
)

gen_a = SAM2AutomaticMaskGenerator(model=sam_model, **common_params)
gen_b = SAM2AutomaticMaskGeneratorModified(model=sam_model, **common_params)

print("[3] Both generators created with identical params:")
for k, v in common_params.items():
    print(f"    {k}: {v}")


# ─── 4. Warmup ──────────────────────────────────────────────────────────
with torch.no_grad():
    _ = gen_a.generate(np.random.rand(512, 512, 3).astype(np.float32))
    _ = gen_b.generate(np.random.rand(512, 512, 3).astype(np.float32))
print("[4] Warmup done")


# ─── 5. Generate masks ──────────────────────────────────────────────────
with torch.inference_mode():
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        masks_a = gen_a.generate(image_rgb)
        masks_b = gen_b.generate(image_rgb)

print(f"\n[5] Results:")
print(f"    A) Stock SAM2 AMG:           {len(masks_a)} masks")
print(f"    B) Modified SAM2 AMG:        {len(masks_b)} masks")

# Check if they differ
if len(masks_a) != len(masks_b):
    print(f"    *** DIFFERENT mask counts! ***")

# Check if masks are identical pixel-by-pixel
# Both sorted by stability_score
masks_a_sorted = sorted(masks_a, key=lambda x: x['stability_score'], reverse=True)
masks_b_sorted = sorted(masks_b, key=lambda x: x['stability_score'], reverse=True)

if len(masks_a_sorted) == len(masks_b_sorted):
    diffs = 0
    for i, (ma, mb) in enumerate(zip(masks_a_sorted, masks_b_sorted)):
        if not np.array_equal(ma['segmentation'], mb['segmentation']):
            diffs += 1
    if diffs == 0:
        print(f"    ✓ All masks identical pixel-by-pixel!")
    else:
        print(f"    ✗ {diffs}/{len(masks_a_sorted)} masks differ")
else:
    print(f"    Can't pixel-compare: different counts")


# ─── 6. Visualize ──────────────────────────────────────────────────────
def draw_masks(image, masks_list, title=""):
    import matplotlib.font_manager as fm
    img = image.copy().astype(np.float32)
    H, W = img.shape[:2]
    colors = (plt.cm.tab20(np.linspace(0, 1, max(len(masks_list), 1)))[:, :3] * 255).astype(np.uint8)
    for i, m in enumerate(masks_list):
        seg = m['segmentation']
        color = colors[i % len(colors)]
        for c in range(3):
            img[seg, c] = img[seg, c] * 0.5 + color[c] * 0.5
        kernel = np.ones((3, 3), np.uint8)
        border = seg.astype(np.uint8) - cv2.erode(seg.astype(np.uint8), kernel)
        img[border > 0] = 0
    img_out = img.astype(np.uint8)

    # Add numbered labels at centroid of each mask
    # Need to convert to PIL for drawing text with good rendering
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(img_out)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(16, H//40))
    except:
        font = ImageFont.load_default()

    for i, m in enumerate(masks_list):
        seg = m['segmentation']
        ys, xs = np.where(seg)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        # Draw white background circle for readability
        bbox = draw.textbbox((0, 0), str(i), font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.ellipse([cx-tw//2-3, cy-th//2-3, cx+tw//2+3, cy+th//2+3], fill=(0,0,0))
        draw.text((cx-tw//2, cy-th//2), str(i), fill=(255,255,255), font=font)

    return np.array(pil_img)

# Side-by-side only if counts match, otherwise show both individually
if len(masks_a) == len(masks_b):
    img_a = draw_masks(image_rgb, masks_a_sorted)
    img_b = draw_masks(image_rgb, masks_b_sorted)

    fig, axes = plt.subplots(1, 3, figsize=(32, 8))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(img_a)
    axes[1].set_title(f"A) Stock SAM2 AMG: {len(masks_a)} masks", fontsize=13)
    axes[1].axis("off")
    axes[2].imshow(img_b)
    axes[2].set_title(f"B) Modified SAM2 AMG: {len(masks_b)} masks", fontsize=13)
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "compare_generators.png"), dpi=150)
    plt.close()
    print(f"\n[6] Saved compare_generators.png")

else:
    img_a = draw_masks(image_rgb, masks_a_sorted)
    img_b = draw_masks(image_rgb, masks_b_sorted)

    fig, axes = plt.subplots(1, 3, figsize=(32, 8))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(img_a)
    axes[1].set_title(f"A) Stock SAM2: {len(masks_a)} masks", fontsize=13)
    axes[1].axis("off")
    axes[2].imshow(img_b)
    axes[2].set_title(f"B) Modified: {len(masks_b)} masks", fontsize=13)
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "compare_generators.png"), dpi=150)
    plt.close()
    print(f"\n[6] Saved compare_generators.png (different counts)")

    # Also show difference overlay
    print(f"\n    A) masks sorted by stability:")
    for i, m in enumerate(masks_a_sorted):
        print(f"       [{i}] area={m['segmentation'].sum():>8}, stability={m['stability_score']:.4f}, iou={m['predicted_iou']:.4f}")
    print(f"\n    B) masks sorted by stability:")
    for i, m in enumerate(masks_b_sorted):
        print(f"       [{i}] area={m['segmentation'].sum():>8}, stability={m['stability_score']:.4f}, iou={m['predicted_iou']:.4f}")
