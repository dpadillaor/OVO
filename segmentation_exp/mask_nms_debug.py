import os, sys, torch
sys.path.insert(0, '/home/padidavid/repos/OVO')
sys.path.insert(0, '/home/padidavid/repos/OVO/thirdParty/segment-anything-2')
import cv2, numpy as np
from ovo.utils.segment_utils import load_sam

PROJECT_ROOT = '/home/padidavid/repos/OVO'
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data/input/Datasets/Replica/office0')
SAM_CKPT = os.path.join(PROJECT_ROOT, 'data/input/sam_ckpts/sam2.1_hiera_large.pt')
H, W = 680, 1200
OUTPUT = os.path.join(PROJECT_ROOT, 'segmentation_exp/output')
frame_path = os.path.join(DATASET_PATH, 'results', f'frame{70:06d}.jpg')
color_data = cv2.imread(frame_path)
color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
image_rgb = cv2.cvtColor(color_data.astype(np.uint8), cv2.COLOR_BGR2RGB)

sam_config = {'sam_version':'2.1','sam_encoder':'hiera_l','sam_ckpt_path':os.path.dirname(SAM_CKPT),
    'points_per_side':16,'crop_n_layers':0,'nms_iou_th':0.8,'stability_score_th':0.95,
    'min_mask_region_area':0,'use_m2m':False}
mg = load_sam(sam_config, device='cuda')
with torch.no_grad(): mg.generate(np.random.rand(512,512,3).astype(np.uint8))
with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    masks_raw = mg.generate(image_rgb)

masks_sorted = sorted(masks_raw, key=lambda x: x['stability_score'], reverse=True)

# ─── Replicar mask_nms EXACTAMENTE, paso a paso ─────────────────────
masks_t = torch.from_numpy(np.stack([m['segmentation'] for m in masks_sorted], axis=0))
scores = torch.from_numpy(np.stack([m['stability_score']*m['predicted_iou'] for m in masks_sorted], axis=0))

print("="*80)
print("PASO A PASO: mask_nms(masks, scores, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)")
print("="*80)

# Step 1: sort by score descending
scores_s, idx = scores.sort(0, descending=True)
num_masks = idx.shape[0]
masks_ord = masks_t[idx.view(-1), :]
masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

print(f"\n1. Ordenado por score descendente:")
print(f"   {'pos':>3} | {'orig_id':>7} | {'score':>8} | {'area':>7}")
print("   " + "-"*35)
for pos in range(num_masks):
    print(f"   {pos:>3} | {idx[pos].item():>7} | {scores_s[pos].item():>8.4f} | {masks_area[pos].item():>7.0f}")

# Step 2: Build IoU matrix and inner matrix
iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float)
inner_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float)

print(f"\n2. Matriz de pares (solo mostramos relevantes para pos=5):")
# Focus on position 5 (orig ID 4)
target_pos = 5
target_orig = idx[target_pos].item()
print(f"\n   Pares donde interviene pos={target_pos} (orig ID {target_orig}):")
found_any = False
for i in range(target_pos + 1):  # only need i <= target_pos
    for j in range(max(i, target_pos), num_masks):
        if i == j:
            continue
        if i != target_pos and j != target_pos:
            continue
        inter = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
        union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
        iou = inter / union if union > 0 else 0
        iou_matrix[i, j] = iou
        if masks_area[i] > 0 and masks_area[j] > 0:
            r_i = inter / masks_area[i]
            r_j = inter / masks_area[j]
            if r_i < 0.5 and r_j >= 0.85:
                inner_matrix[i, j] = 1 - r_j * r_i
            if r_i >= 0.85 and r_j < 0.5:
                inner_matrix[j, i] = 1 - r_i * r_j
        if iou > 0 or inner_matrix[i,j] > 0 or inner_matrix[j,i] > 0:
            found_any = True
            print(f"     pair ({i},{j}): IoU={iou:.4f}, inner[{i},{j}]={inner_matrix[i,j]:.4f}, inner[{j},{i}]={inner_matrix[j,i]:.4f}, "
                  f"area_i={masks_area[i].item():.0f}, area_j={masks_area[j].item():.0f}")

if not found_any:
    print(f"     ¡NINGÚN par con pos {target_pos} tiene IoU>0 ni inner>0!")

# Step 3: Apply criteria
iou_matrix.triu_(diagonal=1)
iou_max, _ = iou_matrix.max(dim=0)

inner_iou_matrix_u = torch.triu(inner_matrix, diagonal=1)
inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
inner_iou_matrix_l = torch.tril(inner_matrix, diagonal=1)
inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

keep_iou = iou_max <= 0.8
keep_conf = scores_s.squeeze() > 0.7
keep_inner_u = inner_iou_max_u <= 1 - 0.5
keep_inner_l = inner_iou_max_l <= 1 - 0.5

# Safety nets
if keep_conf.sum() == 0:
    index = scores_s.topk(3).indices
    keep_conf[index, 0] = True
if keep_inner_u.sum() == 0:
    index = scores_s.topk(3).indices
    keep_inner_u[index, 0] = True
if keep_inner_l.sum() == 0:
    index = scores_s.topk(3).indices
    keep_inner_l[index, 0] = True

keep = keep_iou & keep_conf & keep_inner_u & keep_inner_l
selected_idx = idx[keep]

# Print criteria for ALL positions
print(f"\n3. Criterios para TODAS las posiciones:")
print(f"   {'pos':>3} | {'orig':>3} | {'iou_max':>7} | {'score':>7} | {'inner_u':>7} | {'inner_l':>7} | {'keep_iou':>8} | {'keep_sc':>7} | {'keep_iu':>7} | {'keep_il':>7} | {'KEEP':>4}")
print("   " + "-"*95)
for pos in range(num_masks):
    print(f"   {pos:>3} | {idx[pos].item():>3} | {iou_max[pos].item():>7.4f} | {scores_s[pos].item():>7.4f} | "
          f"{inner_iou_max_u[pos].item():>7.4f} | {inner_iou_max_l[pos].item():>7.4f} | "
          f"{str(keep_iou[pos].item()):>8} | {str(keep_conf[pos].item()):>7} | "
          f"{str(keep_inner_u[pos].item()):>7} | {str(keep_inner_l[pos].item()):>7} | "
          f"{'KEEP' if keep[pos].item() else 'REMOV':>4}")

print(f"\n4. Resultado final: {len(selected_idx)} máscaras kept")
print(f"   Kept orig IDs: {sorted(selected_idx.tolist())}")
removed = [i for i in idx.tolist() if i not in selected_idx.tolist()]
print(f"   Removed orig IDs: {sorted(removed)}")
print(f"   ¿ID 4 (orig) removida? {4 in removed}")
