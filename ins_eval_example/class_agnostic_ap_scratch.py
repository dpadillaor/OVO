import pathlib
import numpy as np
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ins_eval_utils
import replica
import scannet200
from ovo.utils.io_utils import rle_decode


# ============================================================
# ANALYSIS PIPELINE — 3 steps
#
#   Step 1  --align   : Overlay pred PCL vs GT mesh vertices.
#                       Confirms coordinate-frame alignment before any matching.
#
#   Step 2  --match   : Side-by-side instance viewer (GT left, pred right).
#                       Navigate with N=next / P=prev (sorted best→worst IoU).
#                       Requires alignment to be confirmed first.
#
#   Step 3  --eval    : Compute class-agnostic AP metrics.
#                       Requires instances to be loaded & aligned.
#
# Optional:
#   --save-ply        : Export colored instance PLY files (GT + pred).
#   --save-png        : Save matplotlib projection of alignment (no GUI needed).
#   --clip-z FLOAT    : Remove ceiling/floor above this Z value.
# ============================================================


# ============================================================
# Step 1 helpers — alignment check
# ============================================================

EXPERIMENT_PATH = "data/output/Replica/20260310_GTNoise-T0p005-R0p01_SAM3_ComparativaPaper"
# EXPERIMENT_PATH = "data/output/Replica/20260305_GT_CLIP_ComparativaPaper"
# EXPERIMENT_PATH = "data/output/Replica/20260305_GT_CLIP_ComparativaPaper"

SCENE = "office0"
MESH_PATH = f"data/input/Datasets/Replica/{SCENE}_mesh.ply"


def _save_projection_png(pcd_pred, pcd_gt, out_path="alignment_check.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Subsample for speed
    rng = np.random.default_rng(0)
    idx_p = rng.choice(len(pcd_pred), min(50_000, len(pcd_pred)), replace=False)
    idx_g = rng.choice(len(pcd_gt),   min(50_000, len(pcd_gt)),   replace=False)
    p = pcd_pred[idx_p]
    g = pcd_gt[idx_g]

    projections = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, xi, yi) in zip(axes, projections):
        ax.scatter(g[:, xi], g[:, yi], s=0.1, c="green", alpha=0.3, label="GT")
        ax.scatter(p[:, xi], p[:, yi], s=0.1, c="red",   alpha=0.3, label="Pred")
        ax.set_title(name)
        ax.set_aspect("equal")
    axes[0].legend(markerscale=10)
    fig.suptitle("Pred (red) vs GT (green) — sampled 50k pts each")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved projection to: {os.path.abspath(out_path)}")


def visualize_alignment():
    import torch
    import open3d as o3d

    # Load predicted point cloud from checkpoint
    ckpt_path = os.path.join(EXPERIMENT_PATH, SCENE, "ovo_map.ckpt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pcd_pred = ckpt["map_params"]["xyz"].numpy()   # (N, 3)
    obj_ids = ckpt["map_params"]["obj_ids"][:, 0]  # (N,)
    print(f"Predicted cloud: {pcd_pred.shape[0]} points")
    print(f"  bbox: min={pcd_pred.min(axis=0)}, max={pcd_pred.max(axis=0)}")

    # Load GT mesh vertices
    print(f"Loading GT mesh: {MESH_PATH}")
    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    pcd_gt = np.asarray(mesh.vertices)             # (M, 3)
    print(f"GT cloud: {pcd_gt.shape[0]} vertices")
    print(f"  bbox: min={pcd_gt.min(axis=0)}, max={pcd_gt.max(axis=0)}")

    # Build Open3D point clouds
    pred_cloud = o3d.geometry.PointCloud()
    pred_cloud.points = o3d.utility.Vector3dVector(pcd_pred)
    pred_cloud.paint_uniform_color([1, 0, 0])  # red = predicted

    gt_cloud = o3d.geometry.PointCloud()
    gt_cloud.points = o3d.utility.Vector3dVector(pcd_gt)
    gt_cloud.paint_uniform_color([0, 0.7, 0])  # green = GT

    print("Opening Open3D viewer — red=predicted, green=GT")
    try:
        o3d.visualization.draw_geometries(
            [pred_cloud, gt_cloud],
            window_name="Pred (red) vs GT (green)",
        )
    except Exception as e:
        print(f"[INFO] GUI not available ({e}), saving matplotlib projection instead.")
        _save_projection_png(pcd_pred, pcd_gt)


# ============================================================
# Step 2 helpers — instance matching viewer (see inspect_instances below)
# ============================================================

# wall=0, ceiling=1, floor=2, window=8, door=10  (indices in class_names_reduced)
BACKGROUND_CLASS_IDS = {0, 1, 2, 8, 10}


def load_predictions(experiment_path, scene_name, no_background=False):
    pred_dir = pathlib.Path(experiment_path) / "instance_pred"
    pred_file = pred_dir / f"{scene_name}.txt"

    if not pred_file.exists():
        raise ValueError(f"Predictions file not found: {pred_file}")

    with open(pred_file, 'r') as f:
        lines = f.read().splitlines()

    bg_masks_list = []   # background instance masks (to compute bg point mask)
    masks_list = []
    classes_list = []
    scores_list = []

    for line in lines:
        parts = line.split()
        mask_file = parts[0]
        label = int(parts[1])
        conf = float(parts[2])

        mask_path = pred_dir / mask_file
        with open(mask_path, 'r') as f:
            rle = json.load(f)
        mask = rle_decode(rle)

        if no_background and label in BACKGROUND_CLASS_IDS:
            bg_masks_list.append(mask)
            continue

        masks_list.append(mask)
        classes_list.append(label)
        scores_list.append(conf)

    if no_background and bg_masks_list:
        # points the prediction considered background — remove from every remaining mask
        bg_pts = np.stack(bg_masks_list, axis=1).any(axis=1)  # (n_pts,) bool
        masks_list = [m & ~bg_pts for m in masks_list]
        print(f"  [--no-background] removed {len(bg_masks_list)} bg instances "
              f"({bg_pts.sum()} points masked out)")

    if len(masks_list) > 0:
        pred_masks = np.stack(masks_list, axis=1)
        pred_classes = np.array(classes_list)
        pred_scores = np.array(scores_list)
    else:
        pred_masks = np.zeros((0, 0))
        pred_classes = np.array([])
        pred_scores = np.array([])

    return {
        "pred_masks": pred_masks,
        "pred_classes": pred_classes,
        "pred_scores": pred_scores,
    }


def read_gt(gt_file, dataset_module=None):
    with open(gt_file, 'r') as f:
        gt_ids = np.array(f.read().splitlines(), dtype=np.int64)
    ids = np.unique(gt_ids)
    binary_masks = ids[None] == gt_ids[:, None]
    classes = ids // 1000
    if dataset_module:
        new_classes = np.ones_like(classes) * len(dataset_module.valid_ins_class_ids)
        for i, id in enumerate(dataset_module.valid_ins_class_ids):
            new_classes[classes == id] = i
        classes = new_classes
    return binary_masks, classes


def iou_diagnosis(pred_masks, gt_dir):
    """For each predicted instance, compute IoU with its best-matching GT instance."""
    with open(gt_dir / f"{SCENE}.txt") as f:
        gt_ids = np.array(f.read().splitlines(), dtype=np.int64)

    unique_gt = np.unique(gt_ids)
    gt_masks = (unique_gt[None] == gt_ids[:, None])  # (n_points, n_gt)

    n_pred = pred_masks.shape[1]
    best_ious = np.zeros(n_pred)
    for i in range(n_pred):
        p = pred_masks[:, i].astype(bool)
        if p.sum() == 0:
            continue
        inter = (p[:, None] & gt_masks).sum(axis=0)   # (n_gt,)
        union = (p[:, None] | gt_masks).sum(axis=0)   # (n_gt,)
        ious  = inter / np.maximum(union, 1)
        best_ious[i] = ious.max()

    thresholds = [0.1, 0.25, 0.5, 0.75]
    print(f"  GT instances:   {len(unique_gt)}")
    print(f"  Pred instances: {n_pred}")
    print(f"  Best-IoU distribution:")
    print(f"    mean={best_ious.mean():.3f}  median={np.median(best_ious):.3f}  max={best_ious.max():.3f}")
    for t in thresholds:
        count = (best_ious >= t).sum()
        print(f"    IoU >= {t:.2f}: {count}/{n_pred} predictions ({100*count/n_pred:.1f}%)")


def gt_oracle_sanity_check(gt_dir, scene):
    """Evaluate using GT masks directly — should give ~1.0 if eval code is correct."""
    with open(gt_dir / f"{scene}.txt") as f:
        gt_ids = np.array(f.read().splitlines(), dtype=np.int64)

    ids = np.unique(gt_ids)
    binary_masks = (ids[None] == gt_ids[:, None])  # (n_points, n_instances)

    predictions = {scene: {
        "pred_masks": binary_masks,
        "pred_classes": np.zeros(binary_masks.shape[1], dtype=int),
        "pred_scores": np.ones(binary_masks.shape[1]),
    }}

    metrics = ins_eval_utils.evaluate(
        predictions, replica, gt_path=gt_dir, class_agnostic=True
    )
    print(f"GT-oracle class-agnostic mAP: {metrics['AP']:.3f}  (expected ~1.0)")


def test_class_agnostic_ap(args):
    gt_dir = pathlib.Path(args.gt_path)
    if not gt_dir.exists():
        raise ValueError("Wrong path for ground_truth data")

    print(f"Loading predictions from: {EXPERIMENT_PATH}")
    predictions = {SCENE: load_predictions(EXPERIMENT_PATH, SCENE, no_background=args.no_background)}

    pred_masks = predictions[SCENE]['pred_masks']
    pred_classes = predictions[SCENE]['pred_classes']
    pred_scores = predictions[SCENE]['pred_scores']

    print(f"\npred_masks shape:   {pred_masks.shape}  dtype={pred_masks.dtype}  unique={np.unique(pred_masks)}")
    print(f"pred_classes shape: {pred_classes.shape}  unique={np.unique(pred_classes)}")
    print(f"pred_scores shape:  {pred_scores.shape}  range=[{pred_scores.min():.4f}, {pred_scores.max():.4f}]")

    print("\n--- GT-oracle sanity check ---")
    gt_oracle_sanity_check(gt_dir, SCENE)

    print("\n--- IoU distribution (pred vs best GT match) ---")
    iou_diagnosis(pred_masks, gt_dir)

    print("\n--- OVO predictions ---")
    metrics = ins_eval_utils.evaluate(
        predictions,
        replica,
        gt_path=gt_dir,
        class_agnostic=True,
    )
    print(f"class-agnostic mAP: {metrics['AP']:.3f}")


def save_instance_ply(points, instance_label_per_point, out_path, background_id=-1):
    """Save a point cloud as PLY with one random color per instance ID."""
    import torch
    unique_ids = np.unique(instance_label_per_point)
    rng = np.random.default_rng(42)
    color_map = {uid: rng.integers(50, 255, 3) for uid in unique_ids}
    if background_id in color_map:
        color_map[background_id] = np.array([40, 40, 40])  # dark grey for unlabelled

    colors = np.stack([color_map[l] for l in instance_label_per_point])

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved: {os.path.abspath(out_path)}")


def visualize_masks_png(points, gt_ids, pred_assigned, out_path="masks_comparison.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(42)

    def make_color_array(labels):
        unique = np.unique(labels)
        color_map = {u: rng.random(3) for u in unique}
        color_map[unique[labels.min() == unique].item() if labels.min() < 0 else -1] = np.array([0.15, 0.15, 0.15])
        return np.stack([color_map[l] for l in labels])

    gt_colors = make_color_array(gt_ids)
    pred_colors = make_color_array(pred_assigned)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    projections = [("Top (XY)", 0, 1), ("Front (XZ)", 0, 2), ("Side (YZ)", 1, 2)]

    for col, (name, xi, yi) in enumerate(projections):
        axes[0, col].scatter(points[:, xi], points[:, yi], c=gt_colors,   s=0.05, linewidths=0)
        axes[0, col].set_title(f"GT — {name}")
        axes[0, col].set_aspect("equal")

        axes[1, col].scatter(points[:, xi], points[:, yi], c=pred_colors, s=0.05, linewidths=0)
        axes[1, col].set_title(f"Pred — {name}")
        axes[1, col].set_aspect("equal")

    fig.suptitle(f"{SCENE}: GT instances (top) vs Predicted instances (bottom)\nGrey = unassigned", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {os.path.abspath(out_path)}")


def _load_masks_data(gt_path):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    points = np.asarray(mesh.vertices)  # (M, 3)

    gt_file = pathlib.Path(gt_path) / f"{SCENE}.txt"
    with open(gt_file) as f:
        gt_ids = np.array(f.read().splitlines(), dtype=np.int64)

    preds = load_predictions(EXPERIMENT_PATH, SCENE)
    pred_masks = preds["pred_masks"]   # (M, N_inst)
    scores = preds["pred_scores"]      # (N_inst,)
    weighted = pred_masks.astype(np.float32) * scores[None, :]
    assigned = weighted.argmax(axis=1).astype(np.int64)
    assigned[pred_masks[np.arange(len(points)), assigned] == 0] = -1

    return points, gt_ids, assigned


def save_masks_ply(gt_path):
    points, gt_ids, assigned = _load_masks_data(gt_path)
    save_instance_ply(points, gt_ids,  f"{SCENE}_gt_instances.ply")
    save_instance_ply(points, assigned, f"{SCENE}_pred_instances.ply", background_id=-1)
    print("Open both PLY files in MeshLab or CloudCompare to compare.")


def _make_o3d_pcd(points, labels, background_id=-1):
    import open3d as o3d
    rng = np.random.default_rng(42)
    unique = np.unique(labels)
    color_map = {u: rng.random(3) for u in unique}
    if background_id in color_map:
        color_map[background_id] = np.array([0.15, 0.15, 0.15])
    colors = np.stack([color_map[l] for l in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def vis_masks(gt_path, clip_z=None):
    import open3d as o3d
    points, gt_ids, assigned = _load_masks_data(gt_path)

    if clip_z is not None:
        mask = points[:, 2] < clip_z
        print(f"Clipping Z < {clip_z:.2f}: keeping {mask.sum()}/{len(mask)} points")
        points, gt_ids, assigned = points[mask], gt_ids[mask], assigned[mask]

    # Shift pred cloud to the right so both fit in the same window
    x_range = points[:, 0].max() - points[:, 0].min()
    offset = np.array([x_range + 0.5, 0.0, 0.0])

    gt_pcd   = _make_o3d_pcd(points, gt_ids)
    pred_pcd = _make_o3d_pcd(points, assigned, background_id=-1)
    pred_pcd.translate(offset)

    # Labels as text via coordinate frames (small axis at centroid of each cloud)
    gt_frame   = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=points.mean(axis=0))
    pred_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=points.mean(axis=0) + offset)

    print("Left = GT instances   |   Right = Predicted instances (grey = unassigned)")
    o3d.visualization.draw_geometries(
        [gt_pcd, pred_pcd, gt_frame, pred_frame],
        window_name=f"{SCENE} — GT (left) vs Pred (right)",
        width=1600, height=900,
    )


# ============================================================
# Step 3 helpers — AP evaluation
# ============================================================


def _compute_best_ious(pred_masks, gt_ids):
    unique_gt = np.unique(gt_ids)
    gt_masks = (unique_gt[None] == gt_ids[:, None])  # (n_pts, n_gt)
    n_pred = pred_masks.shape[1]
    best_ious = np.zeros(n_pred)
    for i in range(n_pred):
        p = pred_masks[:, i].astype(bool)
        if p.sum() == 0:
            continue
        inter = (p[:, None] & gt_masks).sum(axis=0)
        union = (p[:, None] | gt_masks).sum(axis=0)
        best_ious[i] = (inter / np.maximum(union, 1)).max()
    return best_ious


def inspect_instances(gt_path, clip_z=None, no_background=False):
    """GT (left, green highlight) and Pred (right, red highlight) side by side.
    Pred instances sorted best→worst IoU. N=next, P=prev."""
    import open3d as o3d

    preds = load_predictions(EXPERIMENT_PATH, SCENE, no_background=no_background)
    pred_masks_full = preds["pred_masks"]  # (M, N_pred)

    with open(pathlib.Path(gt_path) / f"{SCENE}.txt") as f:
        gt_ids_full = np.array(f.read().splitlines(), dtype=np.int64)

    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    points = np.asarray(mesh.vertices)

    if clip_z is not None:
        keep = points[:, 2] < clip_z
        points         = points[keep]
        gt_ids_vis     = gt_ids_full[keep]
        pred_masks_vis = pred_masks_full[keep]
    else:
        gt_ids_vis     = gt_ids_full
        pred_masks_vis = pred_masks_full

    # Sort pred instances best→worst IoU
    best_ious  = _compute_best_ious(pred_masks_full, gt_ids_full)
    order      = np.argsort(best_ious)[::-1]
    pred_order = order.tolist()
    iou_list   = best_ious[order]

    # Best GT match per pred instance
    unique_gt    = np.unique(gt_ids_full)
    gt_masks_cmp = (unique_gt[None] == gt_ids_full[:, None])

    def best_gt_id_for(pred_idx):
        p = pred_masks_full[:, pred_idx].astype(bool)
        if p.sum() == 0:
            return unique_gt[0]
        inter = (p[:, None] & gt_masks_cmp).sum(axis=0)
        union = (p[:, None] | gt_masks_cmp).sum(axis=0)
        return unique_gt[(inter / np.maximum(union, 1)).argmax()]

    # Base colors: muted random per instance
    rng = np.random.default_rng(42)

    unique_gt_vis  = np.unique(gt_ids_vis)
    gt_color_map   = {u: rng.random(3) * 0.35 + 0.1 for u in unique_gt_vis}
    gt_base        = np.stack([gt_color_map[l] for l in gt_ids_vis])

    n_inst         = pred_masks_vis.shape[1]
    inst_colors    = rng.random((n_inst, 3)) * 0.35 + 0.1
    pred_base      = np.full((len(points), 3), 0.08)
    for i in range(n_inst):
        pred_base[pred_masks_vis[:, i].astype(bool)] = inst_colors[i]

    # Lateral offset: pred cloud shifted to the right
    x_width = points[:, 0].max() - points[:, 0].min()
    offset  = np.array([x_width + 0.6, 0.0, 0.0])

    gt_pcd   = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(points)
    gt_pcd.colors = o3d.utility.Vector3dVector(gt_base)

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(points + offset)
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_base)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"{SCENE} — GT (left) | Pred (right)   N=next  P=prev  Q=quit",
                      width=1600, height=900)
    vis.add_geometry(gt_pcd)
    vis.add_geometry(pred_pcd)

    state = {"idx": 0}

    def update(step):
        state["idx"] = (state["idx"] + step) % len(pred_order)
        i     = state["idx"]
        pidx  = pred_order[i]
        iou   = iou_list[i]
        best_gt = best_gt_id_for(pidx)

        gt_colors   = gt_base.copy()
        gt_mask     = gt_ids_vis == best_gt
        gt_colors[gt_mask] = [0.0, 1.0, 0.0]   # bright green = matched GT
        gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)

        pred_colors = pred_base.copy()
        pred_mask   = pred_masks_vis[:, pidx].astype(bool)
        pred_colors[pred_mask] = [1.0, 0.15, 0.15]  # bright red = selected pred
        pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)

        vis.update_geometry(gt_pcd)
        vis.update_geometry(pred_pcd)
        print(f"  [{i+1}/{len(pred_order)}] pred inst {pidx}  IoU={iou:.3f}  "
              f"pred_pts={pred_mask.sum()}  gt_pts={gt_mask.sum()}")

    update(0)
    vis.register_key_callback(ord("N"), lambda v: update(+1))
    vis.register_key_callback(ord("P"), lambda v: update(-1))
    print(f"GT (izq, verde) | Pred (dcha, rojo) — {len(pred_order)} instancias, mejor IoU primero")
    print("N=siguiente  P=anterior  Q=salir")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Class-agnostic instance AP analysis — run in order:\n"
            "  Step 1  --align   Overlay pred PCL vs GT mesh (confirm alignment)\n"
            "  Step 2  --match   Navigate pred↔GT instance pairs (N=next P=prev)\n"
            "  Step 3  --eval    Compute class-agnostic AP metrics\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--dataset', default='replica', type=str)
    parser.add_argument('--gt_path', default="./replica_gt/", type=str)

    # --- step flags ---
    parser.add_argument('--all', action='store_true',
                        help="Run all steps in order (close each window to advance)")
    parser.add_argument('--align', action='store_true',
                        help="Step 1: overlay pred PCL (red) vs GT mesh (green) in Open3D")
    parser.add_argument('--match', action='store_true',
                        help="Step 2: side-by-side instance viewer, N=next P=prev (sorted by IoU)")
    parser.add_argument('--eval', action='store_true',
                        help="Step 3: compute class-agnostic AP metrics")

    # --- optional extras ---
    parser.add_argument('--save-ply', action='store_true',
                        help="Export colored instance PLY files (GT + pred) for external viewers")
    parser.add_argument('--save-png', action='store_true',
                        help="Save matplotlib projection of alignment (no GUI needed)")
    parser.add_argument('--clip-z', type=float, default=None,
                        help="Remove points above this Z value (e.g. 1.3 to cut ceiling)")
    parser.add_argument('--no-background', action='store_true',
                        help="Exclude predicted instances classified as wall/ceiling/floor/window/door")

    args = parser.parse_args()

    if args.all:
        print("=== Step 1/3: Alignment check (close window to continue) ===")
        visualize_alignment()
        print("=== Step 2/3: Instance matching viewer (Q to continue) ===")
        inspect_instances(args.gt_path, clip_z=args.clip_z, no_background=args.no_background)
        print("=== Step 3/3: AP metrics ===")
        test_class_agnostic_ap(args)
    elif args.align:
        # Step 1 — alignment check
        if args.save_png:
            import torch
            ckpt = torch.load(
                os.path.join(EXPERIMENT_PATH, SCENE, "ovo_map.ckpt"), map_location="cpu"
            )
            pcd_pred = ckpt["map_params"]["xyz"].numpy()
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(MESH_PATH)
            pcd_gt = np.asarray(mesh.vertices)
            _save_projection_png(pcd_pred, pcd_gt)
        else:
            visualize_alignment()
    elif args.match:
        # Step 2 — instance matching viewer
        inspect_instances(args.gt_path, clip_z=args.clip_z, no_background=args.no_background)
    elif args.eval:
        # Step 3 — AP metrics
        test_class_agnostic_ap(args)
    elif args.save_ply:
        save_masks_ply(args.gt_path)
    else:
        parser.print_help()
