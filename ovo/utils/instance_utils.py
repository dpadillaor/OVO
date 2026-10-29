from dataclasses import dataclass

import open3d as o3d
import torch
import numpy as np

from ..entities.voxel import VoxelSet


@dataclass(frozen=True)
class VoxelFusionThresholds:
    """Thresholds for deciding whether two instances are the same, by voxel overlap."""
    min_cos_sim: float = 0.81
    strong_overlap: float = 0.5
    weak_overlap: float = 0.2

    @classmethod
    def from_cfg(cls, cfg: dict) -> "VoxelFusionThresholds":
        """Build from the `semantic` config section (`th_cossim`, `voxel_th_geom`, `voxel_th_sem`)."""
        return cls(
            min_cos_sim=float(cfg.get("th_cossim", 0.81)),
            strong_overlap=float(cfg.get("voxel_th_geom", 0.5)),
            weak_overlap=float(cfg.get("voxel_th_sem", 0.2)),
        )


def same_instance_by_voxel(instance1, instance2, voxels1: VoxelSet, voxels2: VoxelSet, th: VoxelFusionThresholds) -> bool:
    """Same as `same_instance`, but geometry is voxel overlap and there is no centroid check."""
    cos_sim = torch.nn.functional.cosine_similarity(instance1.clip_feature[0], instance2.clip_feature[0], dim=0).item()
    if cos_sim < th.min_cos_sim:
        return False

    overlap = voxels1.containment(voxels2)
    return overlap > th.strong_overlap or (cos_sim > 0.9 and overlap > th.weak_overlap)


def same_instance(instance1, instance2, points_centroid1, points_centroid2, th_centroid, th_cossim, th_points):
    ## Check centroids
    points1, centroid1 = points_centroid1
    points2, centroid2 = points_centroid2
    distance = ((centroid1 - centroid2) ** 2).sum().sqrt()
    if distance > th_centroid:
        return False
    # Check CLIP similarity
    cos_sim =torch.nn.functional.cosine_similarity(instance1.clip_feature[0], instance2.clip_feature[0], dim=0)
    if cos_sim < th_cossim:
        return
    
    # Check if more than 50% of points are really close
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1.cpu().numpy())
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2.cpu().numpy())
    dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    p_dist = (dists < th_points).astype(float).mean()
    return  p_dist > 0.5 or (cos_sim > 0.9 and p_dist > 0.2)

def fuse_instances(instance1, instance2, map_data):
    points_3d, points_ids, points_ins_ids = map_data
    
    instance1.add_points_ids(instance2.points_ids)
    for kf in instance2.kfs_ids:
        instance1.add_keyframes(kf)
    for (area, kf_id) in instance2.top_kf:
        instance1.add_top_kf(kf_id, area)
    points_ins_ids[points_ins_ids == instance2.id] = instance1.id
    return instance1, points_ins_ids