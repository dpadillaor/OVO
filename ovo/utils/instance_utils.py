"""
Instance Utilities - Pure Mathematical/Geometric Functions

This module provides pure functions for geometric computations and data structure
manipulation. The fusion decision logic is being migrated to ovo.entities.fusion.
"""

import open3d as o3d
import torch
import numpy as np


def compute_centroid_distance(centroid1: torch.Tensor, centroid2: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between two centroids.

    Args:
        centroid1: First centroid tensor
        centroid2: Second centroid tensor

    Returns:
        torch.Tensor: Euclidean distance
    """
    return ((centroid1 - centroid2) ** 2).sum().sqrt()


def compute_aabb_distance(points1: torch.Tensor, points2: torch.Tensor) -> float:
    """
    Minimum 3D distance between the axis-aligned bounding boxes of two point clouds.

    Returns 0.0 when AABBs intersect. Otherwise returns the Euclidean length of
    the per-axis gap vector. More robust than centroid distance for elongated
    objects (sofas, tables) where centroids can be far apart but shapes overlap.
    """
    min1, _ = points1.min(dim=0)
    max1, _ = points1.max(dim=0)
    min2, _ = points2.min(dim=0)
    max2, _ = points2.max(dim=0)
    gap = torch.clamp(torch.maximum(min1 - max2, min2 - max1), min=0)
    return gap.norm().item()


def compute_pcd_overlap(
    points1: torch.Tensor,
    points2: torch.Tensor,
    th_points: float
) -> float:
    """
    Compute the overlap ratio between two point clouds.

    Args:
        points1: First point cloud tensor
        points2: Second point cloud tensor
        th_points: Distance threshold for considering points as overlapping

    Returns:
        float: Ratio of points within threshold distance
    """
    if points1.shape[0] <= points2.shape[0]:
        small, big = points1, points2
    else:
        small, big = points2, points1

    pcd_small = o3d.geometry.PointCloud()
    pcd_small.points = o3d.utility.Vector3dVector(small.cpu().numpy())
    pcd_big = o3d.geometry.PointCloud()
    pcd_big.points = o3d.utility.Vector3dVector(big.cpu().numpy())

    dists = np.asarray(pcd_small.compute_point_cloud_distance(pcd_big))
    return (dists < th_points).astype(float).mean()

def compute_pcd_old_overlap(
      points1: torch.Tensor,
      points2: torch.Tensor,
      th_points: float
  ) -> float:
      pcd1 = o3d.geometry.PointCloud()
      pcd1.points = o3d.utility.Vector3dVector(points1.cpu().numpy())
      pcd2 = o3d.geometry.PointCloud()
      pcd2.points = o3d.utility.Vector3dVector(points2.cpu().numpy())

      dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
      return (dists < th_points).astype(float).mean()


def same_instance(instance1, instance2, points_centroid1, points_centroid2, th_centroid, th_cossim, th_points):
    """
    Legacy function - kept for backward compatibility during migration.
    This logic is being moved to ovo.entities.fusion.FusionStrategy.
    """
    ## Check centroids
    points1, centroid1 = points_centroid1
    points2, centroid2 = points_centroid2
    distance = ((centroid1 - centroid2) ** 2).sum().sqrt()
    if distance > th_centroid:
        return False
    # Check CLIP similarity
    cos_sim = torch.nn.functional.cosine_similarity(instance1.clip_feature[0], instance2.clip_feature[0], dim=0)
    if cos_sim < th_cossim:
        return False

    # Check if more than 50% of points are really close
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1.cpu().numpy())
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2.cpu().numpy())
    dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    p_dist = (dists < th_points).astype(float).mean()
    return p_dist > 0.5 or (cos_sim > 0.9 and p_dist > 0.2)


def fuse_instances(instance1, instance2, map_data):
    """
    Fuse two instances by merging their data structures.

    Args:
        instance1: Target instance (receiver of fusion)
        instance2: Instance to be fused into instance1
        map_data: Tuple of (points_3d, points_ids, points_ins_ids)

    Returns:
        tuple: (fused instance1, updated points_ins_ids)
    """
    points_3d, points_ids, points_ins_ids = map_data

    instance1.add_points_ids(instance2.points_ids)
    for kf in instance2.kfs_ids:
        instance1.add_keyframes(kf)
    for (area, kf_id) in instance2.top_kf:
        instance1.add_top_kf(kf_id, area)
    points_ins_ids[points_ins_ids == instance2.id] = instance1.id
    return instance1, points_ins_ids
