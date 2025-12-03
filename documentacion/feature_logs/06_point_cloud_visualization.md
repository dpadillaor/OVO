# 06 - Point Cloud Visualization

## Objective

Extend the `visualize_trajectory.py` script to allow for the simultaneous visualization of both the ground truth point cloud and the generated (predicted) point cloud from an experiment run.

This provides a crucial tool for direct geometric comparison, enabling us to:
1.  Visually assess the impact of trajectory drift on the generated map.
2.  Confirm the alignment of different point clouds after a loop closure and instance fusion event.

## Status

**COMPLETED**

## Implementation Notes

The `visualize_trajectory.py` script has been updated with the following features:

1.  **Command-line argument `--show_pcds`**: Added to conditionally enable the display of point clouds.
2.  **Command-line argument `--voxel_size`**: Added to allow optional voxel downsampling of both ground truth and predicted point clouds, improving visibility for dense scenes.
3.  **Point Cloud Loading**: Integrated logic to load both the ground truth point cloud (`pcd_gt`) and the predicted point cloud (`pcd_pred`) from an experiment run.
4.  **Color Assignment**: Utilized `ovo.utils.vis_utils.get_cmap` and `get_pcd_colors` to assign distinct colors to the GT and predicted point clouds (Green for GT, Red for Predicted), making them easily distinguishable.
5.  **Console Legend**: Added a clear console legend printed before the visualization window appears, explaining what each color represents for both trajectories and point clouds.