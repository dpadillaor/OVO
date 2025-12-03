"""
This script visualizes and compares an estimated camera trajectory from an experiment run
against its corresponding ground truth trajectory.
It loads both trajectories, extracts their positions, and plots them in a 2D top-down view
using matplotlib for visual analysis of tracking performance.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

from ovo.utils.io_utils import load_config, load_scene_data
from run_eval import load_representation
import yaml
from ovo.utils.vis_utils import get_cmap, get_pcd_colors

def create_trajectory_lineset(positions: np.ndarray, color: list[float]) -> o3d.geometry.LineSet:
    """
    Creates an Open3D LineSet object from a sequence of 3D points.

    Args:
        positions (np.ndarray): An array of shape (N, 3) representing the 3D points.
        color (list[float]): A list of three RGB values (0-1) for the line color.

    Returns:
        o3d.geometry.LineSet: The trajectory visualized as a 3D line.
    """
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(positions)
    
    # Create lines connecting consecutive points
    lines = []
    for i in range(len(positions) - 1):
        lines.append([i, i + 1])
    lineset.lines = o3d.utility.Vector2iVector(lines)
    
    lineset.paint_uniform_color(color)
    return lineset

def load_gt_trajectory(dataset_name: str, scene_name: str) -> dict[int, torch.Tensor]:
    """
    Loads the ground truth camera trajectory from a text file.

    Args:
        dataset_name (str): The name of the dataset (e.g., "Replica").
        scene_name (str): The name of the scene (e.g., "office0").

    Returns:
        dict[int, torch.Tensor]: A dictionary where keys are frame IDs and values are
                                 4x4 ground truth pose tensors.

    Raises:
        FileNotFoundError: If the ground truth trajectory file does not exist.
    """
    traj_file = Path(f"data/input/Datasets/{dataset_name.capitalize()}/{scene_name}/traj.txt")
    
    if not traj_file.exists():
        raise FileNotFoundError(f"Ground truth trajectory file not found: {traj_file}")

    trajectory = {}
    with open(traj_file, 'r') as f:
        for i, line in enumerate(f):
            values = [float(v) for v in line.strip().split()]
            if len(values) == 16:
                pose = torch.tensor(values).reshape(4, 4)
                trajectory[i] = pose
    print(f"Loaded {len(trajectory)} GT poses from {traj_file}.")
    return trajectory

def load_estimated_trajectory(run_path: Path) -> dict[int, torch.Tensor]:
    """
    Loads the estimated camera trajectory from an .npy file saved by the SLAM backend.

    Args:
        run_path (Path): The path to the experiment run directory, which contains
                         the 'estimated_c2w.npy' file.

    Returns:
        dict[int, torch.Tensor]: A dictionary where keys are frame IDs and values are
                                 4x4 estimated pose tensors.

    Raises:
        FileNotFoundError: If the estimated trajectory file does not exist.
    """
    estimated_traj_file = run_path / "estimated_c2w.npy"
    
    if not estimated_traj_file.exists():
        raise FileNotFoundError(f"Estimated trajectory file not found: {estimated_traj_file}")

    estimated_c2w = torch.load(estimated_traj_file)
    print(f"Loaded {len(estimated_c2w)} estimated poses from {estimated_traj_file}.")
    return estimated_c2w

def main(args):
    """
    Main function to orchestrate the loading and visualization of camera trajectories.

    It constructs the run_path from arguments, loads both ground truth and estimated
    trajectories, extracts their 2D positions, and plots them using matplotlib.

    Args:
        args: Command-line arguments parsed by argparse, containing 'experiment_name',
              'dataset_name', and 'scene_name'.
    """
    # Construct the run path from arguments
    # Note: dataset_name is capitalized to match the directory structure (e.g., "Replica")
    run_path = Path(f"data/output/{args.dataset_name.capitalize()}/{args.experiment_name}/{args.scene_name}")

    print(f"Visualizing trajectories for experiment run: {run_path}")

    config = load_config(run_path/"config.yaml")

    dataset_name_capitalized = args.dataset_name.capitalize()
    if dataset_name_capitalized == "Scannet":
        dataset_name_capitalized = "ScanNet"
    
    data_path = Path("data/input/Datasets/") # Assuming this base path for datasets

    # This path might need adjustment based on actual file location
    dataset_info_file_path = Path("data/working/configs/") / dataset_name_capitalized / "eval_info.yaml"
    with open(dataset_info_file_path, 'r') as f:
        dataset_info = yaml.full_load(f)

    # Load trajectories
    gt_trajectory_dict = load_gt_trajectory(args.dataset_name, args.scene_name)
    estimated_trajectory_dict = load_estimated_trajectory(run_path)

    # Extract 3D positions
    gt_positions = np.array([pose[:3, 3].cpu().numpy() for pose in gt_trajectory_dict.values()])
    estimated_positions = np.array([pose[:3, 3] for pose in estimated_trajectory_dict.values()])

    # Create 3D line sets for visualization
    gt_lineset = create_trajectory_lineset(gt_positions, color=[0.0, 1.0, 0.0]) # Green for GT
    estimated_lineset = create_trajectory_lineset(estimated_positions, color=[1.0, 0.0, 0.0]) # Red for Estimated

    # Add coordinate frame for context
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    elements_to_visualize = [gt_lineset, estimated_lineset, coord_frame]

    # --- ADD THIS BLOCK FOR POINT CLOUD LOADING AND VISUALIZATION ---
    if args.show_pcds:
        print("Loading point clouds for visualization...")
        
        # Load Ground Truth Point Cloud
        # pcd_labels_gt is also returned but not used for geometric visualization
        _, pcd_gt = load_scene_data(config["dataset_name"], config["data"]["scene_name"], data_path, dataset_info)

        # Load Predicted Point Cloud (from the experiment run)
        # semantic_module is also returned but not used for geometric visualization
        _, params = load_representation(run_path, eval=True)
        pcd_pred = params["xyz"] # This is the predicted point cloud

        # Create Open3D PointCloud objects and assign colors
        cmap = get_cmap()
        
        # Ground Truth Point Cloud (assigned a unique color)
        gt_colors_idx = np.zeros(pcd_gt.shape[0], dtype=np.int32) # Dummy ID 0
        # gt_pcd_colors = get_pcd_colors(gt_colors_idx, cmap) * 0.7 # Darken slightly for distinction
        gt_pcd_colors = np.tile(np.array([[0.0, 0.0, 1.0]]), (pcd_gt.shape[0], 1)) # Blue for GT
        gt_pcd_o3d = o3d.geometry.PointCloud()
        gt_pcd_o3d.points = o3d.utility.Vector3dVector(pcd_gt)
        gt_pcd_o3d.colors = o3d.utility.Vector3dVector(gt_pcd_colors)

        # Predicted Point Cloud (assigned another unique color)
        pred_colors_idx = np.ones(pcd_pred.shape[0], dtype=np.int32) # Dummy ID 1
        pred_pcd_colors = get_pcd_colors(pred_colors_idx, cmap) * 0.7 # Darken slightly
        pred_pcd_o3d = o3d.geometry.PointCloud()
        pred_pcd_o3d.points = o3d.utility.Vector3dVector(pcd_pred)
        pred_pcd_o3d.colors = o3d.utility.Vector3dVector(pred_pcd_colors)
        
        # Apply voxel downsampling if specified
        if args.voxel_size is not None:
            print(f"Applying voxel downsampling with size: {args.voxel_size}")
            gt_pcd_o3d = gt_pcd_o3d.voxel_down_sample(voxel_size=args.voxel_size)
            pred_pcd_o3d = pred_pcd_o3d.voxel_down_sample(voxel_size=args.voxel_size)

        elements_to_visualize.extend([gt_pcd_o3d, pred_pcd_o3d]) # Add to the list

    print("Displaying 3D trajectories and/or point clouds. Close the window to exit.")

    # --- Console Legend ---
    print("\n--- Visualization Legend ---")
    print("Green Line: Ground Truth Trajectory")
    print("Red Line: Estimated Trajectory")
    if args.show_pcds:
        print("Light Green Points: Ground Truth Point Cloud")
        print("Light Red Points: Predicted Point Cloud")
    print("--------------------------\n")
    # --- End Console Legend ---

    o3d.visualization.draw_geometries(elements_to_visualize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize estimated vs. ground truth camera trajectories.')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment run.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset (e.g., Replica).')
    parser.add_argument('--scene_name', type=str, required=True,
                        help='Name of the scene (e.g., office0).')
    parser.add_argument('--show_pcds', action='store_true',
                        help='If set, visualize ground truth and predicted point clouds.')
    parser.add_argument('--voxel_size', type=float, default=None,
                        help='Voxel size for downsampling point clouds (e.g., 0.05). If None, no downsampling.')
    args = parser.parse_args()
    main(args)
