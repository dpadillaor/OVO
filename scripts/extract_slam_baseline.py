"""
Script to extract SLAM baseline without semantic processing.

Extracts camera poses and 3D map from SLAM backend (ORB-SLAM, vanilla, gaussian_slam)
and saves in a reproducible format for experiments.

Usage:
    python scripts/extract_slam_baseline.py --dataset Replica --scene office0
    python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --no-loop-closure
    python scripts/extract_slam_baseline.py --dataset ScanNet --scene scene0011_00 --slam vanilla
    python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --save-point-clouds
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ovo.utils import io_utils, gen_utils
from ovo.entities.datasets import get_dataset
from ovo.entities.ovomapping import get_slam_backbone


# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    RED_B = BOLD + RED
    YELLOW = '\033[93m'
    YELLOW_B = BOLD + YELLOW
    GREEN = '\033[92m'
    GREEN_B = BOLD + GREEN
    BLUE = '\033[94m'
    BLUE_B = BOLD + BLUE


def extract_slam_baseline(dataset_name, scene_name, slam_module="orbslam2", 
                          close_loops=True, output_dir="data/baselines", validate=True,
                          save_point_clouds=False):
    """
    Extract SLAM baseline from a scene without semantic processing.
    
    Args:
        dataset_name: "Replica", "ScanNet", etc.
        scene_name: "office0", "scene0011_00", etc.
        slam_module: "orbslam2", "vanilla", "gaussian_slam"
        close_loops: Si True, permite loop closures (solo orbslam2)
        output_dir: Directorio donde guardar el .pth
        validate: Si True, valida la estructura del baseline antes de guardar
        save_point_clouds: Si True, guarda nubes de puntos en loop closures (aumenta tamaño del archivo)
    """
    print("\n")
    print(f"{Colors.BLUE}################################################################")
    print(f"{Colors.BLUE_B}Extracting SLAM baseline: {dataset_name}/{scene_name}{Colors.BLUE}")
    print(f"   Backend: {slam_module}")
    print(f"   Loop closures: {close_loops if slam_module == 'orbslam2' else 'N/A'}")
    print(f"################################################################{Colors.RESET}")

    
    # 1. Load and merge configurations
    config = load_configs(dataset_name, scene_name, slam_module, close_loops)

    # 2. Load dataset
    print(f"\nLoading dataset...")
    dataset = get_dataset(dataset_name.lower())({**config["data"], **config["cam"]})
    cam_intrinsics = torch.tensor(dataset.intrinsics.astype(np.float32), device=config.get("device", "cuda"))

    # 3. Initialize SLAM backend
    print(f"Initializing SLAM backend: {slam_module}")
    slam = get_slam_backbone(config, dataset, cam_intrinsics)
    
    # 4. Initialize data structures
    baseline_data = initialize_baseline_data()
    
    # 5. Process all frames
    print(f"\nProcessing {len(dataset)} frames...")
    process_frames(slam, dataset, baseline_data, slam_module, save_point_clouds)
    
    # 6. Create metadata
    lc_flag = "with_lc" if (close_loops and slam_module == "orbslam2") else "no_lc"
    
    # Calculate config hash
    import hashlib
    import json
    config_str = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    metadata = {
        'scene': scene_name,
        'dataset': dataset_name,
        'slam_backend': slam_module,
        'timestamp': datetime.now().isoformat(),
        'config_hash': config_hash,
        'intrinsics': cam_intrinsics.cpu().clone(),
        'loop_closure_enabled': close_loops if slam_module == "orbslam2" else False,
        'total_frames': len(baseline_data['estimated_c2ws']),
        'num_keyframes': len(baseline_data['kfs']),
        'num_loop_closures': len(baseline_data['loop_closures']),
        'final_map_points': slam.pcd.shape[0] if hasattr(slam, 'pcd') else 0,
    }
    
    # 7. Assemble complete baseline
    baseline = {
        'metadata': metadata,
        'estimated_c2ws': baseline_data['estimated_c2ws'],
        'kfs': baseline_data['kfs'],
        'loop_closures': baseline_data['loop_closures'],
    }
    
    # 8. Save file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{dataset_name}_{scene_name}_{slam_module}_{lc_flag}.pth"
    filepath = output_path / filename
    
    print(f"\nSaving baseline...")
    torch.save(baseline, filepath)
    
    # 9. Validate if requested
    if validate:
        is_valid = validate_baseline(baseline)
        if not is_valid:
            print(f"\n{Colors.RED}Validation failed! File may be corrupted.{Colors.RESET}")
            return
    
    # 10. Final statistics
    print_final_statistics(filepath, metadata)


def initialize_baseline_data():
    """Initialize data structures for baseline extraction."""
    return {
        'estimated_c2ws': {},    # ALL frames
        'kfs': [],               # List of keyframe IDs
        'loop_closures': []      # Loop closure events with before/after states
    }


def process_frames(slam, dataset, baseline_data, slam_module, save_point_clouds=False):
    """
    Process all frames: tracking, mapping, and keyframe detection.
    
    Args:
        slam: SLAM backend instance
        dataset: Dataset instance
        baseline_data: Dictionary to accumulate extraction data
        slam_module: Name of the SLAM module ("orbslam2", "vanilla", etc.)
        save_point_clouds: Whether to save point clouds in loop closures
    """
    # for frame_id in range(len(dataset)):
    for frame_id in range(1700):

        frame_data = dataset[frame_id]
        
        # Tracking (all frames)
        slam.track_camera(frame_data)
        c2w = slam.get_c2w(frame_id)
        
        if c2w is None:
            continue  # Tracking failed
        
        # Save pose (ALL frames with successful tracking)
        baseline_data['estimated_c2ws'][frame_id] = c2w.cpu().clone()
        
        # Save state before mapping (for potential loop closure capture)
        if slam_module == "orbslam2":
            save_state_snapshot(slam, baseline_data, save_point_clouds)
        
        # Mapping and keyframe detection
        process_mapping_and_keyframe_detection(
            slam, frame_data, c2w, frame_id, slam_module, baseline_data, save_point_clouds
        )
        
        # Progress reporting
        if frame_id % 50 == 0 or frame_id == len(dataset) - 1:
            report_progress(frame_id, len(dataset), baseline_data, slam)


def save_state_snapshot(slam, baseline_data, save_point_clouds=False):
    """
    Save a snapshot of the current SLAM state (for loop closure before state).
    
    Args:
        slam: SLAM backend (WrapperORBSLAM2)
        baseline_data: Dictionary to accumulate data
        save_point_clouds: Whether to include point clouds in the snapshot
    """
    if hasattr(slam, 'estimated_c2ws') and hasattr(slam, 'kfs'):
        snapshot = {
            'estimated_c2ws': {fid: pose.cpu().clone() for fid, pose in slam.estimated_c2ws.items()},
            'kfs': list(slam.kfs.keys())
        }
        
        # Optionally save point clouds
        if save_point_clouds and hasattr(slam, 'pcd'):
            snapshot['points_3d'] = {
                'pcd': slam.pcd.cpu().clone(),
                'pcd_ids': slam.pcd_ids.cpu().clone() if hasattr(slam, 'pcd_ids') else None,
                'pcd_colors': slam.pcd_colors.cpu().clone() if hasattr(slam, 'pcd_colors') else None
            }
        
        baseline_data['_temp_before_state'] = snapshot


def process_mapping_and_keyframe_detection(slam, frame_data, c2w, frame_id, slam_module, baseline_data, save_point_clouds=False):
    """
    Execute mapping and detect if current frame is a keyframe.
    Also captures loop closure events for ORB-SLAM.
    
    Returns:
        bool: True if frame is a keyframe
    """
    # Capture state before mapping (for loop closure detection)
    if slam_module == "orbslam2" and hasattr(slam, 'last_big_change_id'):
        pre_lc_id = slam.last_big_change_id
    
    if slam_module == "orbslam2":
        # ORB-SLAM handles mapping internally in track_camera
        slam.map(frame_data, c2w)
        
        # Check if a keyframe was created
        if hasattr(slam, 'orbslam') and slam.orbslam.is_last_frame_kf():
            baseline_data['kfs'].append(frame_id)
        
        # Check if loop closure occurred
        if hasattr(slam, 'last_big_change_id') and slam.last_big_change_id != pre_lc_id:
            capture_loop_closure(slam, frame_id, baseline_data, pre_lc_id, save_point_clouds)
            return True
    else:
        # Vanilla/Gaussian: always perform mapping
        slam.map(frame_data, c2w)
        baseline_data['kfs'].append(frame_id)
        return True
    
    return False


def capture_loop_closure(slam, frame_id, baseline_data, pre_lc_id, save_point_clouds=False):
    """
    Capture loop closure event with before/after states.
    
    Args:
        slam: SLAM backend (WrapperORBSLAM2)
        frame_id: Frame ID where loop closure was detected
        baseline_data: Dictionary to accumulate data
        pre_lc_id: Last big change ID before this loop closure
        save_point_clouds: Whether to include point clouds in the capture
    """
    # Get the before state (should have been saved)
    before_state = baseline_data.get('_temp_before_state', None)
    
    # Capture after state
    after_c2ws = {fid: pose.cpu().clone() for fid, pose in slam.estimated_c2ws.items()}
    after_kfs = list(slam.kfs.keys()) if hasattr(slam, 'kfs') else baseline_data['kfs'].copy()
    
    after_state = {
        'estimated_c2ws': after_c2ws,
        'kfs': after_kfs
    }
    
    # Optionally save point clouds in after state
    if save_point_clouds and hasattr(slam, 'pcd'):
        after_state['points_3d'] = {
            'pcd': slam.pcd.cpu().clone(),
            'pcd_ids': slam.pcd_ids.cpu().clone() if hasattr(slam, 'pcd_ids') else None,
            'pcd_colors': slam.pcd_colors.cpu().clone() if hasattr(slam, 'pcd_colors') else None
        }
    
    loop_closure_event = {
        'frame_id': frame_id,
        'before': before_state if before_state else {
            'estimated_c2ws': {},
            'kfs': []
        },
        'after': after_state
    }
    
    baseline_data['loop_closures'].append(loop_closure_event)
    
    # Clear temp state
    if '_temp_before_state' in baseline_data:
        del baseline_data['_temp_before_state']


def report_progress(frame_id, total_frames, baseline_data, slam):
    """Print progress update during frame processing."""
    n_kfs = len(baseline_data['kfs'])
    n_lcs = len(baseline_data['loop_closures'])
    n_points = slam.pcd.shape[0] if hasattr(slam, 'pcd') else 0
    print(f"   Frame {frame_id:4d}/{total_frames} | "
          f"KFs: {n_kfs:3d} | LCs: {n_lcs:2d} | "
          f"Points: {n_points:6d}")


def load_configs(dataset_name, scene_name, slam_module, close_loops):
    """
    Load and merge all necessary configurations.
    
    Args:
        dataset_name: Name of the dataset
        scene_name: Name of the scene
        slam_module: SLAM module name
        close_loops: Whether to enable loop closure
    
    Returns:
        dict: Complete merged configuration
    """
    # Base config
    config = io_utils.load_config("data/working/configs/ovo.yaml")
    
    # Override SLAM settings
    config["slam"]["slam_module"] = slam_module
    if slam_module == "orbslam2":
        config["slam"]["close_loops"] = close_loops
    
    # Load mapper config (orbslam2 uses vanilla mapper)
    map_module = "vanilla" if slam_module == "orbslam2" else slam_module
    slam_config_path = Path(config["slam"]["config_path"]) / map_module / f"{dataset_name.lower()}.yaml"
    
    if slam_config_path.exists():
        config_slam = io_utils.load_config(str(slam_config_path))
        io_utils.update_recursive(config, config_slam)
    else:
        print(f"   {Colors.YELLOW}Warning: Mapper config not found at {slam_config_path}{Colors.RESET}")
    
    # Load dataset config
    dataset_config_path = Path(f"data/working/configs/{dataset_name}/{dataset_name.lower()}.yaml")
    if dataset_config_path.exists():
        config_dataset = io_utils.load_config(str(dataset_config_path))
        io_utils.update_recursive(config, config_dataset)
    else:
        print(f"   {Colors.YELLOW}Warning: Dataset config not found at {dataset_config_path}{Colors.RESET}")
    
    # Load scene-specific config (optional)
    scene_config_path = Path(f"data/working/configs/{dataset_name}/{scene_name}.yaml")
    if scene_config_path.exists():
        config_scene = io_utils.load_config(str(scene_config_path))
        io_utils.update_recursive(config, config_scene)
    
    # Configure scene paths
    config["dataset_name"] = dataset_name
    if "data" not in config:
        config["data"] = {}
    config["data"]["scene_name"] = scene_name
    config["data"]["input_path"] = f"data/input/Datasets/{dataset_name}/{scene_name}"
    
    return config


def print_final_statistics(filepath, metadata):
    """Print final statistics after successful extraction."""
    file_size_mb = filepath.stat().st_size / 1024 / 1024
    
    print(f"\n{Colors.GREEN}Baseline saved successfully!{Colors.RESET}")
    print(f"\nFile: {filepath}")
    print(f"\nStatistics:")
    print(f"   - Total frames processed:  {metadata['total_frames']}")
    print(f"   - Keyframes detected:      {metadata['num_keyframes']}")
    print(f"   - Loop closures detected:  {metadata['num_loop_closures']}")
    print(f"   - Final 3D points:         {metadata['final_map_points']}")
    print(f"   - File size:               {file_size_mb:.2f} MB")


def validate_baseline(baseline):
    """
    Validate baseline structure and data integrity.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print(f"\nValidating baseline...")
    
    errors = []
    warnings = []
    
    # 1. Check required keys
    required_keys = ['metadata', 'estimated_c2ws', 'kfs', 'loop_closures']
    for key in required_keys:
        if key not in baseline:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        print(f"   {Colors.RED}Critical errors found:{Colors.RESET}")
        for err in errors:
            print(f"      - {err}")
        return False
    
    # 2. Check numerical consistency
    meta = baseline['metadata']
    
    if meta['num_keyframes'] != len(baseline['kfs']):
        warnings.append(f"Mismatch in num_keyframes: metadata={meta['num_keyframes']}, actual={len(baseline['kfs'])}")
    
    if meta['total_frames'] != len(baseline['estimated_c2ws']):
        warnings.append(f"Mismatch in total_frames: metadata={meta['total_frames']}, actual={len(baseline['estimated_c2ws'])}")
    
    if meta['num_loop_closures'] != len(baseline['loop_closures']):
        warnings.append(f"Mismatch in num_loop_closures: metadata={meta['num_loop_closures']}, actual={len(baseline['loop_closures'])}")
    
    # 3. Verify all keyframes have poses
    for kf_id in baseline['kfs']:
        if kf_id not in baseline['estimated_c2ws']:
            errors.append(f"Keyframe {kf_id} has no pose in estimated_c2ws")
    
    # 4. Verify loop closures structure
    for i, lc in enumerate(baseline['loop_closures']):
        if 'frame_id' not in lc:
            errors.append(f"Loop closure {i} missing 'frame_id'")
        if 'before' not in lc or 'after' not in lc:
            errors.append(f"Loop closure {i} missing 'before' or 'after' state")
        else:
            if 'estimated_c2ws' not in lc['before'] or 'kfs' not in lc['before']:
                warnings.append(f"Loop closure {i} 'before' state incomplete")
            if 'estimated_c2ws' not in lc['after'] or 'kfs' not in lc['after']:
                errors.append(f"Loop closure {i} 'after' state incomplete")
    
    # Print results
    if errors:
        print(f"   {Colors.RED}- {len(errors)} error(s):{Colors.RESET}")
        for err in errors:
            print(f"      - {err}")
        return False
    
    print(f"   {Colors.GREEN}- Structure is correct{Colors.RESET}")
    
    if warnings:
        print(f"   {Colors.YELLOW}- {len(warnings)} warning(s):{Colors.RESET}")
        for warn in warnings:
            print(f"      - {warn}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract SLAM baseline without semantic processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract baseline with ORB-SLAM3 + loop closure
  python scripts/extract_slam_baseline.py --dataset Replica --scene office0
  
  # Without loop closure
  python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --no-loop-closure
  
  # With vanilla SLAM backend
  python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --slam vanilla
  
  # ScanNet scene
  python scripts/extract_slam_baseline.py --dataset ScanNet --scene scene0011_00
  
  # Save point clouds in loop closures (larger file size)
  python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --save-point-clouds
        """
    )
    
    parser.add_argument('--dataset', required=True, 
                       choices=['Replica', 'ScanNet'],
                       help='Dataset name')
    parser.add_argument('--scene', required=True,
                       help='Scene name (e.g., office0, scene0011_00)')
    parser.add_argument('--slam', default='orbslam2',
                       choices=['orbslam2', 'vanilla', 'gaussian_slam'],
                       help='SLAM backend to use (default: orbslam2)')
    parser.add_argument('--no-loop-closure', action='store_true',
                       help='Disable loop closure (ORB-SLAM only)')
    parser.add_argument('--output-dir', default='data/baselines',
                       help='Output directory for baseline files (default: data/baselines)')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation of baseline structure')
    parser.add_argument('--save-point-clouds', action='store_true',
                       help='Save point clouds in loop closure events (increases file size significantly)')
    
    args = parser.parse_args()
    
    try:
        extract_slam_baseline(
            dataset_name=args.dataset,
            scene_name=args.scene,
            slam_module=args.slam,
            close_loops=not args.no_loop_closure,
            output_dir=args.output_dir,
            validate=not args.no_validate,
            save_point_clouds=args.save_point_clouds
        )
    except Exception as e:
        print(f"\n{Colors.RED}Error during extraction:{Colors.RESET}")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
