"""
Script to extract SLAM baseline without semantic processing.

Extracts camera poses and 3D map from SLAM backend (ORB-SLAM, vanilla, gaussian_slam)
and saves in a reproducible format for experiments.

Usage:
    python scripts/extract_slam_baseline.py --dataset Replica --scene office0
    python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --no-loop-closure
    python scripts/extract_slam_baseline.py --dataset ScanNet --scene scene0011_00 --slam vanilla
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
from ovo.entities.ovomapping import get_slam_backend


# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def extract_slam_baseline(dataset_name, scene_name, slam_module="orbslam2", 
                          close_loops=True, output_dir="data/baselines"):
    """
    Extract SLAM baseline from a scene without semantic processing.
    
    Args:
        dataset_name: "Replica", "ScanNet", etc.
        scene_name: "office0", "scene0011_00", etc.
        slam_module: "orbslam2", "vanilla", "gaussian_slam"
        close_loops: Si True, permite loop closures (solo orbslam2)
        output_dir: Directorio donde guardar el .pth
    """
    
    print(f"{Colors.BOLD}Extracting SLAM baseline: {dataset_name}/{scene_name}{Colors.RESET}")
    print(f"   Backend: {slam_module}")
    print(f"   Loop closures: {close_loops if slam_module == 'orbslam2' else 'N/A'}")
    
    # 1. Load configurations
    config = load_configs(dataset_name, scene_name, slam_module, close_loops)
    
    # 2. Initialize dataset
    print(f"\nLoading dataset...")
    dataset = get_dataset(dataset_name)({**config["data"], **config["cam"]})
    cam_intrinsics = torch.tensor(dataset.intrinsics.astype(np.float32), 
                                   device=config.get("device", "cuda"))
    
    # 3. Initialize ONLY the SLAM backend
    print(f"Initializing SLAM backend: {slam_module}")
    slam = get_slam_backend(config, dataset, cam_intrinsics)
    
    # 4. Data structures to capture information
    baseline_data = {
        'camera_poses': {},      # ALL frames
        'map_evolution': {},     # ONLY keyframes
        'keyframes_info': {
            'ids': [],
            'timestamps': [],
            'frame_to_kf': {},
            'pcd_ranges': {}
        }
    }
    
    last_kf_id = None
    
    # 5. Process frames
    print(f"\nProcessing {len(dataset)} frames...")
    for frame_id in range(len(dataset)):
        frame_data = dataset[frame_id]
        
        # Tracking (todos los frames)
        slam.track_camera(frame_data)
        c2w = slam.get_c2w(frame_id)
        
        if c2w is None:
            continue  # Tracking failed
        
        # Save pose (ALL frames with successful tracking)
        baseline_data['camera_poses'][frame_id] = c2w.cpu().clone()
        
        # Mapping: depends on the backend
        if slam_module == "orbslam2":
            # ORB-SLAM handles mapping internally in track_camera
            # Call map() to trigger keyframe logic
            slam.map(frame_data, c2w)
            
            # Check if a keyframe was created
            if hasattr(slam, 'orbslam') and slam.orbslam.is_last_frame_kf():
                capture_keyframe_state(slam, frame_id, baseline_data)
                last_kf_id = frame_id
        else:
            # Vanilla/Gaussian: always perform mapping
            slam.map(frame_data, c2w)
            capture_keyframe_state(slam, frame_id, baseline_data)
            last_kf_id = frame_id
        
        # Progress
        if frame_id % 50 == 0 or frame_id == len(dataset) - 1:
            n_kfs = len(baseline_data['keyframes_info']['ids'])
            n_points = slam.pcd.shape[0] if hasattr(slam, 'pcd') else 0
            print(f"   Frame {frame_id:4d}/{len(dataset)} | "
                  f"KFs: {n_kfs:3d} | "
                  f"Points: {n_points:6d}")
    
    # 6. Complete frame_to_kf mapping
    print(f"\nCompleting frame->keyframe mapping...")
    complete_frame_to_kf(baseline_data)
    
    # 7. Create metadata
    lc_flag = "with_lc" if (close_loops and slam_module == "orbslam2") else "no_lc"
    metadata = {
        'scene_name': scene_name,
        'dataset': dataset_name,
        'slam_backend': slam_module,
        'loop_closure_enabled': close_loops if slam_module == "orbslam2" else False,
        'config': config,
        'extraction_date': datetime.now().isoformat(),
        'total_frames': len(baseline_data['camera_poses']),
        'num_keyframes': len(baseline_data['keyframes_info']['ids']),
        'final_map_points': slam.pcd.shape[0] if hasattr(slam, 'pcd') else 0,
    }
    
    # 8. Assemble complete baseline
    baseline = {
        'metadata': metadata,
        'camera_poses': baseline_data['camera_poses'],
        'map_evolution': baseline_data['map_evolution'],
        'keyframes_info': baseline_data['keyframes_info'],
        'intrinsics': cam_intrinsics.cpu().clone(),
    }
    
    # 9. Save file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{dataset_name}_{scene_name}_{slam_module}_{lc_flag}.pth"
    filepath = output_path / filename
    
    print(f"\nSaving baseline...")
    torch.save(baseline, filepath)
    
    # 10. Final statistics
    file_size_mb = filepath.stat().st_size / 1024 / 1024
    
    print(f"\n{Colors.GREEN}Baseline saved successfully!{Colors.RESET}")
    print(f"\nFile: {filepath}")
    print(f"\nStatistics:")
    print(f"   - Total frames processed:  {metadata['total_frames']}")
    print(f"   - Keyframes detected:      {metadata['num_keyframes']}")
    print(f"   - Final 3D points:         {metadata['final_map_points']}")
    print(f"   - File size:               {file_size_mb:.2f} MB")
    
    # Basic validation
    validate_baseline(baseline)


def capture_keyframe_state(slam, kf_id, baseline_data):
    """
    Captura el estado completo del mapa en un keyframe.
    
    Args:
        slam: Backend SLAM (WrapperORBSLAM2, VanillaMapper, etc.)
        kf_id: ID del keyframe
        baseline_data: Diccionario donde acumular los datos
    """
    # Guardar estado del mapa (acumulativo)
    baseline_data['map_evolution'][kf_id] = {
        'pcd': slam.pcd.cpu().clone(),
        'pcd_colors': slam.pcd_colors.cpu().clone(),
        'pcd_ids': slam.pcd_ids.cpu().clone(),
    }
    
    # Update keyframes_info
    kf_info = baseline_data['keyframes_info']
    kf_info['ids'].append(kf_id)
    
    # Timestamp: frame_id / 30fps (asumiendo 30fps por defecto)
    kf_info['timestamps'].append(float(kf_id) / 30.0)
    
    # pcd_ranges: determine which points were added by this KF
    if hasattr(slam, 'kfs') and kf_id in slam.kfs:
        kf_info['pcd_ranges'][kf_id] = slam.kfs[kf_id]['pcd_idxs']
    else:
        # If no range info, use current total
        n_points = slam.pcd.shape[0]
        prev_n_points = 0
        if len(kf_info['ids']) > 1:
            prev_kf = kf_info['ids'][-2]
            if prev_kf in baseline_data['map_evolution']:
                prev_n_points = baseline_data['map_evolution'][prev_kf]['pcd'].shape[0]
        kf_info['pcd_ranges'][kf_id] = (prev_n_points, n_points)


def complete_frame_to_kf(baseline_data):
    """
    Completa el mapeo frame_id → keyframe_id para todos los frames.
    
    Cada frame usa el mapa del keyframe más reciente (hacia atrás en el tiempo).
    """
    kf_ids = sorted(baseline_data['keyframes_info']['ids'])
    all_frame_ids = sorted(baseline_data['camera_poses'].keys())
    
    if len(kf_ids) == 0:
        print(f"   {Colors.YELLOW}Warning: No keyframes detected!{Colors.RESET}")
        return
    
    frame_to_kf = {}
    current_kf_idx = 0
    
    for frame_id in all_frame_ids:
        # Find the nearest keyframe looking backward
        while current_kf_idx < len(kf_ids) - 1 and kf_ids[current_kf_idx + 1] <= frame_id:
            current_kf_idx += 1
        
        frame_to_kf[frame_id] = kf_ids[current_kf_idx]
    
    baseline_data['keyframes_info']['frame_to_kf'] = frame_to_kf


def load_configs(dataset_name, scene_name, slam_module, close_loops):
    """
    Load and merge all necessary configurations.
    
    Returns:
        dict: Complete merged configuration
    """
    # Config base
    config = io_utils.load_config("data/working/configs/ovo.yaml")
    
    # Override SLAM settings
    config["slam"]["slam_module"] = slam_module
    if slam_module == "orbslam2":
        config["slam"]["close_loops"] = close_loops
    
    # Cargar config del backend SLAM
    # (orbslam2 usa vanilla configs)
    backend_name = "vanilla" if slam_module == "orbslam2" else slam_module
    slam_config_path = f"data/working/configs/slam/{backend_name}/{dataset_name.lower()}.yaml"
    
    if Path(slam_config_path).exists():
        config_slam = io_utils.load_config(slam_config_path)
        io_utils.update_recursive(config, config_slam)
    
    # Cargar config del dataset
    dataset_config_path = f"data/working/configs/{dataset_name}/{dataset_name.lower()}.yaml"
    if Path(dataset_config_path).exists():
        config_dataset = io_utils.load_config(dataset_config_path)
        io_utils.update_recursive(config, config_dataset)
    
    # Scene-specific config (optional)
    scene_config_path = f"data/working/configs/{dataset_name}/{scene_name}.yaml"
    if Path(scene_config_path).exists():
        config_scene = io_utils.load_config(scene_config_path)
        io_utils.update_recursive(config, config_scene)
    
    # Configure scene paths
    config["dataset_name"] = dataset_name
    if "data" not in config:
        config["data"] = {}
    config["data"]["scene_name"] = scene_name
    config["data"]["input_path"] = f"data/input/Datasets/{dataset_name}/{scene_name}"
    
    return config


def validate_baseline(baseline):
    """
    Basic validation of baseline structure.
    """
    print(f"\nValidating baseline...")
    
    errors = []
    warnings = []
    
    # 1. Required keys
    required_keys = ['metadata', 'camera_poses', 'map_evolution', 'keyframes_info', 'intrinsics']
    for key in required_keys:
        if key not in baseline:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        print(f"   {Colors.RED}Errors found:{Colors.RESET}")
        for err in errors:
            print(f"      - {err}")
        return
    
    # 2. Number consistency
    meta = baseline['metadata']
    kf_info = baseline['keyframes_info']
    
    if meta['num_keyframes'] != len(kf_info['ids']):
        warnings.append(f"Mismatch in num_keyframes: metadata={meta['num_keyframes']}, actual={len(kf_info['ids'])}")
    
    if meta['total_frames'] != len(baseline['camera_poses']):
        warnings.append(f"Mismatch in total_frames: metadata={meta['total_frames']}, actual={len(baseline['camera_poses'])}")
    
    # 3. All keyframes have pose
    for kf_id in kf_info['ids']:
        if kf_id not in baseline['camera_poses']:
            errors.append(f"Keyframe {kf_id} has no pose in camera_poses")
    
    # 4. All keyframes have map state
    for kf_id in kf_info['ids']:
        if kf_id not in baseline['map_evolution']:
            errors.append(f"Keyframe {kf_id} has no state in map_evolution")
    
    # 5. frame_to_kf is complete
    for frame_id in baseline['camera_poses'].keys():
        if frame_id not in kf_info['frame_to_kf']:
            errors.append(f"Frame {frame_id} is not in frame_to_kf")
    
    # 6. Map grows monotonically
    kf_ids = sorted(kf_info['ids'])
    prev_n = 0
    for kf_id in kf_ids:
        n_points = baseline['map_evolution'][kf_id]['pcd'].shape[0]
        if n_points < prev_n:
            warnings.append(f"Map decreased at keyframe {kf_id}: {prev_n} -> {n_points}")
        prev_n = n_points
    
    # Results
    if errors:
        print(f"   {Colors.RED}{len(errors)} error(s):{Colors.RESET}")
        for err in errors:
            print(f"      - {err}")
    else:
        print(f"   {Colors.GREEN}Structure is correct{Colors.RESET}")
    
    if warnings:
        print(f"   {Colors.YELLOW}{len(warnings)} warning(s):{Colors.RESET}")
        for warn in warnings:
            print(f"      - {warn}")


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
    
    args = parser.parse_args()
    
    try:
        extract_slam_baseline(
            dataset_name=args.dataset,
            scene_name=args.scene,
            slam_module=args.slam,
            close_loops=not args.no_loop_closure,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"\n{Colors.RED}Error during extraction:{Colors.RESET}")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
