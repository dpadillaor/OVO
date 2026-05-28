#!/usr/bin/env python3
"""
Visualize SAM2 exploration outputs.
Compare different runs or display results with transparency/overlays.
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import argparse

def load_frame(frame_name: str, dataset_path: str) -> np.ndarray:
    """Load original frame."""
    frame_path = Path(dataset_path) / frame_name
    if not frame_path.exists():
        # Try without extension
        for ext in ['.jpg', '.png']:
            alt_path = Path(dataset_path) / f"{Path(frame_name).stem}{ext}"
            if alt_path.exists():
                frame_path = alt_path
                break
    img = cv2.imread(str(frame_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def display_seg_map(seg_map: np.ndarray, title: str = "Segmentation Map") -> np.ndarray:
    """Convert seg_map to colored image."""
    if seg_map.size == 0:
        return np.zeros((512, 512, 3), dtype=np.uint8)

    num_instances = seg_map.max() + 1
    colors = np.random.RandomState(42).randint(0, 255, (num_instances, 3), dtype=np.uint8)

    colored = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for inst_id in range(num_instances):
        mask = seg_map == inst_id
        colored[mask] = colors[inst_id]

    return colored

def visualize_run(output_dir: Path, frame_name: str, dataset_path: str = None):
    """Display results for a single frame."""
    base_name = Path(frame_name).stem

    # Load outputs
    seg_path = output_dir / f"{base_name}_seg_map.npy"
    overlay_path = output_dir / f"{base_name}_overlay.png"
    binary_path = output_dir / f"{base_name}_binary_maps.npy"

    print(f"\n=== {base_name} ===")

    if seg_path.exists():
        seg_map = np.load(seg_path)
        print(f"Masks: {seg_map.max() + 1}")

    if overlay_path.exists():
        overlay = cv2.imread(str(overlay_path))
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        if dataset_path:
            original = load_frame(frame_name, dataset_path)
            if original is not None:
                # Side-by-side comparison
                h, w = original.shape[:2]
                comparison = np.hstack([original, overlay])
                cv2.imshow(f"Original (left) vs Overlay (right)", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow(f"{base_name} - Overlay", overlay)
        else:
            cv2.imshow(f"{base_name} - Overlay", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if binary_path.exists():
        binary_maps = np.load(binary_path)
        print(f"Binary maps shape: {binary_maps.shape}")

def compare_runs(run_dirs: list, frame_name: str):
    """Compare outputs from multiple runs."""
    base_name = Path(frame_name).stem

    overlays = []
    titles = []

    for run_dir in run_dirs:
        overlay_path = Path(run_dir) / f"{base_name}_overlay.png"
        if overlay_path.exists():
            overlay = cv2.imread(str(overlay_path))
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlays.append(overlay)
            titles.append(Path(run_dir).name)

    if not overlays:
        print("No overlay files found!")
        return

    # Create comparison grid
    h, w = overlays[0].shape[:2]
    cols = min(2, len(overlays))
    rows = (len(overlays) + cols - 1) // cols

    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx, (overlay, title) in enumerate(zip(overlays, titles)):
        r, c = divmod(idx, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = overlay
        # Add text label (basic)
        print(f"  [{r},{c}] {title}")

    cv2.imshow("Comparison", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAM2 exploration outputs")
    parser.add_argument("--output-dir", default="scripts/sam2_exploration/output")
    parser.add_argument("--frame", help="Frame name to visualize (e.g., '000000.jpg')")
    parser.add_argument("--dataset", help="Dataset path for comparison (optional)")
    parser.add_argument("--compare", nargs="+", help="Compare multiple run directories")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.compare:
        if not args.frame:
            print("--frame required for comparison")
            sys.exit(1)
        compare_runs(args.compare, args.frame)
    else:
        if not args.frame:
            # List all frames in output dir
            frames = set()
            for f in output_dir.glob("*_overlay.png"):
                base = f.name.replace("_overlay.png", "")
                frames.add(base)

            print(f"Found {len(frames)} frame(s) in {output_dir}")
            for f in sorted(frames):
                visualize_run(output_dir, f, args.dataset)
        else:
            visualize_run(output_dir, args.frame, args.dataset)
