#!/usr/bin/env python3
"""
Streamlit app for comparing SAM2 exploration results.
"""
import streamlit as st
from pathlib import Path
import json
import cv2
from PIL import Image
import numpy as np

st.set_page_config(layout="wide", page_title="SAM2 Comparison Viewer")

def get_dataset_scene_from_path(dataset_path: str) -> tuple:
    """Extract dataset and scene from path."""
    parts = Path(dataset_path).parts
    if "Datasets" in parts:
        idx = parts.index("Datasets")
        return parts[idx + 1], parts[idx + 2]
    return None, None

@st.cache_data
def load_configs(sam_results_dir: Path) -> dict:
    """Load all configs from sam_results directory."""
    configs = {}

    if not sam_results_dir.exists():
        return configs

    for config_dir in sorted(sam_results_dir.iterdir()):
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name
        metadata_path = config_dir / "metadata.json"

        # Load metadata if exists
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        # Find all frames
        frames = []
        for frame_dir in sorted(config_dir.iterdir()):
            if frame_dir.is_dir() and not frame_dir.name.startswith("."):
                frames.append(frame_dir.name)

        configs[config_name] = {
            "path": config_dir,
            "metadata": metadata,
            "frames": frames,
        }

    return configs

def get_frame_images(config_path: Path, frame_id: str) -> dict:
    """Load 4 images for a frame."""
    frame_dir = config_path / frame_id
    images = {}

    layer_names = ["capa_0", "capa_1", "capa_2", "final"]
    for layer in layer_names:
        img_path = frame_dir / f"{layer}.png"
        if img_path.exists():
            images[layer] = Image.open(img_path)
        else:
            images[layer] = None

    return images

def main():
    st.title("SAM2 Comparison Viewer")

    # Output directory
    output_dir = Path(__file__).parent / "output"

    if not output_dir.exists():
        st.error(f"No output directory found at: {output_dir}")
        return

    # Parse folder names: {Dataset}_{Scene}_{Config}
    datasets = set()
    scenes = set()
    configs_set = set()
    all_folders = []

    for folder in output_dir.iterdir():
        if not folder.is_dir():
            continue

        parts = folder.name.split("_")
        if len(parts) >= 3:
            # Extract Dataset, Scene, Config
            # Assume: Dataset_Scene_Config...
            dataset = parts[0]
            scene = parts[1]
            config = "_".join(parts[2:])

            datasets.add(dataset)
            scenes.add(scene)
            configs_set.add(config)
            all_folders.append((dataset, scene, config, folder))

    # Sidebar selectors
    st.sidebar.header("Filter")

    selected_dataset = st.sidebar.selectbox(
        "Dataset",
        options=sorted(datasets),
        index=0 if datasets else None
    )

    # Filter scenes by dataset
    available_scenes = sorted(set(s for d, s, c, f in all_folders if d == selected_dataset))
    selected_scene = st.sidebar.selectbox(
        "Scene",
        options=available_scenes,
        index=0 if available_scenes else None
    )

    # Load configs for selected dataset+scene
    matching_dirs = [f for d, s, c, f in all_folders if d == selected_dataset and s == selected_scene]
    all_configs = {}

    for config_dir in matching_dirs:
        full_name = config_dir.name
        # Extract only config part (remove Dataset_Scene_)
        parts = full_name.split("_", 2)  # Split on first 2 underscores
        config_name = parts[2] if len(parts) > 2 else full_name  # sam16_iou0.8_...

        metadata_path = config_dir / "metadata.json"

        # Load metadata if exists
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        # Find all frames in this config
        frames = []
        for frame_dir in sorted(config_dir.iterdir()):
            if frame_dir.is_dir() and not frame_dir.name.startswith("."):
                frames.append(frame_dir.name)

        all_configs[config_name] = {
            "path": config_dir,
            "metadata": metadata,
            "frames": frames,
        }

    if not all_configs:
        st.error(f"No results for {selected_dataset}/{selected_scene}")
        return

    st.sidebar.success(f"Found {len(all_configs)} config(s)")

    # Config selection
    st.sidebar.header("Configurations")
    selected_configs = st.sidebar.multiselect(
        "Select configs to compare",
        options=list(all_configs.keys()),
        default=list(all_configs.keys())[:min(3, len(all_configs))],
        help="Select up to 3 configs"
    )

    if not selected_configs:
        st.warning("Select at least one configuration")
        return

    # Find common and all frames
    all_frames = set()
    for config_name in selected_configs:
        all_frames.update(all_configs[config_name]["frames"])

    if not all_frames:
        st.error("No frames found in selected configs")
        return

    sorted_frames = sorted(all_frames)

    # Frame selection slider
    st.sidebar.header("Frame Selection")
    frame_idx = st.sidebar.slider(
        "Frame",
        min_value=0,
        max_value=len(sorted_frames) - 1,
        value=0
    )
    selected_frame = sorted_frames[frame_idx]

    # Display configs info
    st.sidebar.header("Config Details")
    for config_name in selected_configs:
        with st.sidebar.expander(config_name, expanded=False):
            metadata = all_configs[config_name]["metadata"]
            if metadata:
                st.json(metadata.get("sam_config", {}))
            st.write(f"Frames: {len(all_configs[config_name]['frames'])}")

    # Display results
    st.header(f"Frame: {selected_frame}")

    # Display configs as columns (2 per row, each column shows 4 layers vertically)
    cols_per_row = 2
    config_list = list(selected_configs)

    layer_names = ["capa_0", "capa_1", "capa_2", "final"]
    layer_titles = ["Layer 0 (Full)", "Layer 1 (Crops)", "Layer 2 (Detail)", "Final (Filtered)"]

    for row_start in range(0, len(config_list), cols_per_row):
        row_configs = config_list[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_configs), gap="xxsmall")

        for col, config_name in zip(cols, row_configs):
            with col:
                st.subheader(f"📊 {config_name}")

                config_data = all_configs[config_name]
                frame_id = selected_frame

                # Check if frame exists in this config
                if frame_id not in config_data["frames"]:
                    st.warning(f"❌ Frame not available")
                    continue

                # Load images
                images = get_frame_images(config_data["path"], frame_id)

                # Display 4 layers vertically (one per row)
                for layer_name, layer_title in zip(layer_names, layer_titles):
                    if images[layer_name] is not None:
                        st.image(images[layer_name], caption=layer_title, width=500)
                    else:
                        st.error(f"❌ {layer_title} not found")

                # Config Details removed for cleaner layout

    # Summary
    st.sidebar.header("Summary")
    st.sidebar.write(f"**Configs:** {len(selected_configs)}/{len(all_configs)}")
    st.sidebar.write(f"**Total frames:** {len(sorted_frames)}")

if __name__ == "__main__":
    main()
