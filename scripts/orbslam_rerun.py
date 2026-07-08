"""Standalone: run ORB-SLAM3 backbone on a scene and stream its sparse map to Rerun.

No OVO pipeline, no dense reconstruction. Just ORB's own tracked map points
(sparse) + camera trajectory. Updates live as ORB tracks / loop-closes.

Usage:
    python scripts/orbslam_rerun.py --dataset Replica --scene office0
    python scripts/orbslam_rerun.py --dataset ScanNet --scene scene0011_00 --no-loops
    python scripts/orbslam_rerun.py --dataset Replica --scene office0 --save-rrd out.rrd --headless
"""
import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import yaml
import orbslam3 as orbslam

from ovo.entities.datasets import get_dataset


def load_config(dataset: str, scene: str):
    """Mirror run_eval config resolution for the dataset + cam block only."""
    cfg_root = Path("data/working/configs")
    ds_cfg = yaml.full_load(open(cfg_root / dataset / f"{dataset.lower()}.yaml"))
    data = {
        "input_path": f"data/input/Datasets/{dataset}/{scene}",
        "scene_name": scene,
    }
    return ds_cfg, {**data, **ds_cfg["cam"]}


def build_orbslam(dataset: str, scene: str, close_loops: bool, use_viewer: bool):
    """Instantiate ORB-SLAM3 System directly (RGBD), same paths as the OVO wrapper."""
    configs_path = Path("data/working/configs/slam") / "orbslam3"
    vocab = configs_path / "vocabulary" / "ORBvoc.txt"
    if not vocab.exists():
        raise FileNotFoundError(f"ORB vocabulary not found at {vocab}")

    per_scene = configs_path / dataset.lower() / f"{scene}.yaml"
    settings = per_scene if per_scene.exists() else configs_path / f"{dataset.lower()}.yaml"
    print(f"Vocab:    {vocab}")
    print(f"Settings: {settings}")

    system = orbslam.System(
        str(vocab), str(settings), orbslam.Sensor.RGBD, use_viewer, not close_loops
    )
    system.initialize()
    return system


def to_pose(traj_point):
    """[id, 12 row-major 3x4] -> 4x4 c2w."""
    m = np.array(traj_point[-12:], dtype=np.float64).reshape(3, 4)
    out = np.eye(4)
    out[:3, :4] = m
    return out


def sample_colors(ids, pts, rgb, c2w, K, colors):
    """Project map points into the current RGB frame and cache a color per point id.

    ORB discards keyframe images, so map points have no color. We recover one by
    projecting each point into a frame that sees it and sampling the pixel. Cached
    by id, so each point keeps the first color it gets (cheap, good enough to read
    the scene).
    """
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T + w2c[:3, 3:4]).T  # (N,3) camera frame
    z = cam[:, 2]
    front = z > 1e-3
    u = (K[0, 0] * cam[:, 0] / z + K[0, 2]).round().astype(int)
    v = (K[1, 1] * cam[:, 1] / z + K[1, 2]).round().astype(int)
    h, wd = rgb.shape[:2]
    ok = front & (u >= 0) & (u < wd) & (v >= 0) & (v < h)
    for i in np.nonzero(ok)[0]:
        pid = int(ids[i])
        if pid not in colors:
            colors[pid] = rgb[v[i], u[i]]


def log_state(system, colors):
    """Push the full active map (BA/GBA-corrected) + trajectory to Rerun.

    `get_all_mappoints` returns the whole map re-read each call, so points visibly
    shift when local BA / GBA re-optimize them. Each row is (mnId, x, y, z); the id
    is used both as a Rerun keypoint id (stable identity across frames) and to look
    up the cached per-point color.
    """
    mps = system.get_all_mappoints()
    if mps:
        arr = np.array(mps, dtype=np.float64).reshape(-1, 4)
        ids = arr[:, 0].astype(np.int64)
        pts = arr[:, 1:4].astype(np.float32)
        finite = np.isfinite(pts).all(axis=1)
        ids, pts = ids[finite], pts[finite]
        cols = np.array([colors.get(int(i), (128, 128, 128)) for i in ids], dtype=np.uint8)
        rr.log("map/points", rr.Points3D(pts, radii=0.01, colors=cols,
                                         keypoint_ids=ids.astype(np.uint64)))

    traj = system.get_trajectory_points()
    if traj:
        cams = np.array([to_pose(t)[:3, 3] for t in traj], dtype=np.float32)
        rr.log("map/trajectory", rr.LineStrips3D([cams]))
        last = to_pose(traj[-1])
        rr.log(
            "map/camera",
            rr.Transform3D(translation=last[:3, 3], mat3x3=last[:3, :3]),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Replica")
    ap.add_argument("--scene", default="office0")
    ap.add_argument("--no-loops", action="store_true", help="disable loop closure")
    ap.add_argument("--every", type=int, default=5, help="log to rerun every N frames")
    ap.add_argument("--frame-limit", type=int, default=-1)
    ap.add_argument("--save-rrd", default=None, help="path to save .rrd recording")
    ap.add_argument("--headless", action="store_true", help="no live viewer (save only)")
    ap.add_argument("--orb-viewer", action="store_true", help="also open ORB Pangolin viewer")
    args = ap.parse_args()

    rr.init(f"ORBSLAM_{args.scene}", spawn=not args.headless)
    if args.save_rrd:
        rr.save(args.save_rrd)

    # ORB-SLAM3 world frame = first camera pose, in OpenCV convention (X right,
    # Y down, Z forward). Tell rerun so the grid/orbit-cam align to that handedness
    # instead of rerun's default up axis.
    rr.log("map", rr.ViewCoordinates.RDF, static=True)

    ds_cfg, data_cfg = load_config(args.dataset, args.scene)
    if args.frame_limit > 0:
        data_cfg["frame_limit"] = args.frame_limit
    dataset = get_dataset(ds_cfg["dataset_name"])(data_cfg)

    system = build_orbslam(
        args.dataset, args.scene, close_loops=not args.no_loops, use_viewer=args.orb_viewer
    )

    K = np.asarray(dataset.intrinsics, dtype=np.float64)
    colors = {}
    n = len(dataset)
    print(f"Running ORB-SLAM3 on {n} frames ...")
    for i in range(n):
        frame_id, rgb, depth = dataset[i][:3]
        system.process_image_rgbd(rgb, depth, frame_id)

        if i % args.every == 0:
            state = system.get_tracking_state()
            if state == orbslam.TrackingState.OK:
                traj = system.get_trajectory_points()
                mps = system.get_all_mappoints()
                if traj and mps:
                    arr = np.array(mps, dtype=np.float64).reshape(-1, 4)
                    sample_colors(arr[:, 0], arr[:, 1:4].astype(np.float32),
                                  rgb, to_pose(traj[-1]), K, colors)
            rr.set_time_sequence("frame", frame_id)
            log_state(system, colors)
            if state != orbslam.TrackingState.OK:
                print(f"[{frame_id}] tracking state: {state}")

    print(f"Done. Final full map logged ({len(colors)} points colored).")
    rr.set_time_sequence("frame", n)
    log_state(system, colors)
    system.shutdown()


if __name__ == "__main__":
    main()
