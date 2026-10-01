"""Time the Python VanillaMapper alone (GT poses, no SLAM, no semantics) on one Replica scene.

Reference for the C++ DenseMapper (cpp/apps/ovo_dense_map): same scene, same map_every, and the point cloud
is written as a binary xyz PLY so both maps can be compared.

Run from the OVO repo root (it uses the repo's data/ layout and configs):
    python scripts/bench_vanilla_mapper.py office0 --out /tmp/office0_python.ply
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ovo.entities.datasets import Replica
from ovo.slam.vanilla_mapper import VanillaMapper


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(scene: str) -> Replica:
    cam = load_yaml("data/working/configs/Replica/replica.yaml")["cam"]
    return Replica({"input_path": f"data/input/Datasets/Replica/{scene}", **cam})


def build_mapper(dataset: Replica, device: str) -> VanillaMapper:
    mapping = load_yaml("data/working/configs/slam/vanilla/replica.yaml")["mapping"]
    intrinsics = torch.tensor(dataset.intrinsics, dtype=torch.float32, device=device)
    return VanillaMapper({"device": device, "mapping": mapping}, intrinsics)


def write_ply(points: np.ndarray, path: Path) -> None:
    points = np.ascontiguousarray(points, dtype="<f4")
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="Replica scene, e.g. office0")
    parser.add_argument("--map-every", type=int, default=10, help="map one frame in N (ovomapping.py map_every)")
    parser.add_argument("--out", type=Path, default=None, help="write the final point cloud as PLY")
    args = parser.parse_args()

    device = "cuda"
    dataset = build_dataset(args.scene)
    mapper = build_mapper(dataset, device)

    load_s, map_s, first_map_s, mapped = 0.0, 0.0, 0.0, 0
    wall_start = time.perf_counter()
    for i in range(0, len(dataset), args.map_every):
        t0 = time.perf_counter()
        frame = dataset[i]  # (index, rgb, depth, c2w): reads jpg + png from disk
        c2w = torch.from_numpy(frame[3]).to(device)
        t1 = time.perf_counter()
        mapper.map(frame, c2w)
        torch.cuda.synchronize()  # GPU work is asynchronous: wait for it before stopping the clock
        t2 = time.perf_counter()

        load_s += t1 - t0
        if mapped == 0:
            first_map_s = t2 - t1  # includes CUDA warm-up, reported apart
        else:
            map_s += t2 - t1
        mapped += 1
    wall_s = time.perf_counter() - wall_start

    num_points = mapper.pcd.shape[0]
    print(f"scene {args.scene}: {mapped} frames mapped (map_every {args.map_every}), {num_points} points")
    print(f"  wall total         {wall_s:8.2f} s")
    print(f"  load (rgb + depth) {load_s:8.2f} s  ({1000 * load_s / mapped:.1f} ms/frame)")
    print(f"  map, first frame   {first_map_s:8.2f} s  (CUDA warm-up)")
    print(f"  map, rest          {map_s:8.2f} s  ({1000 * map_s / max(mapped - 1, 1):.1f} ms/frame)")

    if args.out is not None:
        write_ply(mapper.pcd.cpu().numpy(), args.out)
        print(f"  cloud -> {args.out}")


if __name__ == "__main__":
    main()
