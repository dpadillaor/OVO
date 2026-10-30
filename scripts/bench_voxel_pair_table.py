"""Benchmark: per-pair VoxelSet.shared vs one flat cell table for the voxel broadphase.

Method 1 is what the fusion does today: find the candidate pairs (instances sharing a
cell), then call `shared` once per pair. Method 2 sorts every cell of every instance once
and counts the shared cells of all pairs in the same pass, so the pair loop needs no
`isin` and no GPU sync per pair.

Both must give the same candidate pairs, the same shared counts and the same eager
fusions; the script asserts it, then reports the time of each method.

    conda run -n ovo2 python scripts/bench_voxel_pair_table.py [--device cuda]
"""

import argparse
import time

import torch

from ovo.entities.fusion.voxel import VoxelGrid, VoxelSet


def scene(seed, n_inst, grid, device):
    """Random boxes of points (varying size and density) in a 6 m room."""
    g = torch.Generator().manual_seed(seed)
    sets = {}
    for i in range(n_inst):
        centre = torch.rand(3, generator=g) * 6.0
        extent = 0.1 + torch.rand(3, generator=g) * 0.8
        n = int(200 + torch.randint(0, 3000, (1,), generator=g))
        pts = centre + (torch.rand(n, 3, generator=g) - 0.5) * extent
        sets[i] = VoxelSet(pts.to(device), grid)
    return sets


def method1(sets):
    """Broadphase pairs as in VoxelOverlapCriterion.candidate_pairs, then shared() per pair."""
    keys = torch.cat([s.cells for s in sets.values()])
    ids = torch.cat([torch.full_like(s.cells, i) for i, s in sets.items()])
    order = torch.argsort(keys)
    keys, ids = keys[order], ids[order]
    _, counts = torch.unique_consecutive(keys, return_counts=True)
    pairs, start = set(), 0
    for cnt in counts.tolist():
        if cnt > 1:
            group = ids[start:start + cnt].tolist()
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    x, y = group[a], group[b]
                    pairs.add((min(x, y), max(x, y)))
        start += cnt
    return {p: sets[p[0]].shared(sets[p[1]]) for p in pairs}


def method2(sets):
    """One sort of all cells; shared counts of every pair from the runs of equal cells."""
    keys = torch.cat([s.cells for s in sets.values()])
    ids = torch.cat([torch.full_like(s.cells, i) for i, s in sets.items()])
    order = torch.argsort(keys, stable=True)
    keys, ids = keys[order], ids[order]
    _, run_id = torch.unique_consecutive(keys, return_inverse=True)
    n_ids = int(ids.max()) + 1
    codes, d = [], 1
    while True:  # pair each element with the one d positions ahead, if in the same cell
        same = run_id[d:] == run_id[:-d]
        if not bool(same.any()):
            break
        a, b = ids[:-d][same], ids[d:][same]
        codes.append(torch.minimum(a, b) * n_ids + torch.maximum(a, b))
        d += 1
    if not codes:
        return {}
    code, cnt = torch.unique(torch.cat(codes), return_counts=True)
    return {(c // n_ids, c % n_ids): k for c, k in zip(code.tolist(), cnt.tolist())}


def eager(sets, shared, th=0.5):
    """Eager loop with the update_map skips; geometry-only rule containment > th."""
    order, fused, near = sorted(sets), {}, {}
    for a, b in shared:
        near.setdefault(a, set()).add(b)
        near.setdefault(b, set()).add(a)
    for i, a in enumerate(order):
        if a in fused:
            continue
        for b in order[i + 1:]:
            if b in fused or b not in near.get(a, ()):
                continue
            if shared[(min(a, b), max(a, b))] / min(len(sets[a]), len(sets[b])) > th:
                fused[b] = a
    return fused


def timed(fn, sets, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(sets)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 40, 150, 500])
    args = parser.parse_args()
    device = torch.device(args.device)
    grid = VoxelGrid(0.05)

    print(f"device={device}")
    for n_inst in args.sizes:
        t1 = t2 = 0.0
        n_pairs = 0
        for seed in range(args.seeds):
            sets = scene(seed, n_inst, grid, device)
            m1, dt1 = timed(method1, sets, device)
            m2, dt2 = timed(method2, sets, device)
            assert m1 == m2, f"pairs/shared differ (seed={seed}, n={n_inst})"
            assert eager(sets, m1) == eager(sets, m2), f"fusions differ (seed={seed}, n={n_inst})"
            t1, t2, n_pairs = t1 + dt1, t2 + dt2, n_pairs + len(m1)
        print(f"n_inst={n_inst:4d}  pairs/scene={n_pairs / args.seeds:7.1f}  "
              f"method1={t1 / args.seeds * 1e3:8.2f} ms  method2={t2 / args.seeds * 1e3:8.2f} ms  "
              f"(identical results)")


if __name__ == "__main__":
    main()
