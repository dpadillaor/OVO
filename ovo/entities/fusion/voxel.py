"""Sparse voxel index for geometric overlap between instances.

An instance's cloud is quantized to the set of occupied cells on a grid shared by every
instance (same size, same origin), so two clouds share a voxel iff they produce the same
integer cell key. Overlap is then integer-set intersection, not nearest-neighbour search:
density collapses (many points in a cell count once) and there is no KD-tree, no tolerance.
"""

from dataclasses import dataclass

import torch

# Bit layout: 21 bits per axis, biased so negative cell indices stay non-negative.
# Range +/-2^20 cells per axis (>=50 km at 5 cm), which no real scene approaches.
_BITS = 21
_BIAS = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1


def _pack(ijk: torch.Tensor) -> torch.Tensor:
    """(N, 3) int64 cell coords -> (N,) int64 keys."""
    biased = (ijk + _BIAS).clamp(0, _MASK)
    return (biased[:, 0] << (2 * _BITS)) | (biased[:, 1] << _BITS) | biased[:, 2]


@dataclass(frozen=True)
class VoxelGrid:
    """An implicit, unbounded voxel grid: 3D space split into cubic cells of edge `size` (m),
    with cell boundaries at `origin + k * size` along each axis (k any integer).

    Nothing is allocated: cells exist only as keys of the sets built on it.
    """
    size: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError(f"voxel size must be > 0, got {self.size}")
        if len(self.origin) != 3:
            raise ValueError(f"voxel origin must have 3 values (x, y, z), got {self.origin}")

    @classmethod
    def from_cfg(cls, cfg: dict) -> "VoxelGrid":
        """Build from the fusion config (`voxel_size`, `voxel_origin` as a 3-item list)."""
        size = float(cfg.get("voxel_size", 0.05))
        x, y, z = cfg.get("voxel_origin", (0.0, 0.0, 0.0))
        origin = (float(x), float(y), float(z))
        return cls(size, origin)


class VoxelSet:
    """The set of grid cells a point cloud occupies, as sorted unique int64 keys."""

    def __init__(self, points: torch.Tensor, grid: VoxelGrid):
        self.grid = grid
        if points.numel() == 0:
            self.cells = points.new_empty(0, dtype=torch.int64)
            return
        origin = torch.as_tensor(grid.origin, dtype=points.dtype, device=points.device)
        cell_ijk = torch.floor((points - origin) / grid.size).to(torch.int64)
        # NOTE: density is dropped here: a cell counts once however many points fall in it.
        # To keep it, `torch.unique(..., return_counts=True)` also returns points per cell.
        self.cells = torch.unique(_pack(cell_ijk))

    def __len__(self) -> int:
        return int(self.cells.numel())

    def shared(self, other: "VoxelSet") -> int:
        """Number of cells occupied by both sets (|A n B|)."""
        if self.grid != other.grid:
            raise ValueError(f"cannot compare sets on different grids: {self.grid} vs {other.grid}")
        if len(self) == 0 or len(other) == 0:
            return 0
        return int(torch.isin(self.cells, other.cells).sum())
