"""Point clouds as sets of occupied voxel cells, for fast geometric overlap.

- VoxelGrid: the cell layout (edge size and origin). Pure description, nothing allocated.
- VoxelSet: the cells one cloud occupies on a grid, stored as unique int64 keys.

Each cell (i, j, k) is packed into a single int64 key (21 bits per axis), so a set is a
flat 1D tensor and torch.unique / torch.isin do all the work, on GPU too. Two clouds on the
same grid occupy the same cell iff they produce the same key, so overlap is the intersection
of their key sets. Density collapses: many points in a cell count once.
"""

from dataclasses import dataclass

import torch

# Key encoding used by _pack. Each axis covers +/-_AXIS_OFFSET cells around the grid origin,
# i.e. +/-_AXIS_OFFSET * grid.size metres (about +/-52 km at 5 cm).
_AXIS_BITS = 21                         # bits per axis; 3 * 21 = 63, so the int64 sign bit stays 0
_AXIS_OFFSET = 1 << (_AXIS_BITS - 1)    # half the axis range: shifts cells -2^20..2^20-1 to 0..2^21-1
_AXIS_MAX = (1 << _AXIS_BITS) - 1       # largest offset value one axis can hold (21 ones in binary)


def _pack(ijk: torch.Tensor) -> torch.Tensor:
    """Pack (N, 3) int64 cell coords into (N,) int64 keys: i in the high bits, then j, then k.

    Out-of-range coords are clamped to the edge cell, not rejected.
    """
    offset_ijk = (ijk + _AXIS_OFFSET).clamp(0, _AXIS_MAX)
    return (offset_ijk[:, 0] << (2 * _AXIS_BITS)) | (offset_ijk[:, 1] << _AXIS_BITS) | offset_ijk[:, 2]


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
