"""Sparse voxel index for geometric overlap between instances.

An instance's cloud is quantized to the set of occupied cells on a grid shared by every
instance (same size, same origin), so two clouds share a voxel iff they produce the same
integer cell key. Overlap is then integer-set intersection, not nearest-neighbour search:
density collapses (many points in a cell count once) and there is no KD-tree, no tolerance.
"""

import torch

# Bit layout: 21 bits per axis, biased so negative cell indices stay non-negative.
# Range +/-2^20 cells per axis (>=50 km at 5 cm), which no real scene approaches.
_BITS = 21
_BIAS = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1


class VoxelIndex:
    """The set of grid cells a point cloud occupies, as sorted unique int64 keys."""

    def __init__(self, cells: torch.Tensor):
        self.cells = cells

    @classmethod
    def build(cls, points: torch.Tensor, size: float, origin: float = 0.0) -> "VoxelIndex":
        """Quantize `points` to occupied cell keys on the grid (size, origin). Stays on device."""
        if points.numel() == 0:
            return cls(points.new_empty(0, dtype=torch.int64))
        q = torch.floor((points - origin) / size).to(torch.int64) + _BIAS
        q = q.clamp_(0, _MASK)
        key = (q[:, 0] << (2 * _BITS)) | (q[:, 1] << _BITS) | q[:, 2]
        return cls(torch.unique(key))

    def __len__(self) -> int:
        return int(self.cells.numel())

    def shared(self, other: "VoxelIndex") -> int:
        """Number of cells occupied by both indices (|A n B|)."""
        if len(self) == 0 or len(other) == 0:
            return 0
        return int(torch.isin(self.cells, other.cells).sum())
