"""Point clouds as sets of occupied voxel cells, for fast geometric overlap.

- VoxelGrid: the cell layout (edge size and origin). Pure description, nothing allocated.
- VoxelSet: the cells one cloud occupies on a grid, stored as unique int64 keys.

Each cell (i, j, k) is packed into a single int64 key (21 bits per axis), so a set is a
flat 1D tensor and torch.unique / torch.isin do all the work, on GPU too. Two clouds on the
same grid occupy the same cell iff they produce the same key, so overlap is the intersection
of their key sets. Density collapses: many points in a cell count once.
"""

from dataclasses import dataclass
from typing import NamedTuple

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


def _containment(n_shared: int, size1: int, size2: int) -> float:
    """Larger containment of two sets, in [0, 1], from their shared cells and sizes. 0 if either
    set is empty.

    `shared` is the same count from either side, so dividing it by the smaller set gives the
    larger containment: a small cloud lying inside a bigger one scores 1.
    """
    # NOTE: if density were taken into account (cells weighted by point count), the shared
    # weight would differ per side, so both containments would have to be computed to keep the max.
    smaller = min(size1, size2)
    return n_shared / smaller if smaller else 0.0


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

    def containment(self, other: "VoxelSet") -> float:
        """Larger containment of the two sets, in [0, 1] (see `_containment`)."""
        return _containment(self.shared(other), len(self), len(other))


class OverlapPair(NamedTuple):
    """Two sets that share cells, `first` before `second` in the dict, and their containment."""
    first: int
    second: int
    overlap: float


def overlapping_pairs(sets: dict[int, VoxelSet]) -> list[OverlapPair]:
    """An `OverlapPair` for every pair of sets sharing at least one cell, in the order a double
    loop over `sets` would visit them.

    A pair missing from the result shares no cell, so its overlap is 0. One sort of all cells
    replaces calling `containment` on every pair.
    """
    if not sets: return []
    keys = list(sets)
    cells, owners = _flatten(sets)                                   # every cell, and the set it came from
    owners, groups = _group_by_cell(cells, owners)                   # equal cells together, numbered
    pairs = _count_shared_pairs(owners, groups, len(keys))           # owner pairs, by position
    return [OverlapPair(keys[p.first], keys[p.second],
                        _containment(p.n_shared, len(sets[keys[p.first]]), len(sets[keys[p.second]])))
            for p in pairs]


class _SharedPair(NamedTuple):
    """Two set positions sharing cells, `first` < `second`, and how many cells they share."""
    first: int
    second: int
    n_shared: int


def _flatten(sets: dict[int, VoxelSet]) -> tuple[torch.Tensor, torch.Tensor]:
    """All cells of all sets in one tensor, `cells`, and alongside it `owners`: for each cell, the
    position (0, 1, 2...) of the set it came from. cells[i] belongs to set owners[i].

    Positions follow the dict order, so sorting by them later gives double-loop order.
    """
    cells = torch.cat([voxels.cells for voxels in sets.values()])
    owners = torch.cat([torch.full_like(voxels.cells, pos) for pos, voxels in enumerate(sets.values())])
    return cells, owners


def _group_by_cell(cells: torch.Tensor, owners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort by cell, so equal cells sit next to each other, and number each cell: `groups[i]` is
    the same for all entries of one cell. Returns `owners` in that order, and `groups`.

    cells [12, 40, 12, 13] -> sorted [12, 12, 13, 40] -> groups [0, 0, 1, 2]
    """
    order = torch.argsort(cells)
    _, groups = torch.unique_consecutive(cells[order], return_inverse=True)
    return owners[order], groups


def _count_shared_pairs(owners: torch.Tensor, groups: torch.Tensor, n_sets: int) -> list[_SharedPair]:
    """A `_SharedPair` of owner positions for every pair sharing at least one cell, sorted by
    (first, second), which is the double-loop order.

    Two entries in the same group are one cell shared by their two owners. Each entry is paired
    with the one `gap` places ahead, for growing `gap`, while some group is still that long.
    A set holds each cell once, so the two owners of a pair are always different.
    """
    pair_codes = []
    gap = 1
    while gap < len(groups):
        in_same_group = groups[gap:] == groups[:-gap]
        if not bool(in_same_group.any()):
            break
        a, b = owners[:-gap][in_same_group], owners[gap:][in_same_group]
        pair_codes.append(torch.minimum(a, b) * n_sets + torch.maximum(a, b))  # one integer per pair
        gap += 1
    if not pair_codes:
        return []

    # each code appears once per shared cell, so counting codes gives the shared cells per pair
    codes, counts = torch.unique(torch.cat(pair_codes), return_counts=True)
    return [_SharedPair(*divmod(code, n_sets), n_shared=count)  # divmod undoes `first * n_sets + second`
            for code, count in zip(codes.tolist(), counts.tolist())]
