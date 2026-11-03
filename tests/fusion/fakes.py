"""Fakes shared by the instance fusion tests: instances, voxel-aligned clouds, random scenes."""

import random
from dataclasses import dataclass

import torch

CELL = 0.05  # default voxel size of the fusion config


@dataclass
class FakeInstance:
    """The only attributes the fusion pipeline reads: an id and a (1, D) descriptor."""
    id: int
    clip_feature: torch.Tensor


def feature(*values) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.float32)


def cells(*ijk) -> torch.Tensor:
    """A cloud with one point at the centre of each given cell (i, j, k) of the default grid."""
    return (torch.tensor(ijk, dtype=torch.float64) + 0.5) * CELL


def line(start: int, stop: int) -> torch.Tensor:
    """One point per cell along x, cells start..stop-1."""
    return cells(*[(i, 0, 0) for i in range(start, stop)])


def random_scene(seed: int):
    """A random scene where instances are fragments of a few objects, as a mapper produces them.

    Each object is a box with a class; each instance covers a random sub-box of one object, so
    fragments of the same object overlap by any amount and boxes of different objects may cross.
    Descriptors are the class descriptor plus noise of varying strength, so pairs land on every
    side of min_cos and weak_min_cos. Ids are non-sequential and shuffled. Returns (instances, pcds).
    """
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    classes = torch.randn(2, 16, generator=gen)
    objects = []
    for _ in range(rng.randint(1, 4)):
        size = 0.1 + 0.3 * torch.rand(3, generator=gen, dtype=torch.float64)
        corner = 0.5 * torch.rand(3, generator=gen, dtype=torch.float64)
        objects.append((corner, size, rng.randrange(2)))

    instances, pcds = [], {}
    for ins_id in rng.sample(range(1000), rng.randint(2, 25)):
        corner, size, cls = objects[rng.randrange(len(objects))]
        lo = torch.rand(3, generator=gen, dtype=torch.float64) * 0.6
        hi = lo + (1 - lo) * torch.rand(3, generator=gen, dtype=torch.float64).clamp(min=0.2)
        n_points = rng.choice([5, 50, 300])
        unit = lo + (hi - lo) * torch.rand(n_points, 3, generator=gen, dtype=torch.float64)
        pcds[ins_id] = corner + size * unit
        sigma = rng.choice([0.2, 0.35, 0.5])  # same-class cos_sim around 1 / (1 + sigma^2)
        instances.append(FakeInstance(ins_id, (classes[cls] + sigma * torch.randn(16, generator=gen))[None]))
    return instances, pcds
