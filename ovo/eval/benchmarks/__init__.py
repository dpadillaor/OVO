"""Instance-segmentation benchmarks: which GT classes count as objects, and their names.

The class index stored with each prediction is a position in ``InstanceBenchmark.class_names``,
so predictions must be classified against that same list (see ``ovo.eval.predictions``).
"""

from dataclasses import dataclass
from types import ModuleType

from . import replica, scannet200


@dataclass(frozen=True)
class InstanceBenchmark:
    name: str
    class_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    head: tuple[str, ...]
    common: tuple[str, ...]
    tail: tuple[str, ...]

    def __post_init__(self):
        if len(self.class_ids) != len(self.class_names):
            raise ValueError(f"{self.name}: {len(self.class_ids)} ids for {len(self.class_names)} names")
        unknown = set(self.head + self.common + self.tail) - set(self.class_names)
        if unknown:
            raise ValueError(f"{self.name}: split classes not in class_names: {sorted(unknown)}")

    @classmethod
    def from_module(cls, name: str, module: ModuleType) -> "InstanceBenchmark":
        return cls(name, module.CLASS_IDS, module.CLASS_NAMES, module.HEAD, module.COMMON, module.TAIL)


REPLICA = InstanceBenchmark.from_module("replica", replica)
SCANNET200 = InstanceBenchmark.from_module("scannet200", scannet200)

# ScanNet instances are only evaluated on ScanNet200: its instance GT uses the 200-class ids.
_BY_DATASET = {"replica": REPLICA, "scannet": SCANNET200}


def for_dataset(dataset_name: str) -> InstanceBenchmark:
    """Benchmark for a dataset as named on the command line (``Replica``, ``ScanNet``)."""
    try:
        return _BY_DATASET[dataset_name.lower()]
    except KeyError:
        raise ValueError(
            f"No instance benchmark for dataset {dataset_name!r}; known: {sorted(_BY_DATASET)}"
        ) from None
