"""Instance benchmarks (ovo.eval.benchmarks): class lists, frequency splits, dataset lookup."""

import dataclasses

import pytest

from ovo.eval.benchmarks import REPLICA, SCANNET200, InstanceBenchmark, for_dataset

BENCHMARKS = [REPLICA, SCANNET200]


@pytest.mark.parametrize("benchmark, n_classes", [(REPLICA, 48), (SCANNET200, 198)])
def test_class_count(benchmark, n_classes):
    assert len(benchmark.class_ids) == len(benchmark.class_names) == n_classes


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda b: b.name)
def test_ids_and_names_are_unique_and_valid(benchmark):
    assert len(set(benchmark.class_ids)) == len(benchmark.class_ids)
    assert len(set(benchmark.class_names)) == len(benchmark.class_names)
    # 0 is void in the GT and ids are encoded as class_id * 1000 + k
    assert all(isinstance(i, int) and 0 < i for i in benchmark.class_ids)


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda b: b.name)
def test_splits_partition_the_class_names(benchmark):
    splits = benchmark.head + benchmark.common + benchmark.tail
    assert len(splits) == len(set(splits))
    assert set(splits) == set(benchmark.class_names)


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda b: b.name)
def test_background_is_not_an_instance_class(benchmark):
    assert not {"wall", "floor"} & set(benchmark.class_names)


def test_known_id_name_pairs():
    replica = dict(zip(REPLICA.class_ids, REPLICA.class_names))
    scannet = dict(zip(SCANNET200.class_ids, SCANNET200.class_names))
    assert replica[3] == "basket" and replica[7] == "bed"
    assert scannet[2] == "chair" and scannet[4] == "table"


@pytest.mark.parametrize("name, expected", [
    ("Replica", REPLICA),
    ("replica", REPLICA),
    ("REPLICA", REPLICA),
    ("ScanNet", SCANNET200),
    ("scannet", SCANNET200),
])
def test_for_dataset_is_case_insensitive(name, expected):
    assert for_dataset(name) is expected


@pytest.mark.parametrize("name", ["scannetv2", "ScanNetV2", "ScanNet200", "ScanNetPP", "replica_v2", ""])
def test_for_dataset_rejects_unknown_names(name):
    with pytest.raises(ValueError, match="No instance benchmark"):
        for_dataset(name)


def test_mismatched_ids_and_names_raise():
    with pytest.raises(ValueError, match="2 ids for 1 names"):
        InstanceBenchmark("toy", (1, 2), ("a",), (), (), ())


def test_split_class_outside_class_names_raises():
    with pytest.raises(ValueError, match=r"\['z'\]"):
        InstanceBenchmark("toy", (1, 2), ("a", "b"), ("a",), ("z",), ())


def test_minimal_benchmark_is_accepted_and_frozen():
    b = InstanceBenchmark("toy", (1,), ("a",), ("a",), (), ())
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.name = "other"
