"""get_slam_backbone: the slam_module names it accepts, and nothing else."""

import pytest
import torch

from ovo.entities.ovomapping import SLAM_MODULES, get_slam_backbone
from ovo.slam.vanilla_mapper import VanillaMapper

K = torch.eye(3)


def config(**slam):
    return {"device": "cpu", "mapping": {"map_every": 5}, "slam": slam}


def test_valid_names():
    assert SLAM_MODULES == ("vanilla", "orbslam3", "gaussian_slam")


@pytest.mark.parametrize("slam", [{}, {"slam_module": "vanilla"}])
def test_vanilla(slam):
    backbone = get_slam_backbone(config(**slam), dataset=[None] * 3, cam_intrinsics=K)
    assert type(backbone) is VanillaMapper


@pytest.mark.parametrize("name", ["orbslam2", "orbslam", "vanila", "Vanilla", "gaussian", ""])
def test_unknown_names_raise(name):
    with pytest.raises(ValueError, match="unknown") as err:
        get_slam_backbone(config(slam_module=name), dataset=[None], cam_intrinsics=K)
    for valid in SLAM_MODULES:
        assert valid in str(err.value)
