"""`python -m ovo replay` (ovo.__main__), and the checkpoint call in OVOSemMap.run."""

import inspect

import pytest
import yaml

from ovo import __main__ as cli
from ovo import layout
from ovo.replay import checkpoint
from replay_fakes import SCENE_NAME, config, scene


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "OUTPUT_ROOT", tmp_path / "output")
    exp = layout.ExperimentLayout.named("Replica", "exp")
    ovo, _, slam = scene()
    saver = checkpoint.Checkpointer("all", exp.scene(SCENE_NAME))
    saver.save(100, slam, ovo, config())
    saver.save(200, slam, ovo, config())
    return exp


def test_replay_with_a_fusion_file(experiment, tmp_path, capsys):
    fusion = tmp_path / "strict.yaml"
    fusion.write_text("gates: [centroid, aabb]\naccept:\n  min_cos: 0.99\n")

    cli.main(["replay", "--dataset", "Replica", "--experiment", "exp", "--name", "strict",
              "--fusion", str(fusion), "--scenes", SCENE_NAME, "--frame", "100", "--device", "cpu"])

    out = experiment.replay("strict")
    cfg = yaml.safe_load((out.scene(SCENE_NAME) / "config.yaml").read_text())
    assert cfg["semantic"]["instance_fusion"] == {"gates": ["centroid", "aabb"], "accept": {"min_cos": 0.99}}
    assert cfg["replayed_from"]["frame_id"] == 100
    assert (out.scene(SCENE_NAME) / "ovo_map.ckpt").exists()
    assert "--experiment exp/replays/strict" in capsys.readouterr().out


def test_replay_defaults_keep_the_original_fusion_and_take_the_last_checkpoint(experiment):
    cli.main(["replay", "--dataset", "Replica", "--experiment", "exp", "--name", "same", "--device", "cpu"])
    cfg = yaml.safe_load((experiment.replay("same").scene(SCENE_NAME) / "config.yaml").read_text())
    assert cfg["semantic"]["instance_fusion"] == config()["semantic"]["instance_fusion"]
    assert cfg["replayed_from"]["frame_id"] == 200


def test_a_fusion_file_with_the_semantic_block_around_it_is_rejected(experiment, tmp_path):
    fusion = tmp_path / "wrapped.yaml"
    fusion.write_text("semantic:\n  instance_fusion:\n    accept:\n      min_cos: 0.9\n")
    with pytest.raises(ValueError, match="unknown key"):
        cli.main(["replay", "--dataset", "Replica", "--experiment", "exp", "--name", "x",
                  "--fusion", str(fusion), "--device", "cpu"])
    assert not (experiment.root / "replays").exists()


@pytest.mark.parametrize("argv", [
    ["replay", "--dataset", "Replica", "--experiment", "exp"],  # no --name
    ["replay", "--dataset", "Replica", "--name", "x"],          # no --experiment
    ["replay", "--dataset", "Replica", "--experiment", "exp", "--name", "x", "--frame", "last"],
])
def test_bad_arguments_exit(argv):
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_ovomapping_checkpoints_right_before_the_fusion():
    from ovo.entities.ovomapping import OVOSemMap

    source = inspect.getsource(OVOSemMap.run)
    save = source.index("self.checkpointer.save(frame_id, self.slam_backbone, self.ovo, self.config)")
    fuse = source.index("self.ovo.update_map(map_data)")
    between = [line.strip() for line in source[save:fuse].splitlines()[1:-1]]
    assert save < fuse and between == ["map_data = self.slam_backbone.get_map()"]
    assert "Checkpointer(checkpoint_mode(config), self.output_path)" in inspect.getsource(OVOSemMap.__init__)
