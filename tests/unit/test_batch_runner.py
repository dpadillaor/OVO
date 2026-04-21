"""
Unit tests for scripts/run_experiments_batch.py

Imports are done via importlib so the scripts/ directory doesn't need to be
a Python package.
"""

import importlib.util
import sys
import datetime
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import the module under test without making scripts/ a package
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_MODULE_PATH = _SCRIPTS_DIR / "run_experiments_batch.py"

spec = importlib.util.spec_from_file_location("run_experiments_batch", _MODULE_PATH)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

# Aliases for convenience
_update_recursive = _mod._update_recursive
_load_experiment_manifest = _mod._load_experiment_manifest
ExperimentRunner = _mod.ExperimentRunner
Experiment = _mod.Experiment
Manifest = _mod.Manifest
OVOConfigOverride = _mod.OVOConfigOverride
SLAMConfigOverride = _mod.SLAMConfigOverride


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(
    *,
    slam_module: str = "groundtruth",
    noise: dict | None = None,
    slam_overrides: dict | None = None,
    semantic_overrides: dict | None = None,
    label: str = "TestLabel",
    dataset: str = "Replica",
    ovo_config_path: str | None = None,
    slam_config_path: str | None = None,
) -> ExperimentRunner:
    """Build an ExperimentRunner with full control over its parameters."""
    ovo_slam = {**{"slam_module": slam_module}, **(slam_overrides or {})}
    experiment = Experiment(
        label=label,
        ovo_config=OVOConfigOverride(
            slam=ovo_slam,
            semantic=semantic_overrides or {},
        ),
        slam_config=SLAMConfigOverride(noise=noise or {}),
        dataset=dataset,
    )
    manifest = Manifest(default_dataset=dataset)
    runner = ExperimentRunner(experiment, manifest)
    if ovo_config_path:
        runner.ovo_config_path = ovo_config_path
    if slam_config_path:
        runner.slam_config_path = slam_config_path
    return runner


# ---------------------------------------------------------------------------
# TestUpdateRecursive
# ---------------------------------------------------------------------------

class TestUpdateRecursive:

    def test_simple_leaf_merge(self):
        d = {"a": 1, "b": 2}
        _update_recursive(d, {"b": 99, "c": 3})
        assert d == {"a": 1, "b": 99, "c": 3}

    def test_nested_preserves_untouched_branches(self):
        d = {"slam": {"module": "gt", "use_viewer": False}, "other": True}
        _update_recursive(d, {"slam": {"module": "orbslam3"}})
        assert d["slam"]["use_viewer"] is False
        assert d["slam"]["module"] == "orbslam3"
        assert d["other"] is True

    def test_deep_override(self):
        d = {"a": {"b": {"c": 1, "d": 2}}}
        _update_recursive(d, {"a": {"b": {"c": 99}}})
        assert d["a"]["b"]["c"] == 99
        assert d["a"]["b"]["d"] == 2


# ---------------------------------------------------------------------------
# TestGetSlamToken
# ---------------------------------------------------------------------------

class TestGetSlamToken:

    def test_groundtruth_no_noise(self):
        runner = _make_runner(slam_module="groundtruth")
        assert runner._get_slam_token() == "GT"

    def test_groundtruth_with_translation_noise(self):
        runner = _make_runner(
            slam_module="groundtruth",
            noise={"translation_noise_std": 0.001, "rotation_noise_std": 0.01},
        )
        token = runner._get_slam_token()
        assert token == "GTNoise-T0p001-R0p01"

    def test_orbslam2(self):
        runner = _make_runner(slam_module="orbslam2")
        assert runner._get_slam_token() == "orbslam2"

    def test_vanilla(self):
        runner = _make_runner(slam_module="vanilla")
        assert runner._get_slam_token() == "Vanilla"

    def test_unknown_module_raises(self):
        with pytest.raises(ValueError):
            _make_runner(slam_module="unknown_slam")


# ---------------------------------------------------------------------------
# TestGetFusionToken
# ---------------------------------------------------------------------------

class TestGetFusionToken:

    def test_clip(self):
        runner = _make_runner(semantic_overrides={"fusion_method": "clip"})
        assert runner._get_fusion_token() == "CLIP"

    def test_none_defaults_to_clip(self):
        runner = _make_runner()
        assert runner._get_fusion_token() == "CLIP"

    def test_pe_core(self):
        runner = _make_runner(
            semantic_overrides={
                "fusion_method": "pe",
                "pe": {"model_card": "PE-Core-L14-336"},
            }
        )
        assert runner._get_fusion_token() == "PE-Core"

    def test_pe_spatial(self):
        runner = _make_runner(
            semantic_overrides={
                "fusion_method": "pe",
                "pe": {"model_card": "PE-Spatial-L14-448"},
            }
        )
        assert runner._get_fusion_token() == "PE-Spatial"

    def test_pe_no_model_card(self):
        runner = _make_runner(semantic_overrides={"fusion_method": "pe"})
        assert runner._get_fusion_token() == "PE"

    def test_sam3(self):
        runner = _make_runner(semantic_overrides={"fusion_method": "sam3"})
        assert runner._get_fusion_token() == "SAM3"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            _make_runner(semantic_overrides={"fusion_method": "unknown_method"})


# ---------------------------------------------------------------------------
# TestGenerateExperimentName
# ---------------------------------------------------------------------------

class TestGenerateExperimentName:

    def test_format_four_parts(self):
        runner = _make_runner(label="MyLabel")
        name = runner._generate_experiment_name()
        parts = name.split("_")
        assert len(parts) == 4, f"Expected 4 parts, got {len(parts)}: {name}"

    def test_date_matches_today(self):
        runner = _make_runner()
        name = runner._generate_experiment_name()
        today = datetime.datetime.now().strftime("%Y%m%d")
        assert name.startswith(today)

    def test_label_in_last_position(self):
        runner = _make_runner(label="SpecialTag")
        name = runner._generate_experiment_name()
        assert name.endswith("_SpecialTag")


# ---------------------------------------------------------------------------
# TestConfigOverrides
# ---------------------------------------------------------------------------

_OVO_BASE = {
    "slam": {"slam_module": "groundtruth"},
    "noise": {
        "noise_enabled": False,
        "translation_noise_std": 0.0,
        "rotation_noise_std": 0.0,
    },
    "semantic": {"fusion_method": "clip"},
}

_SLAM_BASE = {
    "dataset_name": "Replica",
    "kf_dist_thresh": 0.1,
}


class TestConfigOverrides:

    def _setup_files(self, tmp_path: Path) -> tuple[Path, Path]:
        ovo_path = tmp_path / "ovo.yaml"
        slam_path = tmp_path / "slam_config.yaml"
        ovo_path.write_text(yaml.dump(_OVO_BASE))
        slam_path.write_text(yaml.dump(_SLAM_BASE))
        return ovo_path, slam_path

    def test_slam_module_updated_in_ovo(self, tmp_path):
        ovo_path, slam_path = self._setup_files(tmp_path)
        runner = _make_runner(
            slam_module="orbslam2",
            ovo_config_path=str(ovo_path),
            slam_config_path=str(slam_path),
        )
        runner._apply_config_overrides()
        result = yaml.full_load(ovo_path.read_text())
        assert result["slam"]["slam_module"] == "orbslam2"

    def test_fusion_method_updated_in_ovo(self, tmp_path):
        ovo_path, slam_path = self._setup_files(tmp_path)
        runner = _make_runner(
            semantic_overrides={"fusion_method": "pe"},
            ovo_config_path=str(ovo_path),
            slam_config_path=str(slam_path),
        )
        runner._apply_config_overrides()
        result = yaml.full_load(ovo_path.read_text())
        assert result["semantic"]["fusion_method"] == "pe"

    def test_noise_enabled_set_true_when_noise_specified(self, tmp_path):
        ovo_path, slam_path = self._setup_files(tmp_path)
        runner = _make_runner(
            noise={"translation_noise_std": 0.005, "rotation_noise_std": 0.02},
            ovo_config_path=str(ovo_path),
            slam_config_path=str(slam_path),
        )
        runner._apply_config_overrides()
        result = yaml.full_load(ovo_path.read_text())
        assert result["noise"]["noise_enabled"] is True

    def test_translation_noise_std_written_correctly(self, tmp_path):
        ovo_path, slam_path = self._setup_files(tmp_path)
        runner = _make_runner(
            noise={"translation_noise_std": 0.005, "rotation_noise_std": 0.02},
            ovo_config_path=str(ovo_path),
            slam_config_path=str(slam_path),
        )
        runner._apply_config_overrides()
        result = yaml.full_load(ovo_path.read_text())
        assert abs(result["noise"]["translation_noise_std"] - 0.005) < 1e-9

    def test_no_noise_leaves_noise_enabled_false(self, tmp_path):
        ovo_path, slam_path = self._setup_files(tmp_path)
        runner = _make_runner(
            ovo_config_path=str(ovo_path),
            slam_config_path=str(slam_path),
        )
        runner._apply_config_overrides()
        result = yaml.full_load(ovo_path.read_text())
        assert result["noise"]["noise_enabled"] is False

    def test_backups_created_by_setup(self, tmp_path):
        ovo_path, slam_path = self._setup_files(tmp_path)
        runner = _make_runner(
            ovo_config_path=str(ovo_path),
            slam_config_path=str(slam_path),
        )
        runner.ovo_backup_path = str(ovo_path) + ".bak"
        runner.slam_backup_path = str(slam_path) + ".bak"
        runner.setup()
        assert Path(runner.ovo_backup_path).exists()
        assert Path(runner.slam_backup_path).exists()

    def test_cleanup_removes_backups_and_restores_ovo(self, tmp_path):
        ovo_path, slam_path = self._setup_files(tmp_path)
        original_content = ovo_path.read_text()
        runner = _make_runner(
            noise={"translation_noise_std": 0.01, "rotation_noise_std": 0.05},
            ovo_config_path=str(ovo_path),
            slam_config_path=str(slam_path),
        )
        runner.ovo_backup_path = str(ovo_path) + ".bak"
        runner.slam_backup_path = str(slam_path) + ".bak"
        runner.setup()
        runner.cleanup()

        # Backups removed
        assert not Path(runner.ovo_backup_path).exists()
        assert not Path(runner.slam_backup_path).exists()

        # ovo.yaml restored to original content (parse to ignore whitespace diffs)
        restored = yaml.full_load(ovo_path.read_text())
        original_parsed = yaml.full_load(original_content)
        assert restored == original_parsed


# ---------------------------------------------------------------------------
# TestScenesArg
# ---------------------------------------------------------------------------

class TestScenesArg:
    """Verify the scenes_arg string built in ExperimentRunner.run()."""

    def _scenes_arg_for(self, *, scenes_id=None, scenes_list=None) -> str:
        """
        Replicate the logic from run() without actually launching a subprocess.
        """
        experiment = Experiment(
            label="X",
            scenes_id=scenes_id,
            scenes_list=scenes_list,
            ovo_config=OVOConfigOverride(slam={"slam_module": "groundtruth"}),
        )
        if scenes_list:
            return f"--scenes_list {scenes_list}"
        elif scenes_id:
            if isinstance(scenes_id, list):
                return "--scenes " + " ".join(scenes_id)
            else:
                return f"--scenes {scenes_id}"
        return ""

    def test_single_scene_id(self):
        assert self._scenes_arg_for(scenes_id="office0") == "--scenes office0"

    def test_list_of_scene_ids(self):
        arg = self._scenes_arg_for(scenes_id=["room0", "room1"])
        assert arg == "--scenes room0 room1"

    def test_scenes_list_file(self):
        arg = self._scenes_arg_for(scenes_list="scenes.txt")
        assert arg == "--scenes_list scenes.txt"

    def test_no_scenes(self):
        assert self._scenes_arg_for() == ""

    def test_scenes_list_takes_priority_over_scenes_id(self):
        arg = self._scenes_arg_for(scenes_id="office0", scenes_list="scenes.txt")
        assert arg.startswith("--scenes_list")


# ---------------------------------------------------------------------------
# TestLoadManifest
# ---------------------------------------------------------------------------

class TestLoadManifest:

    def _write_manifest(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "manifest.yaml"
        p.write_text(content)
        return p

    def test_experiment_count(self, tmp_path):
        manifest_yaml = """
default_dataset: Replica
experiments:
  - label: exp1
    ovo_config:
      slam:
        slam_module: groundtruth
  - label: exp2
    ovo_config:
      slam:
        slam_module: groundtruth
"""
        path = self._write_manifest(tmp_path, manifest_yaml)
        manifest = _load_experiment_manifest(path)
        assert len(manifest.experiments) == 2

    def test_scenes_id_as_string(self, tmp_path):
        manifest_yaml = """
default_dataset: Replica
experiments:
  - label: exp1
    scenes_id: office0
    ovo_config:
      slam:
        slam_module: groundtruth
"""
        path = self._write_manifest(tmp_path, manifest_yaml)
        manifest = _load_experiment_manifest(path)
        assert manifest.experiments[0].scenes_id == "office0"

    def test_scenes_id_as_list(self, tmp_path):
        manifest_yaml = """
default_dataset: Replica
experiments:
  - label: exp1
    scenes_id:
      - room0
      - room1
    ovo_config:
      slam:
        slam_module: groundtruth
"""
        path = self._write_manifest(tmp_path, manifest_yaml)
        manifest = _load_experiment_manifest(path)
        assert manifest.experiments[0].scenes_id == ["room0", "room1"]

    def test_scenes_list_field(self, tmp_path):
        manifest_yaml = """
default_dataset: Replica
experiments:
  - label: exp1
    scenes_list: scenes.txt
    ovo_config:
      slam:
        slam_module: groundtruth
"""
        path = self._write_manifest(tmp_path, manifest_yaml)
        manifest = _load_experiment_manifest(path)
        assert manifest.experiments[0].scenes_list == "scenes.txt"

    def test_fusion_method_at_root_moved_to_semantic(self, tmp_path):
        manifest_yaml = """
default_dataset: Replica
experiments:
  - label: exp1
    ovo_config:
      slam:
        slam_module: groundtruth
      fusion_method: pe
"""
        path = self._write_manifest(tmp_path, manifest_yaml)
        manifest = _load_experiment_manifest(path)
        exp = manifest.experiments[0]
        assert exp.ovo_config.semantic.get("fusion_method") == "pe"

    def test_default_dataset_used_when_no_per_experiment_dataset(self, tmp_path):
        manifest_yaml = """
default_dataset: ScanNet
experiments:
  - label: exp1
    ovo_config:
      slam:
        slam_module: groundtruth
"""
        path = self._write_manifest(tmp_path, manifest_yaml)
        manifest = _load_experiment_manifest(path)
        assert manifest.default_dataset == "ScanNet"
        # ExperimentRunner should fall back to default_dataset
        exp = manifest.experiments[0]
        assert exp.dataset is None
        runner = ExperimentRunner(exp, manifest)
        assert runner.dataset == "ScanNet"
