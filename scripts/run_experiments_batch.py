import yaml
import subprocess
import shutil
import json
import os
import datetime
import sys
import itertools
import time
import threading
import argparse
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

# --- ANSI Colors Helper ---
class Colors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- Dataclasses for Manifest Structure ---
@dataclass
class OVOConfigOverride:
    slam: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    vis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SLAMConfigOverride:
    noise: Dict[str, Any] = field(default_factory=dict)

@dataclass # Represents a single experiment entry in the 'experiments' list of the manifest
class Experiment:
    label: str
    scenes_id: Optional[str] = None          # single scene name or list of names
    scenes_list: Optional[str] = None        # path to a .txt file with one scene per line
    stages: List[str] = field(default_factory=lambda: ["run", "segment", "eval"])
    ovo_config: OVOConfigOverride = field(default_factory=OVOConfigOverride)
    slam_config: SLAMConfigOverride = field(default_factory=SLAMConfigOverride)
    dataset: Optional[str] = None

@dataclass # Represents the entire experiments_manifest.yaml file
class Manifest:
    default_dataset: str
    experiments: List[Experiment] = field(default_factory=list)

class ExperimentRunner:
    """
    Manages the execution of a single OVO experiment, including
    configuration setup, running run_eval.py, and cleanup.
    """
    def __init__(self, experiment: Experiment, manifest: Manifest, verbose: bool = False):
        # Basic attributes
        self.label = experiment.label   
        self.experiment = experiment
        self.manifest = manifest 
        self.verbose = verbose
        self.dataset = self.experiment.dataset if self.experiment.dataset else self.manifest.default_dataset
        self.slam_module = self.experiment.ovo_config.slam.get("slam_module", "simulated")

        # Paths
        self.ovo_config_path = "data/working/configs/ovo.yaml"
        self.slam_config_path = f"data/working/configs/slam/{self.slam_module}/{self.dataset.lower()}.yaml"
        self.ovo_backup_path = f"{self.ovo_config_path}.bak"
        self.slam_backup_path = f"{self.slam_config_path}.bak"
        
        # Experiment identifier
        self.experiment_name = self._generate_experiment_name()

    def _get_slam_token(self) -> str:
        """
        Generates the SLAM token part of the experiment name.
        """
        match self.slam_module:
            case "simulated":
                if self.experiment.slam_config.noise.get("jump_drift_enabled", False):
                    jumps = self.experiment.slam_config.noise.get("jumps", [])
                    n_jumps = len(jumps)
                    first = jumps[0] if jumps else {}
                    t_mag = first.get("translation_magnitude", 0.0)
                    r_mag = first.get("rotation_magnitude", 0.0)
                    t_str = str(t_mag).replace('.', 'p')
                    r_str = str(r_mag).replace('.', 'p')
                    return f"GTJump-J{n_jumps}-T{t_str}-R{r_str}"
                t_std = self.experiment.slam_config.noise.get("translation_noise_std", 0.0)
                r_std = self.experiment.slam_config.noise.get("rotation_noise_std", 0.0)
                if t_std > 0.0 or r_std > 0.0:
                    t_str = str(t_std).replace('.', 'p')
                    r_str = str(r_std).replace('.', 'p')
                    return f"GTNoise-T{t_str}-R{r_str}"
                else:
                    return "GT"
                    
            case "orbslam2":
                return "orbslam2"
            case "vanilla":
                return "Vanilla"
            case _:
                raise ValueError(f"Slam module '{self.slam_module}' not recognized or supported by experiment runner")

    def _get_fusion_token(self) -> str:
        """
        Generates the fusion config part of the experiment name.
        """
        method = self.experiment.ovo_config.semantic.get("fusion_method")
        method_safe = method.lower() if method else None

        match method_safe:
            case "clip" | None:
                return "CLIP"
            case "dino":
                return "DINO"
            case "pe":
                model_card = self.experiment.ovo_config.semantic.get("pe", {}).get("model_card", "")
                if "Spatial" in model_card:
                    return "PE-Spatial"
                elif "Core" in model_card:
                    return "PE-Core"
                else:
                    return "PE"
            case "sam3":
                 return "SAM3"
            case _:
                raise ValueError(f"Fusion method '{method}' not recognized or supported by experiment runner")


    def _generate_experiment_name(self) -> str:
        """
        Generates the experiment folder name based on the defined convention.
        Format: [DATE]_[SLAM_CONFIG]_[FUSION_CONFIG]_[LABEL]_[UID]
        UID is a 5-char hex suffix that guarantees uniqueness; all experiment
        parameters are stored in the experiment_meta.json sidecar.
        """
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        slam_token = self._get_slam_token()
        fusion_config_token = self._get_fusion_token()
        tag = self.experiment.label
        uid = uuid.uuid4().hex[:5]

        return f"{date_str}_{slam_token}_{fusion_config_token}_{tag}_{uid}"

    def _backup_configs(self):
        """Creates backups of the original config files."""
        print(f"    Backing up configs...")
        shutil.copy(self.ovo_config_path, self.ovo_backup_path)
        shutil.copy(self.slam_config_path, self.slam_backup_path)

    def _build_ovo_data(self) -> dict:
        """
        Reads ovo.yaml and applies all experiment overrides in memory.
        Does NOT write anything to disk. Used by both _apply_config_overrides and preview.
        """
        with open(self.ovo_config_path, 'r') as f:
            ovo_data = yaml.full_load(f)

        if self.experiment.ovo_config.slam:
            _update_recursive(ovo_data, {"slam": self.experiment.ovo_config.slam})

        if self.experiment.ovo_config.semantic:
            _update_recursive(ovo_data, {"semantic": self.experiment.ovo_config.semantic})

        if self.experiment.ovo_config.vis:
            _update_recursive(ovo_data, {"vis": self.experiment.ovo_config.vis})

        # Noise goes to ovo.yaml root (not to the slam config file).
        # SimulatedSLAM reads noise from config["noise"] which comes from ovo.yaml.
        if self.experiment.slam_config.noise:
            noise_cfg = self.experiment.slam_config.noise
            is_jump_drift = noise_cfg.get("jump_drift_enabled", False)
            _update_recursive(ovo_data, {
                "noise": {
                    **({"noise_enabled": True} if not is_jump_drift else {}),
                    **noise_cfg,
                }
            })

        return ovo_data

    def _apply_config_overrides(self):
        """Applies the experiment-specific config changes to the YAML files."""
        print(f"    Applying overrides for experiment '{self.label}'...")

        ovo_data = self._build_ovo_data()   # read before opening for write
        with open(self.ovo_config_path, 'w') as f:
            yaml.dump(ovo_data, f, default_flow_style=False, sort_keys=False)

        # Slam config: read and rewrite (no noise overrides — noise lives in ovo.yaml)
        # ORB-SLAM configs use OpenCV's %YAML:1.0 header which PyYAML cannot parse,
        # and have no Python-side overrides to apply, so skip the round-trip.
        if not self.slam_module.startswith("orbslam"):
            with open(self.slam_config_path, 'r') as f:
                slam_data = yaml.full_load(f)
            with open(self.slam_config_path, 'w') as f:
                yaml.dump(slam_data, f, default_flow_style=False, sort_keys=False)

    def preview(self, output_dir: Path) -> Path:
        """
        Computes the merged ovo.yaml for this experiment and writes it to
        output_dir/{experiment_name}.yaml. Does NOT touch any real config file.
        Returns the path of the written file.
        """
        ovo_data = self._build_ovo_data()
        out_path = output_dir / f"{self.experiment_name}.yaml"
        with open(out_path, 'w') as f:
            yaml.dump(ovo_data, f, default_flow_style=False, sort_keys=False)
        return out_path

    def _restore_configs(self):
        """
        Restores original config files from backups and cleans up backup files.
        """
        print(f"    Restoring original configs...")
        shutil.copy(self.ovo_backup_path, self.ovo_config_path)
        os.remove(self.ovo_backup_path)
        shutil.copy(self.slam_backup_path, self.slam_config_path)
        os.remove(self.slam_backup_path)

    def _write_meta_json(self) -> None:
        """Write experiment_meta.json sidecar to the output directory."""
        ovo_data = self._build_ovo_data()
        semantic = ovo_data.get("semantic", {})
        noise    = ovo_data.get("noise", {})
        slam     = ovo_data.get("slam", {})

        is_jump    = noise.get("jump_drift_enabled", False)
        jumps      = noise.get("jumps", [])
        jump_count = len(jumps) if is_jump else 0
        trans_noise = 0.0 if is_jump else float(noise.get("translation_noise_std", 0.0))
        rot_noise   = 0.0 if is_jump else float(noise.get("rotation_noise_std", 0.0))

        meta = {
            "date":                       datetime.datetime.now().strftime("%Y%m%d"),
            "label":                      self.label,
            "dataset":                    self.dataset,
            "slam_module":                self.slam_module,
            "close_loops":                slam.get("close_loops", True),
            "trans_noise":                trans_noise,
            "rot_noise":                  rot_noise,
            "jump_count":                 jump_count,
            "fusion_method":              semantic.get("fusion_method", "clip"),
            "fusion_criteria":            semantic.get("fusion_criteria", None),
            "th_centroid":                float(semantic.get("th_centroid", 1.5)),
            "th_aabb":                    float(semantic.get("th_aabb", 0.3)),
            "th_cossim":                  float(semantic.get("th_cossim", 0.81)),
            "th_points":                  float(semantic.get("th_points", 0.1)),
            "cooccurrence_veto_threshold": int(semantic.get("cooccurrence_veto_threshold", 5)),
        }

        out_dir = Path(f"data/output/{self.dataset}/{self.experiment_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "experiment_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def setup(self):
        """
        Prepares the environment for the experiment.
        """
        print(f"    Generated Name: {Colors.BOLD}{self.experiment_name}{Colors.ENDC}")
        self._backup_configs()
        self._apply_config_overrides()
        self._write_meta_json()

    def run(self):
        """
        Executes the run_eval.py command for the experiment.
        """
        # Build scenes argument: scenes_list (file) takes priority, then scenes_id (names), then nothing
        if self.experiment.scenes_list:
            scenes_arg = f"--scenes_list {self.experiment.scenes_list}"
            scenes_display = f"(list: {self.experiment.scenes_list})"
        elif self.experiment.scenes_id:
            if isinstance(self.experiment.scenes_id, list):
                scenes_arg = "--scenes " + " ".join(self.experiment.scenes_id)
            else:
                scenes_arg = f"--scenes {self.experiment.scenes_id}"
            scenes_display = str(self.experiment.scenes_id)
        else:
            scenes_arg = ""
            scenes_display = "(from dataset config)"

        # Build stage flags dynamically from experiment.stages
        stage_flags = " ".join([f"--{stage}" for stage in self.experiment.stages])

        command = (
            f"python run_eval.py --dataset_name {self.dataset} "
            f"--experiment_name {self.experiment_name} {scenes_arg} "
            f"{stage_flags}"
        )
        print(f"    {Colors.BOLD}Executing run_eval.py for:{Colors.ENDC} {self.dataset} - {self.experiment_name} - Scenes: {scenes_display}")
        print(f"    Stages: {', '.join(self.experiment.stages)}")
        
        if self.verbose:
            # Run without spinner and let output flow to stdout
            print(f"    {Colors.OKBLUE}Command:{Colors.ENDC} {command}")
            process_result = subprocess.run(command, shell=True, check=False)
        else:
            # Capture the process result for potential error reporting
            process_result = self._run_with_spinner(command)
        
        # Check if there was an error in the subprocess
        if process_result.returncode != 0:
            raise subprocess.CalledProcessError(
                process_result.returncode,
                process_result.args,
                output=process_result.stdout if not self.verbose else None,
                stderr=process_result.stderr if not self.verbose else None
            )

    def _run_with_spinner(self, command: str) -> subprocess.CompletedProcess:
        import os
        import signal

        stop_event = threading.Event()
        t = threading.Thread(target=_spinner, args=("    Running experiment", stop_event), daemon=True)
        t.start()
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group so SIGTERM reaches children
        )
        try:
            stdout, stderr = proc.communicate()
            return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            raise
        finally:
            stop_event.set()
            t.join()
            # Clear the spinner line once the thread has stopped
            sys.stdout.write("\r" + " " * (len("Running experiment...") + 5) + "\r") # Clear the line
            sys.stdout.flush()


    def cleanup(self):
        """Cleans up the environment after the experiment."""
        self._restore_configs()


def _update_recursive(d: Dict[Any, Any], u: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively updates a dictionary `d` with values from dictionary `u`.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def _load_experiment_manifest(manifest_path: Path) -> Manifest:
    print(f"\nLoading experiments from: {Colors.BOLD}{manifest_path}{Colors.ENDC}")
    with open(manifest_path, 'r') as f:
        manifest_dict = yaml.full_load(f)

    loaded_experiments: List[Experiment] = []
    
    for exp_data in manifest_dict.get("experiments", []):
        ovo_data = exp_data.get("ovo_config", {})
        
        # Helper: Allow 'fusion_method' at root of ovo_config for convenience
        if "fusion_method" in ovo_data and "fusion_method" not in ovo_data.get("semantic", {}):
            if "semantic" not in ovo_data:
                ovo_data["semantic"] = {}
            ovo_data["semantic"]["fusion_method"] = ovo_data["fusion_method"]

        ovo_override = OVOConfigOverride(
            slam=ovo_data.get("slam", {}),
            semantic=ovo_data.get("semantic", {}),
            vis=ovo_data.get("vis", {}),
        )
        
        slam_data = exp_data.get("slam_config", {})
        slam_override = SLAMConfigOverride(
            noise=slam_data.get("noise", {})
        )
        
        experiment_obj = Experiment(
            label=exp_data["label"],
            scenes_id=exp_data.get("scenes_id"),
            scenes_list=exp_data.get("scenes_list"),
            stages=exp_data.get("stages", ["run", "segment", "eval"]),
            ovo_config=ovo_override,
            slam_config=slam_override,
            dataset=exp_data.get("dataset")
        )
        loaded_experiments.append(experiment_obj)

    print(f"\nLoaded {Colors.BOLD}{len(loaded_experiments)} experiments {Colors.ENDC}from manifest.")
        
    return Manifest(
        default_dataset=manifest_dict.get("default_dataset", "Replica"),
        experiments=loaded_experiments
    )


def _spinner(msg, stop_event):
    for c in itertools.cycle([".  ", ".. ", "..."]):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\r{msg}{c}")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r")

def main():
    parser = argparse.ArgumentParser(description="Run OVO experiments batch.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output (shows command output).")
    parser.add_argument("--preview", action="store_true",
                        help="Dry-run: compute and dump the merged ovo.yaml for every experiment "
                             "without running anything. Useful to verify configs before execution.")
    parser.add_argument("--preview-dir", default="data/working/config_preview",
                        help="Directory where preview configs are written (default: data/working/config_preview).")
    parser.add_argument("--manifest", default="manifests/experiments_manifest.yaml",
                        help="Path to the experiments manifest YAML (default: manifests/experiments_manifest.yaml).")
    args = parser.parse_args()

    manifest_path = args.manifest
    manifest = _load_experiment_manifest(manifest_path)

    # ------------------------------------------------------------------
    # Preview mode — dump merged ovo.yaml per experiment, no execution
    # ------------------------------------------------------------------
    if args.preview:
        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nPreview mode — writing configs to: {Colors.BOLD}{preview_dir}{Colors.ENDC}\n")
        for experiment in manifest.experiments:
            runner = ExperimentRunner(experiment, manifest)
            out = runner.preview(preview_dir)
            print(f"  {Colors.OKGREEN}OK{Colors.ENDC}  {out.name}")
        print(f"\n{Colors.BOLD}{len(manifest.experiments)} config(s) written.{Colors.ENDC}\n")
        return

    for experiment in manifest.experiments:
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}=== Starting Experiment: {experiment.label} ==={Colors.ENDC}")
        runner = ExperimentRunner(experiment, manifest, verbose=args.verbose)
        try:
            runner.setup()
            runner.run()
            print(f"{Colors.OKGREEN}    Experiment {runner.label} COMPLETED{Colors.ENDC}")
        except subprocess.CalledProcessError as e:
            print(f"\n{Colors.FAIL}{Colors.BOLD}    !!! Experiment {runner.label} FAILED (subprocess error) !!!{Colors.ENDC}")
            if e.stdout:
                print(f"    Stdout:\n{e.stdout}")
            if e.stderr:
                print(f"    Stderr:\n{e.stderr}")
            print(f"    Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}")
        except Exception as e:
            print(f"\n{Colors.FAIL}{Colors.BOLD}    !!! Experiment {runner.label} FAILED (unexpected error) !!!{Colors.ENDC}")
            print(f"    Error: {e}")
        finally:
            runner.cleanup()
        
    print(f"\nAll experiments finished.\n")

if __name__ == "__main__":
    main()
