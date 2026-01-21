import yaml
import subprocess
import shutil
import os
import datetime
import sys
import itertools
import time
import threading
import argparse
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

@dataclass
class SLAMConfigOverride:
    noise: Dict[str, Any] = field(default_factory=dict)

@dataclass # Represents a single experiment entry in the 'experiments' list of the manifest
class Experiment:
    label: str
    scenes_id: str
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
        self.slam_module = self.experiment.ovo_config.slam.get("slam_module", "groundtruth")

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
            case "groundtruth":
                t_std = self.experiment.slam_config.noise.get("translation_noise_std", 0.0)
                r_std = self.experiment.slam_config.noise.get("rotation_noise_std", 0.0)
                
                if t_std > 0.0 or r_std > 0.0:
                    t_str = str(t_std).replace('.', 'p')
                    r_str = str(r_std).replace('.', 'p')
                    return f"GTNoise-T{t_str}-R{r_str}"
                else:
                    return "GT"
                    
            case "orbslam3":
                return "ORBSLAM3"
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
                 return "PE"
            case "sam3":
                 return "SAM3"
            case _:
                raise ValueError(f"Fusion method '{method}' not recognized or supported by experiment runner")


    def _generate_experiment_name(self) -> str:
        """
        Generates the experiment folder name based on the defined convention.
        Format: [DATE]_[SCENES_ID]_[SLAM_CONFIG]_[FUSION_CONFIG]_[DESCRIPTION]
        """
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        slam_token = self._get_slam_token()
        fusion_config_token = self._get_fusion_token()
        tag = self.experiment.label

        return f"{date_str}_{self.experiment.scenes_id}_{slam_token}_{fusion_config_token}_{tag}"

    def _backup_configs(self):
        """Creates backups of the original config files."""
        print(f"    Backing up configs...")
        shutil.copy(self.ovo_config_path, self.ovo_backup_path)
        shutil.copy(self.slam_config_path, self.slam_backup_path)

    def _apply_config_overrides(self):
        """Applies the experiment-specific config changes to the YAML files."""
        print(f"    Applying overrides for experiment '{self.label}'...")
        
        # Apply ovo_config overrides
        with open(self.ovo_config_path, 'r') as f:
            ovo_data = yaml.full_load(f)
        
        # Envolvemos en 'slam' si es necesario, pero como ovo_config.slam ya es el dict correcto para 'slam' section,
        # necesitamos ver si ovo_data tiene 'slam' o si ovo_config.slam debe mergearse en root.
        # ovo.yaml tiene 'slam' en root.
        if self.experiment.ovo_config.slam:
             _update_recursive(ovo_data, {"slam": self.experiment.ovo_config.slam})

        # Apply semantic config (including fusion_method)
        if self.experiment.ovo_config.semantic:
            _update_recursive(ovo_data, {"semantic": self.experiment.ovo_config.semantic})
        
        with open(self.ovo_config_path, 'w') as f:
            yaml.dump(ovo_data, f, default_flow_style=False)

        # Apply slam_config overrides
        with open(self.slam_config_path, 'r') as f:
            slam_data = yaml.full_load(f)
            
        if self.experiment.slam_config.noise:
            slam_overrides = {"noise": self.experiment.slam_config.noise}
            _update_recursive(slam_data, slam_overrides)
        
        with open(self.slam_config_path, 'w') as f:
            yaml.dump(slam_data, f, default_flow_style=False)

    def _restore_configs(self):
        """
        Restores original config files from backups and cleans up backup files.
        """
        print(f"    Restoring original configs...")
        shutil.copy(self.ovo_backup_path, self.ovo_config_path)
        os.remove(self.ovo_backup_path)
        shutil.copy(self.slam_backup_path, self.slam_config_path)
        os.remove(self.slam_backup_path)

    def setup(self):
        """
        Prepares the environment for the experiment.
        """
        print(f"    Generated Name: {Colors.BOLD}{self.experiment_name}{Colors.ENDC}")
        self._backup_configs()
        self._apply_config_overrides()

    def run(self):
        """
        Executes the run_eval.py command for the experiment.
        """
        # TODO: Determine if we use --scenes or --scenes_list
        scenes_arg = f"--scenes {self.experiment.scenes_id}"
        
        # Build stage flags dynamically from experiment.stages
        stage_flags = " ".join([f"--{stage}" for stage in self.experiment.stages])
        
        command = (
            f"python run_eval.py --dataset_name {self.dataset} "
            f"--experiment_name {self.experiment_name} {scenes_arg} "
            f"{stage_flags}"
        )
        print(f"    {Colors.BOLD}Executing run_eval.py for:{Colors.ENDC} {self.dataset} - {self.experiment_name} - Scenes: {self.experiment.scenes_id}")
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
        stop_event = threading.Event()
        # Pass the message to the spinner function
        t = threading.Thread(target=_spinner, args=("    Running experiment", stop_event), daemon=True)
        t.start()
        try:
            result = subprocess.run(command,
                                    shell=True,
                                    check=False, # We will check return code manually
                                    capture_output=True, # Capture stdout and stderr
                                    text=True) # Decode stdout/stderr as text
            return result
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
            semantic=ovo_data.get("semantic", {})
        )
        
        slam_data = exp_data.get("slam_config", {})
        slam_override = SLAMConfigOverride(
            noise=slam_data.get("noise", {})
        )
        
        experiment_obj = Experiment(
            label=exp_data["label"],
            scenes_id=exp_data["scenes_id"],
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
    args = parser.parse_args()

    manifest_path = "scripts/experiments_manifest.yaml"
    manifest = _load_experiment_manifest(manifest_path)

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
