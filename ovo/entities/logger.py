from typing import Any, Dict
from pathlib import Path
import numpy as np
import psutil
import pprint
import torch
import wandb
import csv
import time

_FUSION_LOG_FIELDS = ["frame_id", "result", "accept_mode", "i1", "i2", "reason", "centroid_dist", "aabb_dist", "cos_sim", "p_dist", "shared_kfs"]

# Stats produced by the fusion step itself. A restore run regenerates these, so it must
# NOT inherit them from the source run's logs (see load_stats_from).
_FUSION_OWNED_STATS = {
    "t_fusion", "t_loop_closure_refusion", "t_precompute_fusion", "t_descriptor_update",
    "n_pairs_evaluated", "n_instances_alive",
}


def _is_fusion_owned_stat(key: str) -> bool:
    """True for stats the fusion step regenerates (per-criterion times/counts + fusion totals)."""
    return (
        key in _FUSION_OWNED_STATS
        or key.startswith("t_crit_")
        or key.startswith("t_contest_")
        or _is_contest_stat(key)
        or key.startswith("sc_")
    )


# Telemetría Tier 2 del contest (por KF). Se escriben a logger/contest/ (no logger/).
# kf_frame_id = frame_id fuente de cada muestra Tier2 (ancla para cruzar con la pose de cámara).
_CONTEST_KF_STATS = {"n_matched", "n_pre_assign", "n_used", "n_orphans", "n_births", "n_robos",
                     "kf_frame_id"}


def _is_contest_stat(key: str) -> bool:
    """True para señales del contest que viven en logger/contest/ (timing + telemetría por KF)."""
    return key.startswith("t_contest_") or key in _CONTEST_KF_STATS

class Logger:
    def __init__(self, output_path: str, pid: int | None = None, use_wandb: bool = False) -> None:
        self.output_path = Path(output_path)
        (self.output_path / "logger").mkdir(exist_ok=True, parents=True)
        (self.output_path / "logger" / "segment_vis").mkdir(exist_ok=True, parents=True)
        (self.output_path / "logger" / "contest").mkdir(exist_ok=True, parents=True)
        (self.output_path / "fusion").mkdir(exist_ok=True, parents=True)
        (self.output_path / "fusion" / "contest").mkdir(exist_ok=True, parents=True)
        stat_keys = [
            "frame_id", "t_sam", "t_obj", "n_obj", "n_matches", "t_up", "t_seg", "t_clip",
            "avg_fps", "ram", "vram", "spf", "total_time",
            "t_fusion", "t_crit_cooccurrence", "t_crit_centroid", "t_crit_aabb", "t_crit_cos_sim", "t_crit_p_dist"]

        self.stats ={key: [] for key in stat_keys}
        self.python_process = psutil.Process(pid)
        self.use_wandb = use_wandb
        self._init_fusion_log()

    @property
    def _fusion_log_path(self) -> Path:
        return self.output_path / "fusion" / "fusion_decisions.csv"

    def _init_fusion_log(self) -> None:
        with open(self._fusion_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_FUSION_LOG_FIELDS, extrasaction="ignore").writeheader()

    def log_fusion_timings(self, t_fusion: float, criterion_times: dict) -> None:
        self.stats["t_fusion"].append(t_fusion)
        for key, val in criterion_times.items():
            stat_key = f"t_crit_{key}"
            if stat_key not in self.stats:
                self.stats[stat_key] = []
            self.stats[stat_key].append(val)

    def log_contest_timings(self, contest_times: dict) -> None:
        """Log contest-step timings under their own t_contest_* namespace (distinct from fusion criteria)."""
        for key, val in contest_times.items():
            stat_key = f"t_contest_{key}"
            self.stats.setdefault(stat_key, []).append(val)

    def log_fusion_decisions(self, frame_id: int, decisions: list) -> None:
        with open(self._fusion_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FUSION_LOG_FIELDS, extrasaction="ignore")
            for d in decisions:
                writer.writerow({"frame_id": frame_id, **d})

    def log_ovo_stats(self, stats: Dict[str, Any], print_output=False) -> None:
        """
        Log CLIP extraction, fusion, and association info.

        Args:
            stats (Dict[str, Any]):
            print_output (bool = False): if True prints logged statistics
        """
        for key, item in stats.items():
            if key not in self.stats:
                self.stats[key] = []
            self.stats[key].append(item)
        if self.use_wandb:
            wandb.log({f'Semantic/{key}': value for key, value in stats.items()})
            if "n_obj" in stats.keys():
                for i in range(len(stats["n_obj"])):
                    wandb.log(
                        {
                            "Semantic/Frame": stats["frame_id"],
                            f"Semantic/n_obj_{i}":stats["n_obj"][i],
                        })
            
        if print_output:
            pprint.pprint(stats, width = 160, compact=True)

    def log_fps(self, avg_fps: float):
        self.stats["avg_fps"].append(avg_fps)
        if self.use_wandb:
            wandb.log(
                {
                    "Semantic/avg_fps": avg_fps
                }
            )
            
    def log_spf(self, spf: float):
        self.stats["spf"].append(spf)
            
    def log_memory_usage(self, frame_id: int):
        """
        Logs the memory usage, VRAM and RAM in gigabytes, for a given frame and process ID.
        If `self.use_wandb` is enabled, logs the memory usage statistics to Weights & Biases (wandb).
        Args:
            frame_id (int): Frame associtaed to statistics
        """
        torch.cuda.synchronize()
        vram_used = torch.cuda.memory_allocated("cuda") / (1000 ** 3)
        ram_used = self.python_process.memory_info().rss/(1000 ** 3)
        self.stats["vram"].append(vram_used)
        self.stats["ram"].append(ram_used)
        if self.use_wandb:
            wandb.log(
                {
                    "Semantic/Frame": frame_id,
                    "Semantic/vram": vram_used,
                    "Semantic/ram": ram_used,
                }
            )

    def log_max_memory_usage(self) -> None:
        """
        Logs the max memory usage, VRAM and RAM in gigabytes, from stored memory statistics.
        """
        torch.cuda.synchronize()
        self.stats["max_vram"] = [torch.cuda.max_memory_allocated("cuda") / (1000 ** 3)]
        ram_stats = np.asarray(self.stats["ram"])
        self.stats["max_ram"] = [ram_stats.max() if ram_stats.size > 0 else 0.0]

    def load_stats_from(self, source_path: str) -> None:
        """Load stats from a previous run's logger/ directory into self.stats.

        Values are cast to float where possible. Keys not in self.stats are
        added dynamically (same behaviour as log_ovo_stats). Missing files are
        silently skipped — not every key is written by every run. Fusion-owned
        stats are skipped: this run regenerates them, so inheriting them would
        mix the source's criteria with ours.
        """
        log_dir = Path(source_path) / "logger"
        if not log_dir.exists():
            print(f"Warning: no logger/ dir found at {source_path}, skipping stats load.")
            return
        for log_file in log_dir.glob("*.log"):
            key = log_file.stem
            if _is_fusion_owned_stat(key):
                continue
            lines = log_file.read_text().splitlines()
            values = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = int(line) if ('.' not in line) else float(line)
                    values.append([v] if key == "n_obj" else v)
                except ValueError:
                    pass
            if key not in self.stats:
                self.stats[key] = []
            self.stats[key].extend(values)

    def write_stats(self) -> None:
        """
        Writes statistics to log files. writes each statistic to a separate log file. The log files are named after the keys in the 
        dictionary, except for the key "n_obj", which is skipped. The log files are created with the ".log" extension.
        """

        for key, stat in self.stats.items():
            if key == "n_obj":
                stat_list = [str(v[0]) if isinstance(v, list) else str(v) for v in stat]
            else:
                stat_list = [str(i) for i in stat]
            # las señales del contest (timing + telemetría por KF) van a logger/contest/
            subdir = "logger/contest" if _is_contest_stat(key) else "logger"
            with open(self.output_path/subdir/f"{key}.log", "w") as f:
                f.write('\n'.join(stat_list))

    def print_final_stats(self) -> None:
        """
        Print logged statistics
        """
        stats = {f"Avg {key}": np.asarray(stat).mean().round(3) for key, stat in self.stats.items() if key not in ["frame_id", "max_vram", "max_ram"] and len(stat) > 0}
        if "max_ram" in self.stats:
            stats["Max RAM"] = round(self.stats["max_ram"][0],2)
            stats["Max vRAM"] = round(self.stats["max_vram"][0],2)
        print("Final statistics:")
        pprint.pprint(stats, compact=True)