# OVOSemMap direct config accesses

## Stored as instance attributes
| Key path | file:line | Attr name |
|---|---|---|
| `config` (entire dict) | ovomapping.py:124 | `self.config` |
| `config.get("device", "cuda")` | ovomapping.py:125 | `self.device` |
| `config["dataset_name"]` | ovomapping.py:126 | `self.dataset_name` |

## Passed to child constructors
| Key path | file:line | Passed to |
|---|---|---|
| `config["use_wandb"]` | ovomapping.py:132 | `Logger.__init__()` |
| `{**config["data"], **config["cam"]}` | ovomapping.py:133 | `get_dataset()` |
| `config["semantic"]` | ovomapping.py:141 | `OVO.__init__()` |
| `config["data"]["scene_name"]` | ovomapping.py:141 | `OVO.__init__()` |
| full `config` | ovomapping.py (get_slam_backbone call) | `get_slam_backbone()` → SLAM backends |

## Used in conditions/logic
| Key path | file:line | Purpose |
|---|---|---|
| `config["slam"].get("slam_module", "vanilla")` | ovomapping.py:89 | choose SLAM backend |
| `config["semantic"]["sam"].get("precomputed", False)` | ovomapping.py:145 | check precomputed SAM masks |
| `config["semantic"]["sam"].get("precompute", False)` | ovomapping.py:145 | check if should precompute |
| `self.config.get("restore_map", False)` | ovomapping.py:152 | decide whether to restore map |
| `config["slam"].get("slam_module", "vanilla")` | ovomapping.py:153 | assert for restore compatibility |
| `self.config["slam"].get("save_estimated_cam", False)` | ovomapping.py:202 | check if saving camera poses |
| `self.config["data"]["scene_name"]` | ovomapping.py:259,271 | visualizer process args |

## Mutated before passthrough
| Key path | file:line | Mutation |
|---|---|---|
| `config["output_path"]` | ovomapping.py:121 | injected: `config["output_path"] = str(self.output_path)` |
| `config["semantic"]["debug_info"]` | ovomapping.py:137 | injected from `self.config.get("debug_info", False)` |
| `config["semantic"]["fusion_method"]` | ovomapping.py:138 | defaulted to `"CLIP"` if absent |

## Already delegated (not counted above)
- `config["vis"]` → `VisConfig.from_config()`
- `config["mapping"]["map_every"]` → `SchedulingConfig.from_config()`
- `config["semantic"]["segment_every"]` → `SchedulingConfig.from_config()`
- `config["tracking"]["track_every"]` → `SchedulingConfig.from_config()`
