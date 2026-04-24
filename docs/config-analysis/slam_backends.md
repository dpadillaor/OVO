# config (full) usage — SLAM backends

## vanilla_mapper.py
| Key path | file:line | Usage |
|---|---|---|
| `config.get("device", "cuda")` | vanilla_mapper.py:12 | stored as `self.device` |
| `config["mapping"].get("max_frame_points", 1e5)` | vanilla_mapper.py:13 | stored as `self.max_frame_points` |
| `config["mapping"].get("k_pooling", 3)` | vanilla_mapper.py:26 | configures `MaxPool2d` |
| `config["mapping"].get("downscale_res", 2)` | vanilla_mapper.py:32 | creates downscale lambda |

## gaussian_slam.py (WrapperGaussianSLAM)
| Key path | file:line | Usage |
|---|---|---|
| `config.get("device", "cuda")` | gaussian_slam.py:15 | stored as `self.device` |
| `config["mapping"]["map_every"]` | gaussian_slam.py:29 | mapping frequency, used in list slicing |
| `config["output_path"]` | gaussian_slam.py:32 | passed to `os.makedirs()` |
| `config["output_path"]` | gaussian_slam.py:34 | passed to `Logger()` constructor |
| `config["mapping"]` | gaussian_slam.py:35 | full sub-dict → `Mapper()` constructor |
| `config["tracking"]` | gaussian_slam.py:36 | full sub-dict → `Tracker()` constructor |
| `config["tracking"]`, `config["mapping"]` | gaussian_slam.py:43,45 | debug printing |

## orbslam2.py (WrapperORBSLAM2)
| Key path | file:line | Usage |
|---|---|---|
| full `config` | orbslam2.py:20 | passed to parent `VanillaMapper.__init__` |
| `config["slam"].get("close_loops", True)` | orbslam2.py:22 | stored as `self.close_loops` |
| `config["slam"]["config_path"]` | orbslam2.py:28 | path to SLAM configs |
| `config["dataset_name"]` | orbslam2.py:37,38,40 | config file lookup |
| `config["data"]["scene_name"]` | orbslam2.py:37 | config file lookup |
| `config["slam"].get("use_viewer", False)` | orbslam2.py:42 | passed to `orbslam.System()` |

## groundtruth_slam.py (GroundTruthSLAM)
| Key path | file:line | Usage |
|---|---|---|
| full `config` | groundtruth_slam.py:21 | passed to parent `VanillaMapper.__init__` |
| `self.config["dataset_name"]` | groundtruth_slam.py:23 | trajectory file path |
| `self.config["data"]["scene_name"]` | groundtruth_slam.py:24 | trajectory file path |
| `self.config.get("kf_dist_thresh", 0.1)` | groundtruth_slam.py:41 | stored as attr |
| `self.config.get("kf_rot_thresh", 5.0)` | groundtruth_slam.py:42 | stored as attr |
| `self.config.get("lc_dist_thresh", 0.2)` | groundtruth_slam.py:43 | stored as attr |
| `self.config.get("lc_rot_thresh", 10.0)` | groundtruth_slam.py:44 | stored as attr |
| `self.config.get("slam", {}).get("close_loops", True)` | groundtruth_slam.py:46 | stored as attr |
| `self.config.get("mapping", {}).get("map_every", 10)` | groundtruth_slam.py:49 | stored as attr |
| `self.config["noise"]` | groundtruth_slam.py:65-97 | extensive access for noise/jump params |

## Further passthroughs
| Config key | Destination | File:line |
|---|---|---|
| `config["mapping"]` | `Mapper()` | gaussian_slam.py:35 |
| `config["tracking"]` | `Tracker()` | gaussian_slam.py:36 |
| `config["output_path"]` | `Logger()` | gaussian_slam.py:34 |
| full `config` | `VanillaMapper` parent | orbslam2.py:20, groundtruth_slam.py:21 |
