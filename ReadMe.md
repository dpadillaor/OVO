# Official repository of Open-Vocabulary Online Semantic Mapping for SLAM

<a href="https://tberriel.github.io/">Tomas Berriel-Martins</a>,
<a href="https://oswaldm.github.io/">Martin R. Oswald</a>,
<a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en">Javier Civera</a>,

<div align="left">
    <a href='https://arxiv.org/abs/2411.15043'><img src='https://img.shields.io/badge/arXiv-2404.06836-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://tberriel.github.io/ovo/'><img src='https://img.shields.io/badge/Web-Page-green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>


## Installation
Following instruction are for an Ubuntu>=20.04 system, with installed Conda and CUDA support.

Clone repository with recursive flag:
```
    git clone git@github.com:tberriel/OVO.git --recursive
```

And set up the conda environment:
```
# conda env
conda create -n ovo python=3.11
conda activate ovo

conda install pyyaml tqdm psutil wandb plyfile numpy=2.4 matplotlib seaborn opencv=4.11 imageio scipy scikit-learn pandas -c conda-forge
# torch
pip install torch==2.5.1 torchvision==0.20.1 transformers==4.51.0 open_clip_torch==2.32.0 open3d==0.19.0 huggingface-hub==0.30.1 einops==0
8.1

#sam2
cd /<ovo_path>/thirdParty/segment-anything-2
pip install -e .

# Perception Encoder
cd /<ovo_path>/thirdParty/perception_models
pip install -e . --no-dependencies
```
By default we use Segment Anything 2, but OVO is also compatible with SAM 1. If you want, you can install it following the <a href="https://github.com/facebookresearch/segment-anything">official instructions</a>.


### Gaussian-SLAM (Optional)
If you want to run OVO using Gaussian-SLAM backbone, you will have to additionally run:
```
cd /<ovo_path>/
conda activate ovo
conda install faiss-gpu=1.8.0 cudnn -c pytorch -c conda-forge
conda install cuda-toolkit=12.1 -c nvidia/label/cuda-12.1.0
pip install git+https://github.com/VladimirYugay/simple-knn.git@c7e51a06a4cd84c25e769fee29ab391fe5d5ff8d git+https://github.com/VladimirYugay/gaussian_rasterizer.git@9c40173fcc8d9b16778a1a8040295bc2f9
```
### ORB-SLAM3 (Optional)
Clone the repository:
```
cd thirdParty/
git clone https://github.com/tberriel/ORB_SLAM3
cd ORB_SLAM3
```
Manually install ORB-SLAM3 dependencies into the conda environment:
```
conda activate ovo
# Instal conda C compilers to avoid relying on system defaults. These are generic version hardware agnostic.
conda install cxx-compiler -c conda-forge
# Install OpenGL
conda install libegl libegl-devel libgl libgl-devel libgles libgles-devel libglvnd libglvnd-devel libglx libglx-devel libopengl libopengl-devel -c conda-forge
# Install Eigen, Pangolin, OpenCV, Numpy
conda install glew eigen=3.4 pangolin-opengl=0.9.2 libopencv=4.11 numpy=2.4 boost -c conda-forge 
```
After that run the script `build.sh` to build the *ORB-SLAM3* and the python bindings.
Finally, copy the decoded ORB vocabulary from the installation folder,`ORB_SLAM3/Vocabulary/ORBvoc.txt`, to `data/working/config/slam/orbslam3/vocabulary`.


### Simulated SLAM (no install)
`slam.slam_module: simulated` runs on the dataset's GT poses with controlled drift, jumps and corrections (simulated loop closures), configured in the `slam.simulated` block of `data/working/configs/ovo.yaml`.

## Data
See <a href="./data/input/ReadMe.md">data instructions</a>.

## Run OVO
To run OVO, and compute evaluation metrics use `run_eval.py`. Running flags are:
* `--dataset_name` (required): Dataset used. Choose either `Replica`, `ScanNet`.
* `--experiment_name`: name of the folder used to store the experiment in `data/output/<dataset_name>/<experiment_name>`.
* `--run` If set, run OVO on specified scenes
* `--segment` If set, use the reconstructed scene to segment the gt point-cloud, after running OVO.
* `--eval` If set, compute the final metrics, after runing OVO and segmenting.
* `--eval_instances` If set, compute instance AP (class-aware and class-agnostic), after segmenting. Needs the instance GT described in the <a href="./data/input/ReadMe.md">data instructions</a>.
* `--dataset_info_file`: file that stores dataset info. By default is `eval_info.yaml`, to evaluate scannet200 set as `eval_info_200.yaml`.
* `--scenes`: List of scenes from given dataset to run. If `--scenes_list` is set, this flag will be ignored.
* `--scenes_list`: Path to a txt containing a scene name on each line. If set, `--scenes` is ignored. If neither `--scenes` nor `--scenes_list` are set, the scene list will be loaded from `data/working/config/<dataset_name>/<dataset_info_file>`

OVO configuration can be modified in `data/working/configs/ovo.yaml`. To speedup SAM models, you can set compile mode modifying corresponding config files in `/<conda_path>/envs/ovo/lib/python3.10/site-packages/sam2/configs/sam2.1/`. 
### Examples
* To run, segment GT, and compute metrics, on Replica Office0:
    ```
    python run_eval.py --dataset_name Replica --experiment_name ovo_mapping --run --segment --eval --scenes office0
    ```

* To run on the full ScanNet scene0011_00:

    ```
    python run_eval.py --dataset_name ScanNet --experiment_name ovo_mapping --run --segment --eval --scenes scene0011_00
    ```

### Instance AP
`--segment` writes the instance predictions to `data/output/<dataset_name>/<experiment_name>/instance_pred/`. Instance AP can then be computed again without re-running OVO:
```
python -m ovo eval instances --dataset Replica --experiment ovo_mapping [--scenes office0 room0]
```
By default every scene with predictions is evaluated. The results (AP, AP_50, AP_25, class-aware with head/common/tail splits and class-agnostic, global and per scene) are written to `data/output/<dataset_name>/<experiment_name>/instance_ap.json`; undefined metrics are `null`.

### Fusion decisions
Every map update judges which instances are the same object and fuses them. Each run writes `data/output/<dataset_name>/<experiment_name>/<scene>/fusion/fusion_decisions.csv`, one row per judged pair: `frame_id,first,second,accepted,reason,mode,cos_sim,overlap` plus one column per gate (`aabb`, `centroid`). `reason` is who decided (a gate name, `cos_sim` or `overlap`) and `mode` the accept branch (`strong`, `weak` or `both`). An empty cell is a value that was not needed to decide (an earlier step rejected the pair) or a gate that is off. Pairs the apply policy skipped have no row, and no file means no pair was judged. With the default `apply: greedy` each accepted row is one merge; with `apply: chained` the final merges follow from the accepted rows (A~B and B~C fuse A, B and C).

The time of each fusion step goes to `logger/t_fusion.log` (whole step: extract the clouds, judge, apply) and `logger/t_fusion_judge.log` (judging alone), one line per map update.

### Replay a fusion
Set `replay.checkpoints` to `last` (or `all`) to save each scene's state right before a fusion in `<scene>/fusion/pre_fusion_<frame>.ckpt`. That fusion can then be re-run alone, without SLAM or models, with other `semantic.instance_fusion` parameters (a YAML shaped like that block, which replaces it whole):
```
python -m ovo replay --dataset Replica --experiment ovo_mapping --name strict --fusion strict.yaml [--scenes office0 [--frame 1990]]
```
By default every scene with checkpoints is replayed from its last one. The output, `data/output/<dataset_name>/<experiment_name>/replays/<name>/`, is laid out like an experiment: evaluate it with `--experiment_name ovo_mapping/replays/strict --segment --eval_instances` in `run_eval.py`, or `python -m ovo eval instances --experiment ovo_mapping/replays/strict` after segmenting.

### Evaluate a fusion
A fusion step can be scored against the instance GT (Replica only: it needs the GT mesh `data/input/Datasets/Replica/<scene>_mesh.ply`). It reads the step's pre-fusion checkpoint, so the run needs `replay.checkpoints: last` (or `all`):
```
python -m ovo eval fusion --dataset Replica --experiment ovo_mapping [--scenes office0 [--frame 1990]]
```
By default the last checkpointed fusion of every scene with fusion decisions is evaluated; `--frame` picks another one (one scene only). A replay is evaluated the same way, with `--experiment ovo_mapping/replays/strict`. The pre-fusion map is projected on the GT mesh, and each instance takes the GT instance that covers most of it (any labelled one, walls included). A pair is the same object when both take the same GT instance: merged and same is `TP`, merged and different `FP`, kept apart and same `FN`, kept apart and different `TN`. The output goes to `<scene>/fusion/eval_<frame>/`:
- `decisions_eval.csv`: the step's rows of `fusion_decisions.csv` plus `gt_first, gt_second, same_object, verdict`. These cells are empty when an instance does not reach the mesh.
- `summary.json`: instances before and after, the verdict counts with precision, recall and F1, and the same counts at each step of the pipeline (gates, `cos_sim`, `overlap`, the last one split by accept mode). It also has the class-agnostic AP before and after the fusion (the AP of `eval instances`) and the `instance_fusion` block that was used.
- `instance_stats.csv`: one row per GT instance the AP counts: best IoU and coverage before and after, `change` (`improved`, `worsened`, `unchanged`), `match_50` (`KEPT`, `LOST`, `GAINED`, `UNMATCHED` at IoU 0.5), the matching instance before and after, and the extra fragments it had before.

Every input is checked before anything is written: a missing one, or a decisions log that does not belong to the checkpoint, is an error.

## Changelog:
- 13, January, 2026 - Added implementation using [TextRegion](https://github.com/avaxiao/TextRegion) with the [Perception Encoder](https://github.com/facebookresearch/perception_models) as descriptor. Performance in Replica goes up to 35.7% IoU and 57% Acc (from from 27.0% IoU and 39.1% Acc reported in the paper). 
- 11, January, 2026 - Update conda environment to work with Python 3.11 and Numpy 2.4.
- 7, October, 2025  - Switched from ORB-SLAM2 to ORB-SLAM3 to minimize segmentation fault errors. 
- 10, June, 2025 - Improved integration with ORB-SLAM2 and added loop closure support.
## Citation
If you found our work useful, please cite us.
```
    @article{martins2024ovo,
    title={Open-Vocabulary Online Semantic Mapping for SLAM},
    author={Martins, Tomas Berriel and Oswald, Martin R. and Civera, Javier},
    journal={IEEE Robotics and Automation Letters}, 
    year={2025},
    }
```