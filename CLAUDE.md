# OVO (Open-Vocabulary Online Semantic Mapping)

OVO is a research project for 3D instance-aware semantic mapping. It integrates SLAM backbones with semantic recognition using open-vocabulary models to build 3D maps where objects are recognized as distinct instances. Original work was using CLIP for the semantic part. We'll be testing new things. The current focus is on improving semantic fusion during map optimizations and loop closures.

## Diagrams
Write diagrams as PlantUML (`.puml`) source. Do NOT render them (no inline SVG/widget) — the user renders `.puml` from VSCode.

## TFM (Master's Thesis) — `tfm/`

LaTeX thesis for the EINA / Universidad de Zaragoza master's programme. Layout and title page are ported from the official EINA template.

**Language.** Prose is written in **Spanish**. Technical terms, chapter titles, and figure labels stay in **English** (*loop closure*, *open-vocabulary*, *contest*…) — do not translate them. `babel` is loaded as `[english,spanish]`, so Spanish is the default and English is reachable via `\foreignlanguage{english}{...}`.

**Toolchain.** `pdflatex` + `latexmk` + `biber`/`biblatex`. Do not switch engines: `inputenc` is loaded, so `xelatex`/`lualatex` will break. `tfm/.latexmkrc` pins `$out_dir = 'build'` and is authoritative — every intermediate file lands in `tfm/build/` (gitignored), never next to the sources.

Build and check for regressions with:
```bash
cd tfm && latexmk -interaction=nonstopmode main.tex   # exit=0 and no '^!' lines in build/main.log
```

**Structure.**
```text
tfm/
├── main.tex         # preamble + document skeleton; the only place \usepackage lives
├── refs.bib         # biblatex entries; cite with \cite{key}
├── frontmatter/     # titlepage, acknowledgements, abstract
├── chapters/        # 01..06 + A_appendix, pulled in via \include
├── images/          # raw assets (Unizar logo, Rerun screenshots) → PNG, or PDF if vector
└── figures/         # generated figures (matplotlib plots, exported diagrams) → PDF
```

**Figures.** Vector content (plots, diagrams, anything with text) as **PDF**; pixel content (screenshots, RGB-D frames) as **PNG** at ≥300 dpi. Both dirs are in `\graphicspath`, so reference them bare: `\includegraphics{foo}`.

**Conventions.**
*   `\usepackage` only in `main.tex`. Chapter files hold content, never preamble.
*   Load `xcolor` early and `hyperref` last — reordering triggers an option clash.
*   Never add `\nocite{*}`: it dumps the whole `.bib` into the bibliography, cited or not.
*   Keep `tfm/` self-contained — no absolute paths, no symlinks, no `../` outside it.

## Project Structure
```text
/
├── ovo/                    # Core OVO system
│   ├── entities/           # Instance management, generators, orchestrator, semantic classes
│   │   ├── ovomapping.py   # Main orchestrator (SLAM + Semantics)
│   │   ├── ovo.py          # Semantic workflow & instance handling
│   │   ├── fusion.py       # Fusion and merging logic
│   │   └── datasets.py     # Semantic class definitions and label mapping
│   ├── slam/               # SLAM backbone wrappers (ORB-SLAM, GT, etc.)
│   └── utils/              # Geometry and instance utility functions
├── scripts/                # Experiment orchestration and preprocessing
├── .agents_mapper/         # Task tracking and detailed documentation
├── data/                   # Configs, input datasets, and experiment outputs
├── thirdParty/             # External modules (ORB-SLAM3, SAM 2, etc.)
├── tests/                  # Unit and integration tests
└── run_eval.py             # Main entry point for experiments
```

## Architecture & Workflow
*   **`ovomapping.py`**: The central orchestrator. It manages the data flow between the SLAM backbone and the semantic modules, handles initial configuration, and triggers semantic updates.
*   **`ovo.py`**: Manages the semantic workflow, including instance creation, feature extraction, and the fusion/merging process.
*   **SLAM Integration**: All SLAM-specific logic is contained within `ovo/slam/`.
*   **Semantic Classes**: Management of dataset-specific semantic classes, label mappings, and category definitions is handled within `ovo/entities/` (specifically in `datasets.py`).
*   **Semantic Fusion**: The core logic for merging instances and handling descriptors (CLIP/DINO) resides in `ovo/entities/fusion.py` and `ovo/utils/instance_utils.py`.
/Con
## Running Experiments (conda env)

All experiment commands (`run_eval.py`, `scripts/run_experiments_batch.py`) must run in the **`ovo2`** conda env — it has `wandb`, torch+cuda, and the rest. The base env lacks `wandb` and fails at import. Launch with `conda run -n ovo2 python ...` (note: `conda run` buffers stdout, so a live log may look empty while the run is actually progressing — check the process/GPU, not just the log).

## Worktree Setup (Required After `git worktree add`)

New worktrees ship with empty submodule stubs and no heavy data. Before running experiments in a worktree, symlink `thirdParty/` and `data/` subdirs to the primary repo at `/home/padidavid/repos/OVO/`. This reuses built submodules, model weights, datasets, and centralizes experiment outputs.

```bash
PRIMARY=/home/padidavid/repos/OVO
WT=<absolute-path-to-new-worktree>

# thirdParty: replace empty submodule stubs + add ORB_SLAM3
for sub in perception_models sam3 segment-anything-2; do
  [ -d "$WT/thirdParty/$sub" ] && [ ! -L "$WT/thirdParty/$sub" ] && rmdir "$WT/thirdParty/$sub"
  ln -sfn "$PRIMARY/thirdParty/$sub" "$WT/thirdParty/$sub"
done
ln -sfn "$PRIMARY/thirdParty/ORB_SLAM3" "$WT/thirdParty/ORB_SLAM3"

# data: heavy + write paths. If targets pre-exist as real dirs in the worktree,
# `rm -rf` them first (ln -sfn nests inside otherwise).
mkdir -p "$WT/data/input" "$WT/data/working"
ln -sfn "$PRIMARY/data/baselines"              "$WT/data/baselines"
ln -sfn "$PRIMARY/data/output"                 "$WT/data/output"
ln -sfn "$PRIMARY/data/input/Datasets"         "$WT/data/input/Datasets"
ln -sfn "$PRIMARY/data/input/sam_ckpts"        "$WT/data/input/sam_ckpts"
ln -sfn "$PRIMARY/data/working/config_preview" "$WT/data/working/config_preview"
```

Notes:
*   Centralized `data/output/` → every branch writes here; folder names (`{DATE}_{SLAM_CONFIG}_{FUSION}_{LABEL}`) must encode branch in `LABEL` to avoid clobbering.
*   Removing a worktree (`git worktree remove` or `rm -rf <wt>`) does **not** follow symlinks → primary stays intact.
*   Tracked subdirs (`data/input/{ReadMe.md,replica_semantic_gt,weights_predictor}`, `data/working/configs`) stay local — per-subdir symlinks preserve them.
