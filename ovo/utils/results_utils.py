"""
results_utils.py — Post-hoc analysis of OVO experiment results.

Design principles:
  - Strict separation: parsing functions never touch matplotlib;
    plotting functions never read disk.
  - All plot functions return matplotlib.figure.Figure and never call
    plt.show() or save to disk. The caller decides.
  - Filtering is done in the data layer; plots receive already-filtered DataFrames.
  - No global state.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Section 1 — Parsing & data loading
# ---------------------------------------------------------------------------

# Folder name convention: {DATE}_{SCENE}_{SLAM_CONFIG}_{FUSION}_{LABEL}
# Parts are underscore-separated; SLAM_CONFIG itself uses only hyphens.
# Minimum 5 underscore-split parts required; LABEL may span multiple parts.
_DATE_RE        = re.compile(r'^\d{8}$')
_TRANS_NOISE_RE = re.compile(r'T(\d+)p(\d+)')
_ROT_NOISE_RE   = re.compile(r'R(\d+)p(\d+)')


def _noise_str_to_float(integer_part: str, decimal_part: str) -> float:
    """Convert '0', '001' → 0.001; '0', '05' → 0.05."""
    return float(f"{integer_part}.{decimal_part}")


def parse_experiment_name(folder_name: str) -> dict | None:
    """Parse an experiment folder name into its components.

    Convention: {DATE}_{SLAM_CONFIG}_{FUSION}_{LABEL}
    Parts are split by underscores; SLAM_CONFIG itself uses only hyphens.
    LABEL is the join of all remaining parts (may contain underscores).
    Dataset is NOT part of the folder name — it is inferred from the parent
    directory by the loader functions.

    Returns a dict with keys:
        Date, SLAM_Config, Fusion, Label, Method, Trans_Noise, Rot_Noise
    or None if the name does not match the convention.
    """
    parts = folder_name.split('_')
    if len(parts) < 4:
        return None

    date, slam_cfg, fusion = parts[0], parts[1], parts[2]
    label = '_'.join(parts[3:])

    # Basic validation
    if not _DATE_RE.match(date):
        return None

    # Extract noise values when present; zero-noise tokens (GT, ORBSLAM3) are valid.
    tm = _TRANS_NOISE_RE.search(slam_cfg)
    rm = _ROT_NOISE_RE.search(slam_cfg)
    if tm and rm:
        trans_noise = _noise_str_to_float(tm.group(1), tm.group(2))
        rot_noise   = _noise_str_to_float(rm.group(1), rm.group(2))
    elif not tm and not rm:
        trans_noise = 0.0
        rot_noise   = 0.0
    else:
        # Only one marker found — malformed name.
        return None

    return {
        'Date':        date,
        'SLAM_Config': slam_cfg,
        'Fusion':      fusion,
        'Label':       label,
        'Method':      f"{fusion}_{label}",
        'Trans_Noise': trans_noise,
        'Rot_Noise':   rot_noise,
    }


def parse_statistics_file(file_path: Path) -> dict | None:
    """Parse a statistics.txt file.

    Expected format (first line is header, then one row per class):
        label, acc, iou,

    Returns:
        {"mIoU": float, "mAcc": float, "per_class": pd.DataFrame}
    or None on failure.
    """
    try:
        rows = []
        with open(file_path) as f:
            lines = f.readlines()

        for line in lines[1:]:  # skip header
            line = line.strip().rstrip(',')
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                continue
            class_name = parts[0]
            try:
                acc = float(parts[1])
                iou = float(parts[2])
            except ValueError:
                acc = float('nan')
                iou = float('nan')
            rows.append({'Class': class_name, 'Acc': acc, 'IoU': iou})

        if not rows:
            return None

        df = pd.DataFrame(rows)
        miou = float(np.nanmean(df['IoU'].values))
        macc = float(np.nanmean(df['Acc'].values))

        thirds = len(df) // 3
        head_iou   = float(np.nanmean(df['IoU'].values[0:thirds]))
        head_acc   = float(np.nanmean(df['Acc'].values[0:thirds]))
        common_iou = float(np.nanmean(df['IoU'].values[thirds:2*thirds]))
        common_acc = float(np.nanmean(df['Acc'].values[thirds:2*thirds]))
        tail_iou   = float(np.nanmean(df['IoU'].values[2*thirds:3*thirds]))
        tail_acc   = float(np.nanmean(df['Acc'].values[2*thirds:3*thirds]))

        return {
            'mIoU': miou, 'mAcc': macc,
            'Head_mIoU': head_iou, 'Head_mAcc': head_acc,
            'Common_mIoU': common_iou, 'Common_mAcc': common_acc,
            'Tail_mIoU': tail_iou, 'Tail_mAcc': tail_acc,
            'per_class': df,
        }
    except Exception:
        return None


def parse_instance_counts(folder_path: Path, scene_name: str) -> int:
    """Count predicted instance lines for a scene.

    Looks for `{folder_path}/instance_pred/{scene_name}.txt`.
    Falls back to the first .txt found in instance_pred/.
    Returns 0 if nothing is found.
    """
    instance_dir = folder_path / 'instance_pred'
    if not instance_dir.is_dir():
        return 0

    candidate = instance_dir / f'{scene_name}.txt'
    if not candidate.is_file():
        txts = list(instance_dir.glob('*.txt'))
        if not txts:
            return 0
        candidate = txts[0]

    try:
        with open(candidate) as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def load_experiments(
    output_dir: Path,
    date_filter: str | list[str] | None = None,
    dataset_filter: str | list[str] | None = None,
    method_filter: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan output_dir and build experiment DataFrames.

    output_dir is expected to contain one subdirectory per dataset (e.g.
    Replica/, ScanNet/), each of which contains one subdirectory per
    experiment. The dataset name is inferred from the dataset subdirectory.

    Experiment folders whose names do not match the naming convention are
    silently skipped.

    Parameters
    ----------
    output_dir:
        Root directory containing dataset subdirectories.
    date_filter:
        Keep only experiments whose Date matches (exact string or list).
    dataset_filter:
        Keep only experiments whose Dataset matches (exact string or list).
    method_filter:
        Keep only experiments whose Method is in this list.

    Returns
    -------
    (df_experiments, df_class_iou)
        df_experiments  — one row per experiment.
            Columns: Date, Dataset, SLAM_Config, Fusion, Label, Method,
                     Trans_Noise, Rot_Noise, mIoU, mAcc, Num_Instances,
                     Experiment_ID
        df_class_iou  — one row per (experiment × class).
            Columns: all df_experiments columns + Class, IoU, Acc
    """
    output_dir = Path(output_dir)

    if isinstance(date_filter, str):
        date_filter = [date_filter]
    if isinstance(dataset_filter, str):
        dataset_filter = [dataset_filter]

    exp_rows: list[dict]   = []
    class_rows: list[dict] = []

    for dataset_dir in sorted(output_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        if dataset_filter and dataset not in dataset_filter:
            continue

        for folder in sorted(dataset_dir.iterdir()):
            if not folder.is_dir():
                continue

            meta = parse_experiment_name(folder.name)
            if meta is None:
                continue

            # Apply filters early to avoid unnecessary I/O
            if date_filter and meta['Date'] not in date_filter:
                continue
            if method_filter and meta['Method'] not in method_filter:
                continue

            # Find statistics file — canonical location is replica/statistics.txt
            stats_file = folder / 'replica' / 'statistics.txt'
            if not stats_file.is_file():
                candidates = list(folder.rglob('statistics.txt'))
                stats_file = candidates[0] if candidates else None

            stats = parse_statistics_file(stats_file) if stats_file else None
            num_instances = parse_instance_counts(folder, folder.name)

            exp_row: dict = {
                **meta,
                'Dataset':       dataset,
                'mIoU':          stats['mIoU']        if stats else float('nan'),
                'mAcc':          stats['mAcc']        if stats else float('nan'),
                'Head_mIoU':     stats['Head_mIoU']   if stats else float('nan'),
                'Head_mAcc':     stats['Head_mAcc']   if stats else float('nan'),
                'Common_mIoU':   stats['Common_mIoU'] if stats else float('nan'),
                'Common_mAcc':   stats['Common_mAcc'] if stats else float('nan'),
                'Tail_mIoU':     stats['Tail_mIoU']   if stats else float('nan'),
                'Tail_mAcc':     stats['Tail_mAcc']   if stats else float('nan'),
                'Num_Instances': num_instances,
                'Experiment_ID': folder.name,
            }
            exp_rows.append(exp_row)

            if stats:
                for _, cls_row in stats['per_class'].iterrows():
                    class_rows.append({**exp_row, **cls_row.to_dict()})

    _exp_cols = [
        'Date', 'Dataset', 'SLAM_Config', 'Fusion', 'Label', 'Method',
        'Trans_Noise', 'Rot_Noise', 'mIoU', 'mAcc',
        'Head_mIoU', 'Head_mAcc', 'Common_mIoU', 'Common_mAcc', 'Tail_mIoU', 'Tail_mAcc',
        'Num_Instances', 'Experiment_ID',
    ]
    _class_cols = _exp_cols + ['Class', 'IoU', 'Acc']

    df_exp   = pd.DataFrame(exp_rows)   if exp_rows   else pd.DataFrame(columns=_exp_cols)
    df_class = pd.DataFrame(class_rows) if class_rows else pd.DataFrame(columns=_class_cols)

    return df_exp, df_class


# ---------------------------------------------------------------------------
# Section 2 — Filtering helpers (intended for GUI dropdowns)
# ---------------------------------------------------------------------------

def get_available_dates(df: pd.DataFrame) -> list[str]:
    """Return sorted unique dates present in the DataFrame."""
    return sorted(df['Date'].dropna().unique().tolist())


def get_available_datasets(df: pd.DataFrame) -> list[str]:
    """Return sorted unique dataset names present in the DataFrame."""
    return sorted(df['Dataset'].dropna().unique().tolist())


def get_available_scenes(df: pd.DataFrame) -> list[str]:
    """Return sorted unique scene names present in a per-scene DataFrame."""
    return sorted(df['Scene'].dropna().unique().tolist())


def get_available_methods(df: pd.DataFrame) -> list[str]:
    """Return sorted unique method names present in the DataFrame."""
    return sorted(df['Method'].dropna().unique().tolist())


def get_available_noise_levels(df: pd.DataFrame) -> list[float]:
    """Return sorted unique Trans_Noise values present in the DataFrame."""
    return sorted(df['Trans_Noise'].dropna().unique().tolist())


def filter_experiments(
    df: pd.DataFrame,
    dates: list[str] | None = None,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    noise_levels: list[float] | None = None,
    experiment_ids: list[str] | None = None,
    exclude_experiment_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Filter an experiment DataFrame with AND logic across all provided criteria.

    Pass None for any parameter to skip that filter.
    experiment_ids: keep only these IDs (include filter).
    exclude_experiment_ids: drop these IDs (exclude filter).
    """
    mask = pd.Series(True, index=df.index)
    if dates is not None:
        mask &= df['Date'].isin(dates)
    if datasets is not None:
        mask &= df['Dataset'].isin(datasets)
    if methods is not None:
        mask &= df['Method'].isin(methods)
    if noise_levels is not None:
        mask &= df['Trans_Noise'].isin(noise_levels)
    if experiment_ids is not None:
        mask &= df['Experiment_ID'].isin(experiment_ids)
    if exclude_experiment_ids is not None:
        mask &= ~df['Experiment_ID'].isin(exclude_experiment_ids)
    return df[mask].copy()


def load_scene_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan output_dir and build per-scene DataFrames.

    Mirrors the two-level structure of load_experiments(): output_dir contains
    dataset subdirs, each of which contains experiment folders.  For each
    experiment, detects per-scene statistics files
    (replica/statistics_{scene}.txt) and builds DataFrames with one row per
    (experiment × scene).

    Returns
    -------
    (df_scene, df_scene_class)
        df_scene — one row per (experiment × scene).
            Columns: Date, Dataset, SLAM_Config, Fusion, Label, Method,
                     Trans_Noise, Rot_Noise, Scene, mIoU, mAcc,
                     Num_Instances, Experiment_ID
        df_scene_class — one row per (experiment × scene × class).
            Columns: all df_scene columns + Class, IoU, Acc
    """
    output_dir = Path(output_dir)

    scene_rows: list[dict] = []
    scene_class_rows: list[dict] = []

    for dataset_dir in sorted(output_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        for folder in sorted(dataset_dir.iterdir()):
            if not folder.is_dir():
                continue

            meta = parse_experiment_name(folder.name)
            if meta is None:
                continue

            replica_dir = folder / 'replica'
            search_dir = replica_dir if replica_dir.is_dir() else folder

            for stats_file in sorted(search_dir.glob('statistics_*.txt')):
                scene_name = stats_file.stem[len('statistics_'):]

                stats = parse_statistics_file(stats_file)
                num_instances = parse_instance_counts(folder, scene_name)

                scene_row: dict = {
                    **meta,
                    'Dataset':       dataset,
                    'Scene':         scene_name,
                    'mIoU':          stats['mIoU']        if stats else float('nan'),
                    'mAcc':          stats['mAcc']        if stats else float('nan'),
                    'Head_mIoU':     stats['Head_mIoU']   if stats else float('nan'),
                    'Head_mAcc':     stats['Head_mAcc']   if stats else float('nan'),
                    'Common_mIoU':   stats['Common_mIoU'] if stats else float('nan'),
                    'Common_mAcc':   stats['Common_mAcc'] if stats else float('nan'),
                    'Tail_mIoU':     stats['Tail_mIoU']   if stats else float('nan'),
                    'Tail_mAcc':     stats['Tail_mAcc']   if stats else float('nan'),
                    'Num_Instances': num_instances,
                    'Experiment_ID': folder.name,
                }
                scene_rows.append(scene_row)

                if stats:
                    for _, cls_row in stats['per_class'].iterrows():
                        scene_class_rows.append({**scene_row, **cls_row.to_dict()})

    _scene_cols = [
        'Date', 'Dataset', 'SLAM_Config', 'Fusion', 'Label', 'Method',
        'Trans_Noise', 'Rot_Noise', 'Scene', 'mIoU', 'mAcc',
        'Head_mIoU', 'Head_mAcc', 'Common_mIoU', 'Common_mAcc', 'Tail_mIoU', 'Tail_mAcc',
        'Num_Instances', 'Experiment_ID',
    ]
    _scene_class_cols = _scene_cols + ['Class', 'IoU', 'Acc']

    df_scene = (
        pd.DataFrame(scene_rows) if scene_rows
        else pd.DataFrame(columns=_scene_cols)
    )
    df_scene_class = (
        pd.DataFrame(scene_class_rows) if scene_class_rows
        else pd.DataFrame(columns=_scene_class_cols)
    )
    return df_scene, df_scene_class


def filter_scene_results(
    df: pd.DataFrame,
    dates: list[str] | None = None,
    datasets: list[str] | None = None,
    scenes: list[str] | None = None,
    methods: list[str] | None = None,
    noise_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Filter a per-scene DataFrame with AND logic across all provided criteria."""
    mask = pd.Series(True, index=df.index)
    if dates is not None:
        mask &= df['Date'].isin(dates)
    if datasets is not None:
        mask &= df['Dataset'].isin(datasets)
    if scenes is not None:
        mask &= df['Scene'].isin(scenes)
    if methods is not None:
        mask &= df['Method'].isin(methods)
    if noise_levels is not None:
        mask &= df['Trans_Noise'].isin(noise_levels)
    return df[mask].copy()


def load_reference_results(csv_path: Path) -> pd.DataFrame:
    """Load static paper reference results from a CSV file.

    Returns an empty DataFrame if the file doesn't exist.
    """
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df


def get_available_scenes_from_data(df_scene: pd.DataFrame) -> list[str]:
    """Return sorted unique scene names from a per-scene DataFrame."""
    return sorted(df_scene['Scene'].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Section 3 — Plots (all return Figure, never call plt.show())
# ---------------------------------------------------------------------------

def plot_bar_metrics(
    df: pd.DataFrame,
    metric: str = 'mIoU',
    x: str = 'Trans_Noise',
    hue: str = 'Method',
    figsize: tuple = (10, 6),
) -> Figure:
    """Bar plot of a scalar metric grouped by x with hue.

    Parameters
    ----------
    df:      Experiment DataFrame (from load_experiments).
    metric:  Column to plot ('mIoU' or 'mAcc').
    x:       Column to use on the x-axis.
    hue:     Column to use for colour grouping.
    figsize: Figure dimensions in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x=x, y=metric, hue=hue, ax=ax)
    ax.set_title(f'{metric} by {x}')
    ax.set_xlabel(x)
    ax.set_ylabel(metric)
    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig


def plot_line_metrics(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    x: str = 'Trans_Noise',
    hue: str = 'Method',
    figsize: tuple = (12, 6),
) -> Figure:
    """Line plot of one or more metrics vs a continuous variable.

    Parameters
    ----------
    df:      Experiment DataFrame (from load_experiments).
    metrics: Columns to plot (default: ['mIoU', 'mAcc']).
    x:       Column to use on the x-axis.
    hue:     Column to use for colour grouping.
    figsize: Figure dimensions in inches.
    """
    if metrics is None:
        metrics = ['mIoU', 'mAcc']

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    for i, metric in enumerate(metrics):
        ax = axes[0][i]
        sns.lineplot(data=df, x=x, y=metric, hue=hue, ax=ax, marker='o')
        ax.set_title(f'{metric} vs {x}')
        ax.set_xlabel(x)
        ax.set_ylabel(metric)
        ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig


def plot_class_heatmap(
    df_class: pd.DataFrame,
    value: str = 'IoU',
    figsize: tuple = (16, 10),
) -> Figure:
    """Heatmap of a value (IoU or Acc) across class × experiment.

    Parameters
    ----------
    df_class: Per-class DataFrame (from load_experiments).
    value:    Column to display ('IoU' or 'Acc').
    figsize:  Figure dimensions in inches.
    """
    pivot = df_class.pivot_table(
        index='Class', columns='Experiment_ID', values=value, aggfunc='mean',
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
        linewidths=0.3, linecolor='grey', annot=False,
        cbar_kws={'label': value},
    )
    ax.set_title(f'Per-class {value} across experiments')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Class')
    fig.tight_layout()
    return fig


def plot_instance_counts(
    df: pd.DataFrame,
    x: str = 'Trans_Noise',
    hue: str = 'Method',
    figsize: tuple = (10, 6),
) -> Figure:
    """Bar plot of Num_Instances with numeric labels on each bar.

    Parameters
    ----------
    df:      Experiment DataFrame (from load_experiments).
    x:       Column to use on the x-axis.
    hue:     Column to use for colour grouping.
    figsize: Figure dimensions in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x=x, y='Num_Instances', hue=hue, ax=ax)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f'{int(h)}',
                ha='center', va='bottom', fontsize=8,
            )
    ax.set_title(f'Instance count by {x}')
    ax.set_xlabel(x)
    ax.set_ylabel('Num_Instances')
    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    experiment_id: str,
    output_dir: Path,
    figsize: tuple = (8, 6),
) -> Figure | None:
    """Load and display the confmat.png for a given experiment.

    Looks for `{output_dir}/{experiment_id}/replica/confmat.png`.

    Returns None if the file does not exist.
    """
    img_path = Path(output_dir) / experiment_id / 'replica' / 'confmat.png'
    if not img_path.is_file():
        return None

    img = mpimg.imread(str(img_path))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Confusion matrix — {experiment_id}')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Section 4 — Export helper
# ---------------------------------------------------------------------------

def save_figure(fig: Figure, output_path: Path, dpi: int = 150) -> None:
    """Save a figure to disk at the specified path and DPI."""
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
