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

# Folder name convention: {DATE}_{SLAM_CONFIG}_{FUSION}_{LABEL}
# Parts are underscore-separated; SLAM_CONFIG itself uses only hyphens.
# Minimum 4 underscore-split parts required; LABEL may span multiple parts.
_DATE_RE        = re.compile(r'^\d{8}$')
_TRANS_NOISE_RE = re.compile(r'T(\d+)p(\d+)')
_ROT_NOISE_RE   = re.compile(r'R(\d+)p(\d+)')
_JUMP_RE        = re.compile(r'J(\d+)')


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

    jm = _JUMP_RE.search(slam_cfg)
    jump_count = int(jm.group(1)) if jm else 0

    return {
        'Date':        date,
        'SLAM_Config': slam_cfg,
        'Fusion':      fusion,
        'Label':       label,
        'Method':      f"{slam_cfg}_{fusion}_{label}",
        'Trans_Noise': trans_noise,
        'Rot_Noise':   rot_noise,
        'Jump_Count':  jump_count,
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


def parse_instance_ap_file(file_path: Path) -> dict | None:
    """Parse an instance_ap.txt file.

    Expected format:
        metric, value
        AP, 0.012
        AP_50, 0.021
        ...

    Returns dict with keys AP, AP_50, AP_25, AP_agnostic, AP_agnostic_50,
    AP_agnostic_25, or None on failure.
    """
    try:
        result = {}
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('metric'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 2:
                    continue
                try:
                    result[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return result if result else None
    except Exception:
        return None


_FUSION_DECISION_COLS = ['frame_id', 'result', 'i1', 'i2', 'reason',
                          'centroid_dist', 'cos_sim', 'p_dist', 'shared_kfs']
_FUSION_DECISION_ZERO = {
    'Fusion_Total': 0, 'Fusion_Accepted': 0, 'Fusion_Accept_Rate': float('nan'),
    'Fusion_Reject_Centroid': 0, 'Fusion_Reject_AABB': 0, 'Fusion_Reject_CosSim': 0,
    'Fusion_Reject_Overlap': 0, 'Fusion_Reject_Cooccurrence': 0,
}


def parse_fusion_decisions(file_path: Path) -> dict:
    """Aggregate stats from a fusion_decisions.csv for one scene.

    Returns dict with Fusion_Total, Fusion_Accepted, Fusion_Accept_Rate,
    Fusion_Reject_Centroid, Fusion_Reject_AABB, Fusion_Reject_CosSim, Fusion_Reject_Overlap,
    Fusion_Reject_Cooccurrence.
    Returns zeros (Accept_Rate=nan) if file missing or empty.
    """
    if not file_path.is_file():
        return _FUSION_DECISION_ZERO.copy()
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return _FUSION_DECISION_ZERO.copy()
        total    = len(df)
        accepted = int((df['result'] == 'ACCEPTED').sum())
        rejected = df[df['result'] == 'REJECTED']
        reason_counts = rejected['reason'].value_counts()
        return {
            'Fusion_Total':              total,
            'Fusion_Accepted':           accepted,
            'Fusion_Accept_Rate':        accepted / total if total > 0 else float('nan'),
            'Fusion_Reject_Centroid':    int(reason_counts.get('centroid', 0)),
            'Fusion_Reject_AABB':        int(reason_counts.get('aabb', 0)),
            'Fusion_Reject_CosSim':      int(reason_counts.get('cos_sim', 0)),
            'Fusion_Reject_Overlap':     int(reason_counts.get('overlap', 0)),
            'Fusion_Reject_Cooccurrence': int(reason_counts.get('cooccurrence', 0)),
        }
    except Exception:
        return _FUSION_DECISION_ZERO.copy()


def _aggregate_fusion_stats(stats_list: list[dict]) -> dict:
    """Sum per-scene fusion stats into experiment-level totals."""
    total    = sum(s['Fusion_Total']           for s in stats_list)
    accepted = sum(s['Fusion_Accepted']        for s in stats_list)
    return {
        'Fusion_Total':               total,
        'Fusion_Accepted':            accepted,
        'Fusion_Accept_Rate':         accepted / total if total > 0 else float('nan'),
        'Fusion_Reject_Centroid':     sum(s['Fusion_Reject_Centroid']     for s in stats_list),
        'Fusion_Reject_AABB':         sum(s['Fusion_Reject_AABB']         for s in stats_list),
        'Fusion_Reject_CosSim':       sum(s['Fusion_Reject_CosSim']       for s in stats_list),
        'Fusion_Reject_Overlap':      sum(s['Fusion_Reject_Overlap']      for s in stats_list),
        'Fusion_Reject_Cooccurrence': sum(s['Fusion_Reject_Cooccurrence'] for s in stats_list),
    }


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

            ap_file = folder / 'replica' / 'instance_ap.txt'
            ap = parse_instance_ap_file(ap_file) if ap_file.is_file() else None

            # Aggregate fusion decisions across all scene subdirs.
            scene_fusion_stats = [
                parse_fusion_decisions(d / 'fusion_decisions.csv')
                for d in sorted(folder.iterdir())
                if d.is_dir() and (d / 'fusion_decisions.csv').is_file()
            ]
            fusion_agg = (
                _aggregate_fusion_stats(scene_fusion_stats)
                if scene_fusion_stats else _FUSION_DECISION_ZERO.copy()
            )

            exp_row: dict = {
                **meta,
                'Dataset':            dataset,
                'mIoU':               stats['mIoU']        if stats else float('nan'),
                'mAcc':               stats['mAcc']        if stats else float('nan'),
                'Head_mIoU':          stats['Head_mIoU']   if stats else float('nan'),
                'Head_mAcc':          stats['Head_mAcc']   if stats else float('nan'),
                'Common_mIoU':        stats['Common_mIoU'] if stats else float('nan'),
                'Common_mAcc':        stats['Common_mAcc'] if stats else float('nan'),
                'Tail_mIoU':          stats['Tail_mIoU']   if stats else float('nan'),
                'Tail_mAcc':          stats['Tail_mAcc']   if stats else float('nan'),
                'Num_Instances':      num_instances,
                'AP':                 ap.get('AP',               float('nan')) if ap else float('nan'),
                'AP_50':              ap.get('AP_50',            float('nan')) if ap else float('nan'),
                'AP_25':              ap.get('AP_25',            float('nan')) if ap else float('nan'),
                'AP_agnostic':        ap.get('AP_agnostic',      float('nan')) if ap else float('nan'),
                'AP_agnostic_50':     ap.get('AP_agnostic_50',   float('nan')) if ap else float('nan'),
                'AP_agnostic_25':     ap.get('AP_agnostic_25',   float('nan')) if ap else float('nan'),
                **fusion_agg,
                'Experiment_ID':      folder.name,
            }
            exp_rows.append(exp_row)

            if stats:
                for _, cls_row in stats['per_class'].iterrows():
                    class_rows.append({**exp_row, **cls_row.to_dict()})

    _exp_cols = [
        'Date', 'Dataset', 'SLAM_Config', 'Fusion', 'Label', 'Method',
        'Trans_Noise', 'Rot_Noise', 'Jump_Count', 'mIoU', 'mAcc',
        'Head_mIoU', 'Head_mAcc', 'Common_mIoU', 'Common_mAcc', 'Tail_mIoU', 'Tail_mAcc',
        'Num_Instances',
        'AP', 'AP_50', 'AP_25', 'AP_agnostic', 'AP_agnostic_50', 'AP_agnostic_25',
        'Fusion_Total', 'Fusion_Accepted', 'Fusion_Accept_Rate',
        'Fusion_Reject_Centroid', 'Fusion_Reject_AABB', 'Fusion_Reject_CosSim', 'Fusion_Reject_Overlap', 'Fusion_Reject_Cooccurrence',
        'Experiment_ID',
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


def get_available_fusions(df: pd.DataFrame) -> list[str]:
    """Return sorted unique Fusion values present in the DataFrame."""
    return sorted(df['Fusion'].dropna().unique().tolist())


def get_available_slam_configs(df: pd.DataFrame) -> list[str]:
    """Return sorted unique SLAM_Config values present in the DataFrame."""
    return sorted(df['SLAM_Config'].dropna().unique().tolist())


def get_available_noise_levels(df: pd.DataFrame) -> list[float]:
    """Return sorted unique Trans_Noise values present in the DataFrame."""
    return sorted(df['Trans_Noise'].dropna().unique().tolist())


def get_available_jump_counts(df: pd.DataFrame) -> list[int]:
    """Return sorted unique Jump_Count values present in the DataFrame."""
    return sorted(df['Jump_Count'].dropna().unique().tolist())


def filter_experiments(
    df: pd.DataFrame,
    dates: list[str] | None = None,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    fusions: list[str] | None = None,
    slam_configs: list[str] | None = None,
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
    if fusions is not None:
        mask &= df['Fusion'].isin(fusions)
    if slam_configs is not None:
        mask &= df['SLAM_Config'].isin(slam_configs)
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

                ap_scene_file = folder / 'replica' / f'instance_ap_{scene_name}.txt'
                ap = parse_instance_ap_file(ap_scene_file) if ap_scene_file.is_file() else None

                fusion_csv = folder / scene_name / 'fusion_decisions.csv'
                fusion = parse_fusion_decisions(fusion_csv)

                scene_row: dict = {
                    **meta,
                    'Dataset':            dataset,
                    'Scene':              scene_name,
                    'mIoU':               stats['mIoU']        if stats else float('nan'),
                    'mAcc':               stats['mAcc']        if stats else float('nan'),
                    'Head_mIoU':          stats['Head_mIoU']   if stats else float('nan'),
                    'Head_mAcc':          stats['Head_mAcc']   if stats else float('nan'),
                    'Common_mIoU':        stats['Common_mIoU'] if stats else float('nan'),
                    'Common_mAcc':        stats['Common_mAcc'] if stats else float('nan'),
                    'Tail_mIoU':          stats['Tail_mIoU']   if stats else float('nan'),
                    'Tail_mAcc':          stats['Tail_mAcc']   if stats else float('nan'),
                    'Num_Instances':      num_instances,
                    'AP':                 ap.get('AP',               float('nan')) if ap else float('nan'),
                    'AP_50':              ap.get('AP_50',            float('nan')) if ap else float('nan'),
                    'AP_25':              ap.get('AP_25',            float('nan')) if ap else float('nan'),
                    'AP_agnostic':        ap.get('AP_agnostic',      float('nan')) if ap else float('nan'),
                    'AP_agnostic_50':     ap.get('AP_agnostic_50',   float('nan')) if ap else float('nan'),
                    'AP_agnostic_25':     ap.get('AP_agnostic_25',   float('nan')) if ap else float('nan'),
                    **fusion,
                    'Experiment_ID':      folder.name,
                }
                scene_rows.append(scene_row)

                if stats:
                    for _, cls_row in stats['per_class'].iterrows():
                        scene_class_rows.append({**scene_row, **cls_row.to_dict()})

    _scene_cols = [
        'Date', 'Dataset', 'SLAM_Config', 'Fusion', 'Label', 'Method',
        'Trans_Noise', 'Rot_Noise', 'Jump_Count', 'Scene', 'mIoU', 'mAcc',
        'Head_mIoU', 'Head_mAcc', 'Common_mIoU', 'Common_mAcc', 'Tail_mIoU', 'Tail_mAcc',
        'Num_Instances',
        'AP', 'AP_50', 'AP_25', 'AP_agnostic', 'AP_agnostic_50', 'AP_agnostic_25',
        'Fusion_Total', 'Fusion_Accepted', 'Fusion_Accept_Rate',
        'Fusion_Reject_Centroid', 'Fusion_Reject_AABB', 'Fusion_Reject_CosSim', 'Fusion_Reject_Overlap', 'Fusion_Reject_Cooccurrence',
        'Experiment_ID',
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
    fusions: list[str] | None = None,
    slam_configs: list[str] | None = None,
    noise_levels: list[float] | None = None,
    exclude_experiment_ids: list[str] | None = None,
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
    if fusions is not None:
        mask &= df['Fusion'].isin(fusions)
    if slam_configs is not None:
        mask &= df['SLAM_Config'].isin(slam_configs)
    if noise_levels is not None:
        mask &= df['Trans_Noise'].isin(noise_levels)
    if exclude_experiment_ids is not None:
        mask &= ~df['Experiment_ID'].isin(exclude_experiment_ids)
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
    style: str | None = None,
    figsize: tuple = (12, 6),
) -> Figure:
    """Line plot of one or more metrics vs a continuous variable.

    Parameters
    ----------
    df:      Experiment or per-scene DataFrame.
    metrics: Columns to plot (default: ['mIoU', 'mAcc']).
    x:       Column for x-axis ('Trans_Noise' or 'Jump_Count').
    hue:     Column for colour grouping.
    style:   Optional column for line style grouping (e.g. 'Scene').
    figsize: Figure dimensions in inches.
    """
    if metrics is None:
        metrics = ['mIoU', 'mAcc']

    available = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not available:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data for selected metrics', ha='center', va='center')
        return fig

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    for i, metric in enumerate(available):
        ax = axes[0][i]
        kwargs = dict(data=df, x=x, y=metric, hue=hue, ax=ax, marker='o')
        if style is not None and style in df.columns:
            kwargs['style'] = style
        sns.lineplot(**kwargs)
        ax.set_title(f'{metric} vs {x}')
        ax.set_xlabel(x)
        ax.set_ylabel(metric)
        legend_title = f'{hue} / {style}' if style else hue
        ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
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


def plot_radar_chart(
    df: pd.DataFrame,
    experiment_ids: list[str],
    metrics: list[str] | None = None,
    figsize: tuple = (7, 7),
) -> Figure:
    """Radar/spider chart comparing experiments across multiple metrics.

    Each axis is normalised to [0, max] across the selected experiments.
    NaN values are drawn as 0.
    """
    if metrics is None:
        metrics = ['mIoU', 'mAcc', 'AP_agnostic', 'Num_Instances', 'Fusion_Accept_Rate']

    df_sel = df[df['Experiment_ID'].isin(experiment_ids)].copy()
    # Drop metrics that are all-NaN for selected experiments.
    available = [m for m in metrics if m in df_sel.columns and df_sel[m].notna().any()]
    if not available:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    n = len(available)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    maxvals = df_sel[available].max().replace(0, 1)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})
    colors = plt.cm.tab10.colors

    for idx, (_, row) in enumerate(df_sel.iterrows()):
        vals = [0.0 if pd.isna(row.get(m, float('nan'))) else float(row[m]) / float(maxvals[m])
                for m in available]
        vals += vals[:1]
        label = f"{row['Label']} ({str(row['Date'])[:10]})"
        ax.plot(angles, vals, color=colors[idx % 10], linewidth=2, label=label)
        ax.fill(angles, vals, color=colors[idx % 10], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, size=9)
    ax.set_ylim(0, 1)
    ax.set_title('Experiment comparison (normalised)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def plot_compare_scene_bars(
    df_scene: pd.DataFrame,
    experiment_ids: list[str],
    metrics: list[str] | None = None,
    figsize: tuple = (14, 10),
) -> Figure:
    """Faceted bar chart: one subplot per metric, X=Scene, hue=experiment.

    Default metrics: mIoU, AP_agnostic, Num_Instances.
    """
    if metrics is None:
        metrics = ['mIoU', 'AP_agnostic', 'Num_Instances']

    df_sel = df_scene[df_scene['Experiment_ID'].isin(experiment_ids)].copy()
    df_sel['_label'] = df_sel['Label'] + '\n' + df_sel['Date'].astype(str).str[:10]

    available = [m for m in metrics if m in df_sel.columns]
    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        sns.barplot(data=df_sel, x='Scene', y=metric, hue='_label', ax=ax)
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.legend(title='Experiment', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)

    axes[-1].set_xlabel('Scene')
    fig.suptitle('Scene-by-scene comparison', y=1.01)
    fig.tight_layout()
    return fig


def plot_delta_heatmap(
    df_scene: pd.DataFrame,
    baseline_id: str,
    compare_ids: list[str],
    metrics: list[str] | None = None,
    figsize: tuple = (10, 6),
) -> Figure:
    """Heatmap of metric deltas vs baseline, per scene × experiment.

    Rows = scenes, columns = (experiment, metric) pairs.
    Cell value = experiment_value − baseline_value. Diverging colormap.
    """
    if metrics is None:
        metrics = ['mIoU', 'mAcc', 'AP_agnostic']

    available = [m for m in metrics if m in df_scene.columns]
    all_ids = [baseline_id] + compare_ids
    df_sel = df_scene[df_scene['Experiment_ID'].isin(all_ids)].copy()

    baseline = df_sel[df_sel['Experiment_ID'] == baseline_id].set_index('Scene')

    records = []
    for exp_id in compare_ids:
        exp_df = df_sel[df_sel['Experiment_ID'] == exp_id].set_index('Scene')
        short = exp_df['Label'].iloc[0] if not exp_df.empty else exp_id
        for scene in exp_df.index:
            for m in available:
                if scene in baseline.index:
                    delta = float(exp_df.loc[scene, m]) - float(baseline.loc[scene, m])
                    records.append({'Scene': scene, 'col': f'{short}\n{m}', 'delta': delta})

    if not records:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center')
        return fig

    df_delta = pd.DataFrame(records)
    pivot = df_delta.pivot_table(index='Scene', columns='col', values='delta')
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].max()), 1e-6)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, ax=ax, cmap='RdYlGn', center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.4, linecolor='grey', annot=True, fmt='.3f',
        cbar_kws={'label': 'Δ vs baseline'},
    )
    ax.set_title(f'Delta vs baseline ({baseline_id})')
    fig.tight_layout()
    return fig


def plot_fusion_compare(
    df_scene: pd.DataFrame,
    experiment_ids: list[str],
    figsize: tuple = (14, 8),
) -> Figure:
    """Two-panel fusion behaviour comparison.

    Top: accept rate per scene per experiment (grouped bars).
    Bottom: rejection reason breakdown per experiment (stacked bars).
    """
    df_sel = df_scene[df_scene['Experiment_ID'].isin(experiment_ids)].copy()
    df_sel['_label'] = df_sel['Label'] + '\n' + df_sel['Date'].astype(str).str[:10]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    if 'Fusion_Accept_Rate' in df_sel.columns:
        sns.barplot(data=df_sel, x='Scene', y='Fusion_Accept_Rate', hue='_label', ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accept Rate')
        ax1.set_xlabel('')
        ax1.set_title('Fusion accept rate per scene')
        ax1.legend(title='Experiment', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)

    reason_cols = ['Fusion_Reject_Centroid', 'Fusion_Reject_AABB', 'Fusion_Reject_CosSim', 'Fusion_Reject_Overlap', 'Fusion_Reject_Cooccurrence']
    available_reasons = [c for c in reason_cols if c in df_sel.columns]
    if available_reasons:
        agg = (
            df_sel.groupby('_label')[available_reasons]
            .sum()
            .rename(columns={c: c.replace('Fusion_Reject_', '') for c in available_reasons})
        )
        agg.plot(kind='bar', stacked=True, ax=ax2, colormap='tab10')
        ax2.set_ylabel('Rejected pairs')
        ax2.set_xlabel('Experiment')
        ax2.set_title('Rejection reason breakdown (all scenes summed)')
        ax2.tick_params(axis='x', rotation=30)
        ax2.legend(title='Reason', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)

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
