"""Contest Tier2 telemetry visualization: per-signal per-KF figures."""

from .loader import (
    SceneContestData,
    discover_scenes,
    load_scene,
    load_experiment,
    contest_log_dir,
    OUTPUT_ROOT,
    KF_SIGNALS,
    SIGNAL_META,
)
from .figures_mpl import signal_figure, save_scene_figures
