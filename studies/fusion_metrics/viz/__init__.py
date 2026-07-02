"""Fusion evaluation visualization: scene, experiment, and cross-experiment dashboards."""

from .loader import (
    SceneFusionData,
    discover_scenes,
    load_scene,
    load_experiment,
    load_experiments,
    pool_verdicts,
    pool_instance_stats,
    OUTPUT_ROOT,
)
from .charts import (
    confusion_heatmap,
    gate_waterfall,
    gate_sankey,
    ap_curves,
    instance_composition,
    instance_waterfall,
    instance_scatter,
    epoch_heatmap,
    scene_dots,
    compare_heatmap,
    compare_scatter,
    scene_summary_blurb,
)
from .dashboards import (
    scene_dashboard,
    experiment_summary,
    compare_dashboard,
)
