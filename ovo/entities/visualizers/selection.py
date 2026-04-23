from .rerun import stream_rerun, stream_rerun_fusion, stream_rerun_loopclosure
from ..visualizer import stream_pcd


VALID_RERUN_VISUAL_MODES = {"off", "spawn", "serve"}


def resolve_rerun_visual_mode(config_mode: str | None, legacy_show_stream: bool) -> str:
    """Resolve rerun visual mode from explicit config or legacy show_stream flag."""
    if config_mode is None:
        return "spawn" if legacy_show_stream else "off"

    mode = str(config_mode).lower()
    if mode not in VALID_RERUN_VISUAL_MODES:
        raise ValueError(
            f"Invalid rerun_visual_mode='{config_mode}'. "
            f"Expected one of: {sorted(VALID_RERUN_VISUAL_MODES)}"
        )
    return mode


def select_visualizer_target(vis_type: str, rerun_mode: str):
    """Resolve process target function and process name for visual streaming."""
    if vis_type == "rerun":
        if rerun_mode == "fusion":
            return stream_rerun_fusion, "RerunFusionVis"
        if rerun_mode == "loop_closure":
            return stream_rerun_loopclosure, "RerunLCVis"
        return stream_rerun, "RerunVisualizer"

    return stream_pcd, "O3DVisualizer"
