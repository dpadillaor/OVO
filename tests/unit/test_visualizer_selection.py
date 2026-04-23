import pytest

from ovo.entities.visualizers.rerun import (
    stream_rerun,
    stream_rerun_fusion,
    stream_rerun_loopclosure,
)
from ovo.entities.visualizers.selection import resolve_rerun_visual_mode, select_visualizer_target
from ovo.entities.visualizer import stream_pcd


class TestSelectVisualizerTarget:

    def test_selects_rerun_stream_mode_by_default(self):
        func, name = select_visualizer_target("rerun", "stream")
        assert func is stream_rerun
        assert name == "RerunVisualizer"

    def test_selects_rerun_fusion_mode(self):
        func, name = select_visualizer_target("rerun", "fusion")
        assert func is stream_rerun_fusion
        assert name == "RerunFusionVis"

    def test_selects_rerun_loop_closure_mode(self):
        func, name = select_visualizer_target("rerun", "loop_closure")
        assert func is stream_rerun_loopclosure
        assert name == "RerunLCVis"

    def test_selects_open3d_for_non_rerun_type(self):
        func, name = select_visualizer_target("open3d", "fusion")
        assert func is stream_pcd
        assert name == "O3DVisualizer"


class TestResolveRerunVisualMode:

    def test_uses_explicit_mode_when_valid(self):
        assert resolve_rerun_visual_mode("serve", legacy_show_stream=True) == "serve"
        assert resolve_rerun_visual_mode("off", legacy_show_stream=True) == "off"

    def test_falls_back_to_legacy_show_stream_true(self):
        assert resolve_rerun_visual_mode(None, legacy_show_stream=True) == "spawn"

    def test_falls_back_to_legacy_show_stream_false(self):
        assert resolve_rerun_visual_mode(None, legacy_show_stream=False) == "off"

    def test_rejects_invalid_mode(self):
        with pytest.raises(ValueError):
            resolve_rerun_visual_mode("banana", legacy_show_stream=True)
