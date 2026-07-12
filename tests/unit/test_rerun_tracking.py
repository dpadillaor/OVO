import numpy as np
import pytest

from ovo.entities.visualizers.rerun_frame import decode
from ovo.entities.visualizers.rerun_sink import RerunSink
from ovo.entities.visualizers.rerun_tracking import (
    SignalsPainter,
    TrackingPainter,
    _to_classes,
    split_layers,
)
from ovo.entities.visualizers.rerun_utils import get_instance_cmap


class FakeSink(RerunSink):
    """Sink that records what would be logged instead of talking to rerun."""

    def __init__(self, max_points=None):
        super().__init__(recording=None, max_points=max_points)
        self.logged: dict[str, object] = {}
        self.cleared: list[str] = []

    def log(self, path, entity, *, static=False):
        self.logged[path] = entity

    def clear(self, path):
        self.cleared.append(path)


def make_frame(points, point_ids, signals, frame_id=7):
    """Build a Frame through the real decoder, so the ceiling cut is exercised too."""
    return decode(
        {
            "type": "stream_frame",
            "frame_id": frame_id,
            "points": points,
            "obj_ids": np.zeros(len(points), dtype=np.int16),
            "colors": None,
            "normals": None,
            "c2w": np.eye(4, dtype=np.float32),
            "rgb": None,
            "ins_map": None,
            "sam_map": None,
            "corrected_trajectory": None,
            "point_ids": point_ids,
            "track_signals": signals,
        },
        fallback_step=0,
    )


# 4 map points below the ceiling + 1 ceiling point that the decoder must cut.
POINTS_4 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 5]], dtype=np.float32)
IDS_4 = np.array([10, 11, 12, 13, 99], dtype=np.int64)


@pytest.fixture
def painter():
    return TrackingPainter(get_instance_cmap())


class TestBudget:

    def test_budget_indices_none_when_under_limit(self):
        assert RerunSink(max_points=10).budget_indices(10) is None

    def test_budget_indices_are_sorted_and_capped(self):
        idx = RerunSink(max_points=3).budget_indices(100)
        assert len(idx) == 3
        assert list(idx) == sorted(idx)
        assert len(set(idx)) == 3

    def test_no_budget_keeps_everything(self):
        assert RerunSink().budget_indices(1_000_000) is None


class TestPartition:

    def test_splits_matched_points_by_pre_existing_instance(self, painter):
        signals = {
            "frame_id": 7,
            "matched_ids": np.array([10, 11, 12, 13], dtype=np.int64),
            "matched_px": np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32),
            "matched_pre": np.array([True, False, True, False]),
            "robbed_ids": np.array([12], dtype=np.int64),
            "birth_ids": np.array([13], dtype=np.int64),
        }
        layers = split_layers(make_frame(POINTS_4, IDS_4, signals))

        assert list(layers["assigned"][0]) == [10, 12]
        assert list(layers["unassigned"][0]) == [11, 13]
        assert list(layers["robbed"][0]) == [12]
        assert list(layers["births"][0]) == [13]

    def test_paints_only_matched_points_in_3d(self, painter):
        # 4 map points below the ceiling, only two of them matched this KF (+1 ceiling point, cut).
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 5]], dtype=np.float32
        )
        signals = {
            "frame_id": 7,
            "matched_ids": np.array([10, 12], dtype=np.int64),
            "matched_px": np.array([[0, 0], [2, 2]], dtype=np.int32),
            "matched_pre": np.array([True, False]),
            "robbed_ids": np.array([], dtype=np.int64),
            "birth_ids": np.array([], dtype=np.int64),
        }
        frame = make_frame(points, np.array([10, 11, 12, 13, 99], dtype=np.int64), signals)
        sink = FakeSink()

        painter.paint([sink], frame, split_layers(frame))

        assert len(sink.logged["tracking/greymap"].positions) == 4  # ceiling point cut
        assert len(sink.logged["tracking/assigned"].positions) == 1
        assert len(sink.logged["tracking/unassigned"].positions) == 1
        assert len(sink.logged["frame/pts2d/assigned"].positions) == 1
        # Empty subsets are cleared, not left behind from the previous KF.
        assert "tracking/robbed" in sink.cleared
        assert "frame/pts2d/births" in sink.cleared


class TestStaleSignals:

    def test_clears_layers_when_signals_belong_to_another_frame(self, painter):
        # update_map re-sends the frame with a newer frame_id but the previous KF's signals.
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 5]], dtype=np.float32)
        stale = {
            "frame_id": 5,
            "matched_ids": np.array([10], dtype=np.int64),
            "matched_px": np.array([[0, 0]], dtype=np.int32),
            "matched_pre": np.array([True]),
            "robbed_ids": np.array([], dtype=np.int64),
            "birth_ids": np.array([], dtype=np.int64),
        }
        frame = make_frame(points, np.array([10, 11, 99], dtype=np.int64), stale, frame_id=9)
        sink = FakeSink()

        painter.paint([sink], frame, split_layers(frame))

        assert "tracking/assigned" not in sink.logged
        assert "tracking/assigned" in sink.cleared
        assert "frame/pts2d/assigned" in sink.cleared
        assert "tracking/greymap" in sink.logged  # the map itself still shows

    def test_no_signals_at_all_is_safe(self, painter):
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 5]], dtype=np.float32)
        frame = make_frame(points, np.array([10, 11, 99], dtype=np.int64), None)

        painter.paint([FakeSink()], frame, split_layers(frame))


class TestSignals:

    def test_counts_are_the_layer_sizes(self):
        # The graphs must be readable back onto the 3D layers, so they are derived from them.
        signals = {
            "frame_id": 7,
            "matched_ids": np.array([10, 11, 12, 13], dtype=np.int64),
            "matched_px": np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32),
            "matched_pre": np.array([True, False, True, False]),
            "robbed_ids": np.array([12], dtype=np.int64),
            "birth_ids": np.array([13], dtype=np.int64),
        }
        frame = make_frame(POINTS_4, IDS_4, signals)
        sink = FakeSink()

        SignalsPainter().paint([sink], frame, split_layers(frame))

        assert sink.logged["signals/count/assigned"].scalars.as_arrow_array().to_pylist() == [2.0]
        assert sink.logged["signals/count/unassigned"].scalars.as_arrow_array().to_pylist() == [2.0]
        assert sink.logged["signals/count/robbed"].scalars.as_arrow_array().to_pylist() == [1.0]
        assert sink.logged["signals/count/births"].scalars.as_arrow_array().to_pylist() == [1.0]

    def test_stale_signals_emit_no_scalars(self):
        stale = {
            "frame_id": 5,
            "matched_ids": np.array([10], dtype=np.int64),
            "matched_px": np.array([[0, 0]], dtype=np.int32),
            "matched_pre": np.array([True]),
            "robbed_ids": np.array([], dtype=np.int64),
            "birth_ids": np.array([], dtype=np.int64),
        }
        frame = make_frame(POINTS_4, IDS_4, stale, frame_id=9)
        sink = FakeSink()

        SignalsPainter().paint([sink], frame, split_layers(frame))

        assert sink.logged == {}


class TestSegClasses:

    def test_background_maps_to_transparent_class_zero(self):
        classes = _to_classes(np.array([[-1, 0], [1, 39]], dtype=np.int32), n_colors=40)
        assert classes[0, 0] == 0
        assert classes[0, 1] == 1
        assert classes[1, 1] == 40

    def test_ids_wrap_around_the_palette(self):
        classes = _to_classes(np.array([[40, 41]], dtype=np.int32), n_colors=40)
        assert list(classes[0]) == [1, 2]
