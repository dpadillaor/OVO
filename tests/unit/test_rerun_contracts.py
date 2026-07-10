import numpy as np

from ovo.entities.visualizers.rerun_contracts import (
    is_stream_frame_message,
    is_stream_message,
)


class TestRerunMessageContracts:

    def test_is_stream_message_accepts_valid_tuple(self):
        msg = (
            np.zeros((10, 3), dtype=np.float32),
            np.zeros((10,), dtype=np.int32),
            None,
            np.eye(4, dtype=np.float32),
        )
        assert is_stream_message(msg)

    def test_is_stream_message_rejects_wrong_shape(self):
        msg = (
            np.zeros((10, 3), dtype=np.float32),
            np.zeros((10,), dtype=np.int32),
            None,
        )
        assert not is_stream_message(msg)

    def test_is_stream_message_rejects_non_array_payload(self):
        msg = ("points", "ids", None, "c2w")
        assert not is_stream_message(msg)

    def test_is_stream_frame_message_accepts_valid_payload(self):
        msg = {
            "type": "stream_frame",
            "frame_id": 11,
            "points": np.zeros((10, 3), dtype=np.float32),
            "obj_ids": np.zeros((10,), dtype=np.int32),
            "colors": None,
            "c2w": np.eye(4, dtype=np.float32),
            "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "ins_map": np.zeros((8, 8), dtype=np.int32),
        }
        assert is_stream_frame_message(msg)

    def test_is_stream_frame_message_accepts_optional_none_fields(self):
        msg = {
            "type": "stream_frame",
            "frame_id": 11,
            "points": np.zeros((10, 3), dtype=np.float32),
            "obj_ids": np.zeros((10,), dtype=np.int32),
            "colors": None,
            "c2w": np.eye(4, dtype=np.float32),
            "rgb": None,
            "ins_map": None,
        }
        assert is_stream_frame_message(msg)

    def test_is_stream_frame_message_rejects_missing_keys(self):
        msg = {
            "type": "stream_frame",
            "frame_id": 11,
            "points": np.zeros((10, 3), dtype=np.float32),
            "obj_ids": np.zeros((10,), dtype=np.int32),
            "c2w": np.eye(4, dtype=np.float32),
        }
        assert not is_stream_frame_message(msg)

    def test_is_stream_frame_message_accepts_tracking_extras(self):
        # tracking mode adds point_ids + track_signals; the guard must still accept it
        msg = {
            "type": "stream_frame",
            "frame_id": 11,
            "points": np.zeros((10, 3), dtype=np.float32),
            "obj_ids": np.zeros((10,), dtype=np.int32),
            "colors": None,
            "c2w": np.eye(4, dtype=np.float32),
            "rgb": None,
            "ins_map": None,
            "point_ids": np.arange(10, dtype=np.int64),
            "track_signals": {"frame_id": 11, "n_robos": 3, "robbed_ids": [1, 2], "new_ids": []},
        }
        assert is_stream_frame_message(msg)
