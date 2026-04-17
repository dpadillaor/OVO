import numpy as np

from ovo.entities.visualizers.rerun_contracts import (
    is_fusion_message,
    is_loop_closure_message,
    is_stream_message,
)


class TestRerunMessageContracts:

    def test_is_fusion_message_accepts_valid_payload(self):
        msg = {
            "type": "fusion",
            "points": np.zeros((10, 3), dtype=np.float32),
            "before_ids": np.zeros((10,), dtype=np.int32),
            "after_ids": np.ones((10,), dtype=np.int32),
            "frame_id": 42,
        }
        assert is_fusion_message(msg)

    def test_is_fusion_message_rejects_missing_key(self):
        msg = {
            "type": "fusion",
            "points": np.zeros((10, 3), dtype=np.float32),
            "before_ids": np.zeros((10,), dtype=np.int32),
            "frame_id": 42,
        }
        assert not is_fusion_message(msg)

    def test_is_fusion_message_rejects_wrong_type(self):
        msg = {
            "type": "loop_closure",
            "points": np.zeros((10, 3), dtype=np.float32),
            "before_ids": np.zeros((10,), dtype=np.int32),
            "after_ids": np.ones((10,), dtype=np.int32),
            "frame_id": 42,
        }
        assert not is_fusion_message(msg)

    def test_is_loop_closure_message_accepts_valid_payload(self):
        msg = {
            "type": "loop_closure",
            "pcd_before": np.zeros((10, 3), dtype=np.float32),
            "pcd_after": np.ones((10, 3), dtype=np.float32),
            "ids": np.zeros((10,), dtype=np.int32),
            "traj_before": {0: np.eye(4, dtype=np.float32)},
            "traj_after": {0: np.eye(4, dtype=np.float32)},
            "frame_id": 7,
        }
        assert is_loop_closure_message(msg)

    def test_is_loop_closure_message_rejects_missing_keys(self):
        msg = {
            "type": "loop_closure",
            "pcd_before": np.zeros((10, 3), dtype=np.float32),
            "ids": np.zeros((10,), dtype=np.int32),
        }
        assert not is_loop_closure_message(msg)

    def test_is_loop_closure_message_rejects_non_dict(self):
        assert not is_loop_closure_message(["not", "a", "dict"])

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
