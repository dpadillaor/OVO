"""Rerun output destinations. Each sink owns one recording plus that destination's policy."""

from typing import Sequence

import numpy as np
import rerun as rr


class RerunSink:
    """One rerun recording, with its own point budget and static-logging policy."""

    def __init__(self, recording=None, max_points: int | None = None, allow_static: bool = True):
        self.recording = recording
        self.max_points = max_points
        self.allow_static = allow_static
        self._rec = {"recording": recording} if recording is not None else {}

    def set_time(self, timeline: str, sequence: int) -> None:
        rr.set_time(timeline, sequence=sequence, **self._rec)

    def log(self, path: str, entity, *, static: bool = False) -> None:
        rr.log(path, entity, static=static and self.allow_static, **self._rec)

    def clear(self, path: str) -> None:
        self.log(path, rr.Clear(recursive=False))

    def send_blueprint(self, blueprint) -> None:
        rr.send_blueprint(blueprint, **self._rec)

    def budget_indices(self, n: int) -> np.ndarray | None:
        """Sorted random subset of range(n) fitting this sink's budget; None means keep everything."""
        if self.max_points is None or n <= self.max_points:
            return None
        idx = np.random.choice(n, self.max_points, replace=False)
        idx.sort()
        return idx


def log_all(sinks: Sequence[RerunSink], path: str, entity, *, static: bool = False) -> None:
    for sink in sinks:
        sink.log(path, entity, static=static)


def set_time_all(sinks: Sequence[RerunSink], timeline: str, sequence: int) -> None:
    for sink in sinks:
        sink.set_time(timeline, sequence)


def clear_all(sinks: Sequence[RerunSink], path: str) -> None:
    for sink in sinks:
        sink.clear(path)
