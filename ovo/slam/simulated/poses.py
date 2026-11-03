"""PoseSimulator: answers "what pose does the simulated SLAM estimate for frame N?".

    estimated(N) = error · GT(N)

`error` is the accumulated world-frame error. Each tracked frame, drift adds its per-frame error
on top (in camera axes) and any jump that is due moves the pose. The first tracked frame is
exact: like a real SLAM, the simulator anchors the world there. A correction (simulated loop
closure) clears the error, which then starts over from zero (with drift, the next frame already
carries one step of it). After `stop()` (the final correction) every estimate is the exact GT.
While nothing has gone wrong, `error` is None and the estimate is exactly the GT.
"""

from typing import Optional

import torch

from .drift import Drift
from .jumps import Jump


class PoseSimulator:
    def __init__(self, drift: Optional[Drift], jumps: list[Jump]) -> None:
        self.drift = drift
        self._pending = sorted(jumps, key=lambda j: j.at_frame)
        self._error: Optional[torch.Tensor] = None
        self._started = False
        self._stopped = False

    @property
    def has_error(self) -> bool:
        return self._error is not None

    def track(self, frame_id: int, gt: torch.Tensor) -> tuple[torch.Tensor, list[Jump]]:
        """Estimated pose of `frame_id` from its GT (4x4 float64), and the jumps fired at it."""
        if self._stopped:
            return gt, []
        estimated = gt if self._error is None else self._error @ gt
        if self.drift is not None and self._started:
            estimated = estimated @ self.drift.step()
        self._started = True

        fired = []
        while self._pending and self._pending[0].at_frame <= frame_id:
            jump = self._pending.pop(0)
            estimated = jump.apply(estimated)
            fired.append(jump)

        if estimated is not gt:
            self._error = estimated @ torch.linalg.inv(gt)
        return estimated, fired

    def correct(self) -> None:
        """Simulated loop closure: from now on the estimate starts again from the GT."""
        self._error = None

    def stop(self) -> None:
        """Final correction: no more error for the rest of the sequence (pending jumps are dropped)."""
        self._error = None
        self._stopped = True
