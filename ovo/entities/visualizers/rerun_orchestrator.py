import time
from typing import Any, Protocol

from .rerun_utils import QUEUE_EMPTY, drain_until_sentinel, get_nowait_or_empty


class RendererProtocol(Protocol):
    name: str
    idle_sleep_s: float
    error_sleep_s: float

    def setup(self) -> None:
        ...

    def handle_message(self, data: Any) -> None:
        ...


class QueueRendererOrchestrator:
    """Run a renderer against a multiprocessing queue with shared error handling."""

    def __init__(self, renderer: RendererProtocol, mpqueue):
        self.renderer = renderer
        self.mpqueue = mpqueue

    def run(self):
        self.renderer.setup()

        while True:
            try:
                data = get_nowait_or_empty(self.mpqueue)
                if data is QUEUE_EMPTY:
                    time.sleep(self.renderer.idle_sleep_s)
                    continue

                if data is None:
                    break

                self.renderer.handle_message(data)
                print(f"[{self.renderer.name}] sent frame", flush=True)

            except Exception as exc:
                print(f"[{self.renderer.name}] Warning: {exc}")
                if drain_until_sentinel(self.mpqueue):
                    return
                time.sleep(self.renderer.error_sleep_s)
