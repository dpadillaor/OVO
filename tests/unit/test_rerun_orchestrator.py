from ovo.entities.visualizers import rerun_orchestrator as orch
from ovo.entities.visualizers.rerun_utils import QUEUE_EMPTY


class DummyRenderer:
    name = "DummyRenderer"
    idle_sleep_s = 0.001
    error_sleep_s = 0.002

    def __init__(self):
        self.setup_called = False
        self.messages = []
        self.raise_on_handle = False

    def setup(self):
        self.setup_called = True

    def handle_message(self, data):
        if self.raise_on_handle:
            raise RuntimeError("boom")
        self.messages.append(data)


class TestQueueRendererOrchestrator:

    def test_run_processes_messages_until_sentinel(self, monkeypatch):
        renderer = DummyRenderer()
        orchestrator = orch.QueueRendererOrchestrator(renderer, mpqueue=object())

        seq = iter([QUEUE_EMPTY, {"foo": 1}, None])

        monkeypatch.setattr(orch, "get_nowait_or_empty", lambda _: next(seq))
        monkeypatch.setattr(orch, "drain_until_sentinel", lambda _: False)

        sleeps = []
        monkeypatch.setattr(orch.time, "sleep", lambda t: sleeps.append(t))

        orchestrator.run()

        assert renderer.setup_called
        assert renderer.messages == [{"foo": 1}]
        assert sleeps == [renderer.idle_sleep_s]

    def test_run_stops_when_drain_finds_sentinel_after_exception(self, monkeypatch):
        renderer = DummyRenderer()
        renderer.raise_on_handle = True
        orchestrator = orch.QueueRendererOrchestrator(renderer, mpqueue=object())

        seq = iter([{"foo": 1}])
        monkeypatch.setattr(orch, "get_nowait_or_empty", lambda _: next(seq))

        drained = {"called": False}

        def _drain(_):
            drained["called"] = True
            return True

        monkeypatch.setattr(orch, "drain_until_sentinel", _drain)

        sleeps = []
        monkeypatch.setattr(orch.time, "sleep", lambda t: sleeps.append(t))

        orchestrator.run()

        assert renderer.setup_called
        assert drained["called"]
        assert sleeps == []
