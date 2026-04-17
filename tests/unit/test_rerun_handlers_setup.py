from ovo.entities.visualizers import rerun_handlers as handlers


class DummyRenderer(handlers.BaseRerunRenderer):
    def build_blueprint(self):
        return "blueprint"

    def post_setup(self):
        pass

    def handle_message(self, data):
        return None


class TestBaseRerunRendererSetup:

    def test_setup_spawn_mode_respects_show_flag(self, monkeypatch, tmp_path):
        calls = {"init": [], "connect": [], "save": [], "send_blueprint": []}

        monkeypatch.setattr(handlers.rr, "init", lambda *args, **kwargs: calls["init"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "connect", lambda *args, **kwargs: calls["connect"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "save", lambda *args, **kwargs: calls["save"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "send_blueprint", lambda *args, **kwargs: calls["send_blueprint"].append((args, kwargs)))

        renderer = DummyRenderer(
            cam_intrinsic={"width": 1, "height": 1, "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            scene_name="scene",
            output_path=tmp_path,
            show=True,
            save_rrd=False,
            visual_mode="spawn",
        )
        renderer.setup()

        assert calls["init"][0][1]["spawn"] is True
        assert calls["connect"] == []
        assert len(calls["send_blueprint"]) == 1

    def test_setup_connect_mode_calls_connect_and_can_save(self, monkeypatch, tmp_path):
        calls = {"init": [], "connect": [], "save": [], "send_blueprint": []}

        monkeypatch.setattr(handlers.rr, "init", lambda *args, **kwargs: calls["init"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "connect", lambda *args, **kwargs: calls["connect"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "save", lambda *args, **kwargs: calls["save"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "send_blueprint", lambda *args, **kwargs: calls["send_blueprint"].append((args, kwargs)))

        renderer = DummyRenderer(
            cam_intrinsic={"width": 1, "height": 1, "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            scene_name="scene",
            output_path=tmp_path,
            show=True,
            save_rrd=True,
            visual_mode="connect",
            connect_addr="127.0.0.1:9876",
        )
        renderer.setup()

        assert calls["init"][0][1]["spawn"] is False
        assert calls["connect"][0][0] == ("127.0.0.1:9876",)
        assert len(calls["save"]) == 1

    def test_setup_connect_mode_requires_address(self, monkeypatch, tmp_path):
        monkeypatch.setattr(handlers.rr, "init", lambda *args, **kwargs: None)
        monkeypatch.setattr(handlers.rr, "send_blueprint", lambda *args, **kwargs: None)

        renderer = DummyRenderer(
            cam_intrinsic={"width": 1, "height": 1, "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            scene_name="scene",
            output_path=tmp_path,
            show=True,
            save_rrd=False,
            visual_mode="connect",
            connect_addr=None,
        )

        try:
            renderer.setup()
            assert False, "Expected ValueError when connect_addr is missing"
        except ValueError as exc:
            assert "connect_addr" in str(exc)
