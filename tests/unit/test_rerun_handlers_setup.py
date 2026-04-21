from ovo.entities.visualizers import rerun_handlers as handlers


class DummyRenderer(handlers.BaseRerunRenderer):
    def build_blueprint(self):
        return "blueprint"

    def post_setup(self):
        pass

    def handle_message(self, data):
        return None


class TestBaseRerunRendererSetup:

    def test_setup_spawn_mode_with_show_calls_spawn(self, monkeypatch, tmp_path):
        calls = {"init": [], "spawn": [], "send_blueprint": []}

        monkeypatch.setattr(handlers.rr, "init", lambda *args, **kwargs: calls["init"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "spawn", lambda *args, **kwargs: calls["spawn"].append((args, kwargs)))
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

        assert len(calls["spawn"]) == 1
        assert len(calls["send_blueprint"]) == 1

    def test_setup_spawn_mode_with_show_false_skips_spawn(self, monkeypatch, tmp_path):
        calls = {"init": [], "spawn": [], "send_blueprint": []}

        monkeypatch.setattr(handlers.rr, "init", lambda *args, **kwargs: calls["init"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "spawn", lambda *args, **kwargs: calls["spawn"].append((args, kwargs)))
        monkeypatch.setattr(handlers.rr, "send_blueprint", lambda *args, **kwargs: calls["send_blueprint"].append((args, kwargs)))

        renderer = DummyRenderer(
            cam_intrinsic={"width": 1, "height": 1, "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            scene_name="scene",
            output_path=tmp_path,
            show=False,
            save_rrd=False,
            visual_mode="spawn",
        )
        renderer.setup()

        assert calls["spawn"] == []
        assert len(calls["send_blueprint"]) == 1
