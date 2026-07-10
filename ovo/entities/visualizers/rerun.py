from .rerun_handlers import StreamRenderer, TrackingRenderer
from .rerun_orchestrator import QueueRendererOrchestrator


def stream_rerun(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show, save_rrd=False):
    visual_mode = "spawn"
    if isinstance(query_data, (list, tuple)) and len(query_data) > 2:
        visual_mode = query_data[2]

    renderer = StreamRenderer(
        cam_intrinsic,
        scene_name,
        output_path,
        show,
        save_rrd=save_rrd,
        visual_mode=visual_mode,
    )
    QueueRendererOrchestrator(renderer, mpqueue).run()


def stream_rerun_tracking(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show, save_rrd=False):
    visual_mode = "spawn"
    if isinstance(query_data, (list, tuple)) and len(query_data) > 2:
        visual_mode = query_data[2]

    renderer = TrackingRenderer(
        cam_intrinsic,
        scene_name,
        output_path,
        show,
        save_rrd=save_rrd,
        visual_mode=visual_mode,
    )
    QueueRendererOrchestrator(renderer, mpqueue).run()
