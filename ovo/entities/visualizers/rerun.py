from .rerun_handlers import FusionRenderer, LoopClosureRenderer, StreamRenderer
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


def stream_rerun_fusion(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show, save_rrd=False):
    visual_mode = "spawn"
    if isinstance(query_data, (list, tuple)) and len(query_data) > 2:
        visual_mode = query_data[2]

    renderer = FusionRenderer(
        cam_intrinsic,
        scene_name,
        output_path,
        show,
        save_rrd=save_rrd,
        visual_mode=visual_mode,
    )
    QueueRendererOrchestrator(renderer, mpqueue).run()


def stream_rerun_loopclosure(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show, save_rrd=False):
    visual_mode = "spawn"
    if isinstance(query_data, (list, tuple)) and len(query_data) > 2:
        visual_mode = query_data[2]

    renderer = LoopClosureRenderer(
        cam_intrinsic,
        scene_name,
        output_path,
        show,
        save_rrd=save_rrd,
        visual_mode=visual_mode,
    )
    QueueRendererOrchestrator(renderer, mpqueue).run()
