# Rerun Stream Data Flow (OVO)

Este documento resume dónde vive cada tipo de información y cómo fluye hasta Rerun.

## Vista general

    Dataset frame
    (image, depth, estimated pose)
        |
        v
    OVOSemMap._run_semantic_step
    -> ovo/entities/ovomapping.py:237
        |
        +--> OVO.detect_and_track_objects
        |    -> ovo/entities/ovo.py:184
        |        |
        |        +--> OVO._get_masks
        |        |    -> ovo/entities/ovo.py:232
        |        |        |
        |        |        +--> MaskGenerator.get_masks
        |        |             -> ovo/entities/mask_generator.py:81
        |        |             |
        |        |             +--> segment (SAM online)
        |        |             |    -> ovo/entities/mask_generator.py:102
        |        |             |
        |        |             +--> _load_masks (precomputed)
        |        |                  -> ovo/entities/mask_generator.py:170
        |        |
        |        +--> Semantic queue (internal OVO)
        |             keyframes_queue.append([matched_ins_ids, binary_maps, image, kf_id])
        |             -> ovo/entities/ovo.py:212
        |             consumed in -> ovo/entities/ovo.py:397 and ovo/entities/ovo.py:400
        |
        +--> SLAM map snapshot for rerun stream
             (points, obj_ids, colors, c2w)
             queued at -> ovo/entities/ovomapping.py:292

        |
        v
    mpqueue (IPC)
    non-blocking put/drop helper -> ovo/entities/ovomapping.py:18

        |
        v
    Rerun visualization process
    started by -> ovo/entities/ovomapping.py:169
        |
        +--> stream_rerun entrypoint
        |    -> ovo/entities/visualizers/rerun.py:5
        |
        +--> QueueRendererOrchestrator.run
             -> ovo/entities/visualizers/rerun_orchestrator.py:26
             |
             +--> StreamRenderer.handle_message
                  -> ovo/entities/visualizers/rerun_handlers.py:151
                  |
                  +--> rr.log world/points, world/camera, world/trajectory

## Contrato actual del stream

- Definición de mensaje stream: ovo/entities/visualizers/rerun_contracts.py:24
- Validador de mensaje stream: ovo/entities/visualizers/rerun_contracts.py:47
- Formato esperado hoy: tuple/list de 4 elementos
  - points
  - obj_ids
  - colors
  - c2w

## Qué se ve hoy vs qué no

- Sí se ve hoy en Rerun:
  - mapa 3D de puntos
  - pose de cámara
  - trayectoria
- No se ve hoy en Rerun:
  - frame RGB
  - máscaras SAM (seg_map o binary_maps)

## Por qué no aparece frame+máscaras hoy

La rama que alimenta Rerun stream toma datos del mapa SLAM en ovo/entities/ovomapping.py:292.

En cambio, image y binary_maps viven en el flujo semántico interno de OVO (ovo/entities/ovo.py:212, ovo/entities/ovo.py:400) y no cruzan el contrato de stream actual (ovo/entities/visualizers/rerun_contracts.py:24).
