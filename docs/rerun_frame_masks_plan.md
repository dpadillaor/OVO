# Plan de Integracion Rerun: Frame + Mascaras (sin raw SAM)

## Objetivo
Visualizar en Rerun, ademas del mapa 3D actual, el frame RGB y las mascaras de segmentacion utiles para debugging:
- Mascara post-SAM filtrada (seg_map).
- Mascara post-tracking/asociacion a instancia (ins_map opcional).

Queda explicitamente fuera de alcance en esta etapa:
- Mostrar proposals crudas de SAM (raw SAM).

## Estado actual (resumen)
- El stream a Rerun hoy envia solo datos 3D (`points`, `obj_ids`, `colors`, `c2w`).
- `image` y `binary_maps/seg_map` viven en el flujo semantico de OVO y no salen por la cola del visualizador.
- Hay una sola cola de visualizacion (`mpqueue`) y un renderer de stream dedicado.

## Decision de arquitectura
Usar una sola cola IPC al visualizador, con un mensaje unificado de stream por `frame_id`.

Razon:
- Evita desincronizacion de dos colas paralelas.
- Mantiene renderer simple (renderiza, no reconcilia fuentes).
- Permite timeline consistente por frame.

## Que se mostrara exactamente
1. RGB del frame (opcional por config).
2. `seg_map` post-SAM filtrado (opcional por config).
3. `ins_map` post-tracking (opcional por config, fase 2).

Nota:
- En fase 1 no se enviaran `binary_maps` completas para evitar payloads grandes.

## Contrato de mensaje propuesto
Nuevo mensaje de stream unificado (nombre orientativo: `StreamFrameMessage`):

- `type`: `"stream_frame"`
- `frame_id`: `int`
- `points`: `np.ndarray` (N,3) `float16/float32`
- `obj_ids`: `np.ndarray` (N,...) `int16/int32`
- `c2w`: `np.ndarray` (4,4) `float16/float32`
- `colors`: `Any` (mantener compatibilidad actual)
- `rgb`: `np.ndarray | None` (H,W,3) `uint8`
- `seg_map`: `np.ndarray | None` (H,W) `uint16/int32`
- `ins_map`: `np.ndarray | None` (H,W) `int32` (fase 2)

## Plan por fases

### Fase 1: Integracion minima (RGB + seg_map)
Objetivo: ver imagen y segmentacion filtrada sincronizadas con 3D en Rerun.

Cambios:
1. `ovo/entities/ovo.py`
- Exponer un snapshot visual ligero por frame segmentado:
  - `frame_id`
  - `rgb` (CPU `uint8`)
  - `seg_map` (CPU `uint16/int32`)
- API sugerida:
  - `get_last_visual_snapshot()`

2. `ovo/entities/ovomapping.py`
- Tras `detect_and_track_objects(...)`, tomar snapshot desde OVO.
- Construir mensaje unificado con la parte SLAM + parte visual.
- Enviar por `mpqueue` usando `_queue_put_dropping(...)`.

3. `ovo/entities/visualizers/rerun_contracts.py`
- Definir TypedDict y validador para `stream_frame`.
- Mantener compatibilidad temporal con contrato antiguo.

4. `ovo/entities/visualizers/rerun_handlers.py`
- Extender `StreamRenderer.handle_message(...)` para:
  - interpretar nuevo contrato.
  - loggear RGB y `seg_map` en rutas separadas.
  - mantener logs 3D actuales sin cambios funcionales.

Criterio de salida:
- Rerun muestra `world/points` + `world/camera` + `frame/rgb` + `frame/seg_map` alineados por timeline.

### Fase 2: Capa de instancias 2D (ins_map)
Objetivo: depurar asociaciones 2D->3D.

Cambios:
1. Exponer `ins_map` cuando este disponible.
2. Incluirlo en mensaje unificado como campo opcional.
3. Renderizar capa separada `frame/ins_map`.

Criterio de salida:
- Puede alternarse entre `seg_map` e `ins_map` en viewer.

### Fase 3: Robustez y rendimiento
Objetivo: no degradar FPS ni estabilidad.

Cambios:
1. Flags de config para habilitar/deshabilitar cada capa:
- `vis.stream_rgb`
- `vis.stream_seg_map`
- `vis.stream_ins_map`

2. Control de payload:
- mantener `seg_map` compacta.
- sin `binary_maps` por defecto.

3. Politica de degradacion:
- si cola esta saturada, priorizar ultimo frame sobre historico (ya existe drop-oldest).

Criterio de salida:
- overhead acotado y sin bloqueos del loop principal.

## Archivos impactados
- `ovo/entities/ovo.py`
- `ovo/entities/ovomapping.py`
- `ovo/entities/visualizers/rerun_contracts.py`
- `ovo/entities/visualizers/rerun_handlers.py`
- (opcional) tests en `tests/unit/`

## Estrategia de testing
1. Unit tests contrato:
- valida `stream_frame` con y sin campos opcionales.
- rechaza payloads invalidos.

2. Unit tests renderer:
- con `rgb` y `seg_map` presentes.
- con `rgb/seg_map` ausentes (compatibilidad).

3. Smoke test manual:
- ejecutar escena corta con streaming activado.
- verificar alineacion temporal entre 3D y 2D.

## Riesgos y mitigaciones
1. Desfase temporal entre datos 3D y 2D.
- Mitigacion: indexar por `frame_id`, loggear mismo step.

2. Overhead por copia CPU.
- Mitigacion: enviar solo `seg_map` compacta; `binary_maps` fuera de alcance.

3. Compatibilidad con payload viejo.
- Mitigacion: mantener parser dual durante transicion.

## Definicion de terminado (DoD)
- Se visualiza RGB + seg_map junto al stream 3D.
- No se usa raw SAM.
- El stream no rompe modos `fusion` ni `loop_closure`.
- Tests de contrato y renderer pasan.
