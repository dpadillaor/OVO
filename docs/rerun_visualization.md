# Visualización con Rerun — OVO

**Última actualización:** 2026-07-12
**Versión Rerun:** 0.29.2

Rerun corre en un **proceso aparte** que consume una cola multiproceso alimentada por
`OVOSemMap`. Hay dos modos: `stream` (el mapa en vivo) y `tracking` (diagnóstico del
tracking de máscaras). Los modos antiguos `fusion` y `loop_closure` se eliminaron.

## Configuración

```yaml
vis:
  stream: true                # activa el envío de datos al visualizador
  type: "rerun"               # "rerun" | "open3d"
  rerun_mode: "stream"        # "stream" | "tracking"
  rerun_visual_mode: "off"    # "off" | "spawn" | "serve"
  save_rrd: true              # escribe el .rrd en la carpeta de la escena
```

| Parámetro | Valores | Qué hace |
|---|---|---|
| `stream` | bool | Sin esto no se envía nada |
| `type` | `rerun` / `open3d` | Selecciona el visualizador (`visualizers/selection.py`) |
| `rerun_mode` | `stream` / `tracking` | Qué renderer se lanza. **Cualquier otro valor desactiva el envío**: `_send_stream_frame` filtra por él y no manda ni un frame |
| `rerun_visual_mode` | `off` / `spawn` / `serve` | `spawn` abre un viewer local; `serve` levanta gRPC en el puerto **9877** (`rerun connect grpc://localhost:9877`); `off` no muestra nada en vivo |
| `save_rrd` | bool | Independiente de lo anterior. Escribe `stream.rrd` o `tracking.rrd` en `data/output/<dataset>/<exp>/<scene>/` |
| `show_stream` | bool | Legacy; equivale a `rerun_visual_mode: spawn`. Preferir el nuevo |

## Arquitectura

```
rerun_contracts.py   Mensajes de la cola (stream_frame, update_map, jump_event) + TrackSignals
      |
rerun_frame.py       decode(mensaje) -> Frame: float32, corte de techo, instance ids resueltos.
      |              Único sitio que conoce el layout crudo del mensaje.
      |
rerun_sink.py        RerunSink = UN recording + SU presupuesto de puntos + su política de
      |              `static`. Los renderers escriben a una lista de sinks (live y/o file);
      |              cada sink submuestrea según su propio límite (80k live, 300k file).
      |
rerun_scene.py       MapScene: lo que ambos modos enseñan bajo `world/` — nube de instancias
rerun_tracking.py    (una entidad por instancia), cámara, trayectoria, normales, eventos.
      |              TrackingPainter / SignalsPainter: capas y señales del modo tracking.
      |
rerun_handlers.py    BaseRerunRenderer  -> ciclo de vida de rerun + despacho de mensajes
      |              SceneRenderer      -> + MapScene y sus eventos
      |              StreamRenderer     -> paneles RGB/máscaras   } hermanos, sin
      |              TrackingRenderer   -> capas de tracking      } herencia cruzada
      |
rerun_orchestrator.py  Bucle sobre la cola, manejo de errores, sentinel de cierre.
```

Los renderers **nunca** llaman a `rr.log` con un `recording` concreto: hablan con sus sinks.
Añadir un destino (otro presupuesto, otro fichero) es añadir un `RerunSink`.

## Flujo de datos

```
Dataset frame (image, depth, pose estimada)
  |
  v
OVOSemMap._run_semantic_step            ovo/entities/ovomapping.py
  |
  +--> OVO.detect_and_track_objects     ovo/entities/ovo.py
  |      +--> _get_masks (SAM)
  |      +--> _match_and_track_instances -> frustum + reproyección + match por profundidad
  |             +--> _track_objects      -> robos, nacimientos, y (modo tracking)
  |                                         la partición assigned/unassigned
  |      +--> _last_visual_snapshot: rgb, sam_map, ins_map, assigned_ins_map, kf_id
  |
  +--> _send_stream_frame -> mensaje "stream_frame" a la cola
         (points, obj_ids, normals, c2w, + snapshot visual,
          + point_ids y track_signals SOLO en modo tracking)
  |
  v
mpqueue (IPC; put no bloqueante que descarta si está llena)
  |
  v
Proceso visualizador -> QueueRendererOrchestrator -> Renderer.handle_message
```

`_send_stream_frame` se llama **3 veces por paso semántico**: una tras el tracking y dos más
en `update_map` (antes y después de la fusión), con `frame_id` distinto. Por eso las señales de
tracking llevan su propio `frame_id`: el painter sólo las pinta si coincide con el del frame, y
si no, limpia las capas. Sin eso reaparecerían los highlights del KF anterior.

## Modo `stream` (`stream.rrd`)

- **Instances3D** — `world/**`: una entidad por instancia (`world/instances/obj_<id>`,
  toggleables), fondo (`world/background`), cámara + frustum, trayectoria cyan.
  Puntos amarillos = eventos `update_map`; rojos = saltos de pose (jump drift).
  `world/normals/obj_<id>`: normales por instancia, estáticas, **ocultas** por defecto.
- **Panel 2D**: `frame/rgb`, `frame/sam_map` (máscaras SAM), `frame/assigned_map` (todas las
  instancias asignadas — lo que cuenta la co-ocurrencia), `frame/ins_map` (supervivientes
  top-KF, lo que alimenta a CLIP).
- `frame/kf_id`: TextLog para mapear índices de KF a steps del rrd.

Filtro de techo (`ceiling_mask`): se descartan los puntos a menos de 0.2 m del z máximo.

## Modo `tracking` (`tracking.rrd`)

Diagnóstico del tracking de máscaras. Contesta: *de los puntos que la cámara ve y que
sobreviven al match por profundidad, ¿cuáles ya tenían dueño y cuáles no?*

**Vista 3D "Tracking KF"** y **vista 2D "Tracking KF (2D)"** — mismos conjuntos, mismos colores,
sincronizados en el timeline `step` (== `frame_id`):

| Capa | Color | 3D | 2D | Por defecto |
|---|---|---|---|---|
| Mapa completo | gris | `tracking/greymap` | — | visible |
| **assigned** — ya tenían `ins_id` al entrar al KF | verde | `tracking/assigned` | `frame/pts2d/assigned` | visible |
| **unassigned** — sin dueño (`ins_id == -1`) | amarillo | `tracking/unassigned` | `frame/pts2d/unassigned` | visible |
| robbed — cambian de dueño (⊂ assigned) | rojo | `tracking/robbed` | `frame/pts2d/robbed` | **oculta** |
| births — siembran una instancia nueva (⊂ unassigned) | cyan | `tracking/births` | `frame/pts2d/births` | **oculta** |

assigned + unassigned = `n_matched`. Lo que queda gris está fuera del frustum o murió en el test
de profundidad (oclusión, drift de pose, `depth == 0`) — esa mortandad **no** se visualiza hoy.

La vista 2D lleva el RGB de fondo y tres overlays de máscara **ocultos** por defecto
(`frame/seg/{sam,ins,assigned}`, `SegmentationImage` con `opacity=0.45`): sirven para ver sobre
qué máscara SAM aterrizan los puntos amarillos.

**Series temporales** (columna derecha) — 4 vistas:

| Vista | Path | Contenido | Unidades |
|---|---|---|---|
| **Points per KF** | `signals/count/*` | Una curva por capa 3D, **con el color de la capa**: `assigned`, `unassigned`, `robbed`, `births` | puntos |
| **Points Δ/frame** | `signals/deriv/*` | Derivada cruda de cada una (`d_assigned`, …), mismos colores | puntos/frame |
| **Camera speed** | `signals/cam/*` | `v_lin`, `v_ang` | m/frame, deg/frame |
| **Camera jerk** | `signals/accel/*` | `a_lin`, `a_ang` — el tirón. Un pico aquí mueve el mapa más de lo que tolera `match_distance_th`: el match se rompe, un objeto deja de reconocerse y **nace un duplicado**. Un pico de `a_ang` justo antes de uno de `births` es la firma del fallo | m/frame², deg/frame² |

Los contadores se derivan **del tamaño de las propias capas**, no de tallies paralelos: si un punto
está pintado, está contado. La gráfica no puede desincronizarse del 3D.

Rerun no permite etiquetar los ejes (`rrb.ScalarAxis` sólo tiene `range`/`zoom_lock`), así que las
unidades de cámara van en el **nombre de la serie**, o sea en la leyenda: `v_lin [m/frame]`,
`a_ang [deg/frame²]`. Los paths de las entidades no cambian.

Las derivadas son crudas (sin suavizar, para que un pico de un solo KF sobreviva) y dividen por el
Δframe **real**: sólo emiten los KFs, así que el gap es `segment_every`, no 1. Un `d_robbed` de 50
significa "500 robos más que el KF anterior".

`n_orphans` y el conteo de *instancias* nacidas se siguen registrando en el logger (telemetría
Tier-2) pero **no** se grafican: el primero no tiene capa 3D que lo localice, y el segundo cuenta
instancias mientras que la capa cyan pinta puntos.

Las capas y las señales sólo existen en los KFs de segmentación (`segment_every`); en el resto de
frames se limpian. Es lo esperado.

Los datos salen de `OVO._track_objects`, capturados **antes** de que el bucle de máscaras mute
`points_ins_ids` (si no, la partición ya estaría contaminada). Sólo se calculan si
`rerun_mode == tracking` (`track_viz_enabled`): coste cero en el resto de runs.

## Abrir una grabación

```bash
rerun data/output/Replica/<experiment>/<scene>/stream.rrd
rerun data/output/Replica/<experiment>/<scene>/tracking.rrd
```
