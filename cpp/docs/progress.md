# Progreso del port C++

Última sesión: 2026-10-01 (rama `feat/cpp-core`).

## Hecho
| Pieza | Fichero | Tests |
|---|---|---|
| Build: CMake + Ninja, C++23, warnings estrictos + `-Werror`, GoogleTest, Eigen 3.4 (FetchContent) | `CMakeLists.txt`, `cmake/` | sin tests |
| `StrongId<Tag>` → `FrameId`, `PointId`, `InstanceId` | `core/include/ovo/core/common/strong_id.hpp` | 4 + 4 `static_assert` |
| `PinholeCamera`: validación, `project`, `unproject` | `core/include/ovo/core/geometry/pinhole_camera.hpp` | 19 (incl. 2 death tests) |
| `Pose` (c2w, `R` + `t`): constructor privado + `fromCamToWorld` / `identity`, `camToWorld`, `worldToCam`, `center`; `assert` en el constructor (ortonormal, `det = +1`, finito) | `core/include/ovo/core/geometry/pose.hpp` | 15 (incl. 3 death tests; comprobados con mutaciones) |
| `Frustum`: 6 planos `(n, d)` con normales orientadas hacia dentro con el centroide (no depende del orden de las esquinas); constructor en `.cpp` (una vez por frame), `contains` en header (una vez por punto); `assert` `0 < minDepth < maxDepth`. Primer `.cpp`: `ovo_core` pasa a `STATIC` | `core/include/ovo/core/geometry/frustum.hpp`, `core/src/geometry/frustum.cpp` | 16 (incl. 3 death tests; comprobados con mutaciones) |
| Diseño: contexto, as-is, to-be, ADR-0001/0002/0003, modelo de dominio | `docs/` | sin tests |

## Siguiente (en orden)
1. **`Image<T>`** (`common/image.hpp`, ADR-0003): dueño de un `std::vector<T>`, `width()`, `height()`,
   `at(u, v)`, `data()`; memoria como numpy (fila a fila, HWC). Nada más.
2. **`PointPixelMatcher`**: proyecta, lee profundidad (`Image<float>`, metros), empareja si |Δz| < 3 cm.
   Única pieza con interfaz (CPU hoy, CUDA mañana). Ignora profundidad 0 (sin medida).
3. Con 1-2: `VanillaMapper` en C++ (`mapping/`), comparado con la salida de referencia del Python.

## Decisiones pendientes (cuando toque)
- `Frustum`: AABB (fase rápida) cuando exista el mapa por bloques (caja del frustum contra caja del bloque),
  o si un benchmark lo pide; interfaz por lotes cuando se decida el layout del mapa (SoA).
- `Pose`, para el loop closure: componer, pose relativa (`new * old⁻¹`), transformar por lotes.
- Paso de coordenada a píxel en el Matcher: el Python redondea (`round`); decidir si lo copiamos.
- Pertenencia punto↔instancia: una sola fuente de verdad.
- `virtual` vs `concepts` en puertos (tendencia: `virtual` en fronteras, `concepts` dentro).
- Canal de eventos geometría→semántica (colas, orden, backpressure) y cómo lee la semántica la geometría.
- Residencia del mapa en GPU y quién lo lee en CPU (ADR-0002).
- ADRs de telemetría y de gestión de errores.
- Salida de referencia del Python para los tests de regresión (volcar el mapa de una escena).

## Notas del análisis del Python (para el adaptador de ORB-SLAM3)
- El binding devuelve poses ya en **Twc (c2w)**, 12 valores fila a fila; OVO añade `[0,0,0,1]` y multiplica a la
  izquierda por `world_ref` (pose GT del frame 0) para alinear con el mundo del GT. `float32`, sin volteos.
- Intrínsecos duplicados a mano (YAML de ORB y dataset; ScanNet con desfase `crop_edge = 12`): en C++ una sola fuente.
- El Atlas (multimapa) de ORB-SLAM3 no se trata: relevante para los submapas semánticos.
- **Bug en los planos del frustum** (`ovo/submodules/gaussian_slam/utils/mapper_utils.py`,
  `compute_camera_frustum_planes`), verificado a mano con la cámara de test (pose identidad, profundidad [1, 5]):
  `D` usa `corners[i]` con `i` = índice del plano, así que el plano "far" pasa por una esquina near y duplica al near;
  los planos "top"/"bottom" se construyen con esquinas de los lados (normales sin componente y). Resultado: no hay
  límite vertical (`(0, ±2, 3)` pasa) y el límite far es accidental (`(0, 0, 6)` pasa, `(0, 0, 100)` no).
  Impacto: casi seguro solo rendimiento, porque `match_3d_points_to_2d_pixels` vuelve a filtrar por imagen y por
  |Δz|. En C++ el frustum es correcto (normales orientadas con un punto interior), así que la salida del frustum
  diferirá del Python: los tests de regresión deben comparar el mapa final, no el paso intermedio.
- Posibles bugs del Python detectados (sin verificar; chips abiertos en la sesión): limpieza de keyframes borrados
  en `ovo.py`, y `slam_module` desconocido que cae en silencio a poses GT.

## Diario de la tesis
El submódulo `tfm/` no está inicializado en el Windows local, así que la entrada del diario de esta sesión
no se ha escrito en `tfm/diario/2026-09-30.md`. El resumen de decisiones está en este fichero, en los ADRs y en
`CLAUDE.md`; pasarlo al diario cuando `tfm/` esté disponible.
