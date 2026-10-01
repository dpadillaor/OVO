# Progreso del port C++

Última sesión: 2026-10-01 (rama `feat/cpp-core`).

## Hecho
| Pieza | Fichero | Tests |
|---|---|---|
| Build: CMake + Ninja, C++23, warnings estrictos + `-Werror`, GoogleTest, Eigen 3.4 (FetchContent) | `CMakeLists.txt`, `cmake/` | sin tests |
| `Makefile`: atajos sobre CMake + CTest (`gmake`, `gmake test T=^Suite`; en Linux `make`) | `Makefile` | sin tests |
| `StrongId<Tag>` → `FrameId`, `PointId`, `InstanceId` | `core/include/ovo/core/common/strong_id.hpp` | 4 + 4 `static_assert` |
| `PinholeCamera`: validación, `project`, `unproject` | `core/include/ovo/core/geometry/pinhole_camera.hpp` | 19 (incl. 2 death tests) |
| `Pose` (c2w, `R` + `t`): constructor privado + `fromCamToWorld` / `identity`, `camToWorld`, `worldToCam`, `center`; `assert` en el constructor (ortonormal, `det = +1`, finito) | `core/include/ovo/core/geometry/pose.hpp` | 15 (incl. 3 death tests; comprobados con mutaciones) |
| `Frustum`: 6 planos `(n, d)` con normales orientadas hacia dentro con el centroide (no depende del orden de las esquinas); constructor en `.cpp` (una vez por frame), `contains` en header (una vez por punto); `assert` `0 < minDepth < maxDepth`. Primer `.cpp`: `ovo_core` pasa a `STATIC` | `core/include/ovo/core/geometry/frustum.hpp`, `core/src/geometry/frustum.cpp` | 16 (incl. 3 death tests; comprobados con mutaciones) |
| `Image<T>` (ADR-0003): dueño de un `std::vector<T>`, `width()`, `height()`, `at(u, v)` (versión que escribe y versión `const`), `data()` como puntero (se pueden cambiar los píxeles, no el tamaño); memoria fila a fila como numpy; `assert` en `at` y en tamaño 0 | `core/include/ovo/core/common/image.hpp` | 11 (incl. 3 death tests) |
| `PointPixelMatcher`: sin estado; config (cámara por copia, tolerancia) en el constructor, que lanza si la tolerancia no es finita y > 0; `match(span de puntos, pose, profundidad)` devuelve `vector<PointPixelMatch>` `{pointIndex, u, v}` (índice en la entrada, no `PointId`). Redondeo al píxel más cercano con signo, luego límites, luego profundidad > 0 y `\|Δz\| < tolerancia`. `assert` si la profundidad no tiene el tamaño de la cámara | `core/include/ovo/core/geometry/point_pixel_matcher.hpp`, `core/src/geometry/point_pixel_matcher.cpp` | 19 (incl. 1 death test; comprobados con mutaciones) |
| Diseño: contexto, as-is, to-be, ADR-0001/0002/0003, modelo de dominio | `docs/` | sin tests |

## Siguiente (en orden)
1. **`VanillaMapper`** en C++ (`mapping/`): frustum → matcher → cobertura (marcar píxeles emparejados,
   dilatar 3x3, submuestrear a la mitad) → desproyectar los píxeles sin cubrir. Comparado con la salida de
   referencia del Python. Antes: decidir el layout del mapa de puntos (SoA) y quién es dueño de qué.

## Decisiones pendientes (cuando toque)
- `Frustum`: AABB (fase rápida) cuando exista el mapa por bloques (caja del frustum contra caja del bloque),
  o si un benchmark lo pide; interfaz por lotes cuando se decida el layout del mapa (SoA).
- `Pose`, para el loop closure: componer, pose relativa (`new * old⁻¹`), transformar por lotes.
- `PointPixelMatcher`: sacar el cuerpo del bucle a un ladrillo por punto en el header (`matchPoint`) cuando
  llegue la versión CUDA, para que CPU y kernel compartan la lógica. Hoy no: el bucle se lee bien entero.
- `PointPixelMatcher` con puntos detrás de la cámara (z ≤ 0): hoy no se tratan (llegan filtrados por el
  frustum; el `assert` de `project` los caza en Debug). Decidir si se ignoran o es precondición si se usa suelto.
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
- **Bug de configuración en `VanillaMapper`** (`vanilla_mapper.py:32`): lee `config["mapping"]["downscale_res"]`,
  pero los yaml (`data/working/configs/slam/vanilla/*.yaml`) escriben `downscale_ratio`. Siempre usa el valor por
  defecto, 2: ScanNet++ pide 1 y corre con 2 (4 veces menos puntos nuevos). Ojo al comparar con resultados publicados.
- `VanillaMapper` (densidad, configurable en C++): dilatación `k_pooling` (3, 5 en ScanNet++), hecha como
  `~maxpool(~mask)` = cada punto emparejado bloquea su vecindario; submuestreo `[::2, ::2]` solo al crear puntos
  (matcher y dilatación a resolución completa), conservando las coordenadas originales del píxel.
  `max_frame_points` se lee y no se usa. Frame sin profundidad válida: `min()` de vacío revienta.
- `match_3d_points_to_2d_pixels` comenta que descartar profundidad 0 "se podría saltar": falso. Un punto a
  menos de `th_dist` de la cámara emparejaría con un píxel sin medida (test `IgnoresPixelsWithoutDepth`).
- Posibles bugs del Python detectados (sin verificar; chips abiertos en la sesión): limpieza de keyframes borrados
  en `ovo.py`, y `slam_module` desconocido que cae en silencio a poses GT.

## Diario de la tesis
El submódulo `tfm/` no está inicializado en el Windows local, así que la entrada del diario de esta sesión
no se ha escrito en `tfm/diario/2026-09-30.md`. El resumen de decisiones está en este fichero, en los ADRs y en
`CLAUDE.md`; pasarlo al diario cuando `tfm/` esté disponible.
