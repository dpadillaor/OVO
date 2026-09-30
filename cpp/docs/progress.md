# Progreso del port C++

Última sesión: 2026-09-30 (rama `feat/cpp-core`).

## Hecho
| Pieza | Fichero | Tests |
|---|---|---|
| Build: CMake + Ninja, C++23, warnings estrictos + `-Werror`, GoogleTest, Eigen 3.4 (FetchContent) | `CMakeLists.txt`, `cmake/` | — |
| `StrongId<Tag>` → `FrameId`, `PointId`, `InstanceId` | `core/include/ovo/core/common/strong_id.hpp` | 4 + 4 `static_assert` |
| `PinholeCamera`: validación, `project`, `unproject` | `core/include/ovo/core/geometry/pinhole_camera.hpp` | 19 (incl. 2 death tests) |
| `Pose` (c2w, `R` + `t`): constructor privado + `fromCamToWorld` / `identity`, `camToWorld`, `worldToCam` | `core/include/ovo/core/geometry/pose.hpp` | 10 (comprobados con mutaciones) |
| Diseño: contexto, as-is, to-be, ADR-0001/0002, modelo de dominio | `docs/` | — |

## Siguiente (en orden)
1. **Terminar `Pose`** (la lista de tests pendientes está al final de `tests/unit/core/geometry/test_pose.cpp`):
   - `center()`: devuelve `translation_`. Tests `PoseCenter`.
   - `assert` en `fromCamToWorld`: `R` ortonormal (`RᵀR ≈ I`), `det(R) ≈ +1`, todo finito. Death tests `PoseDeathTest`.
   - Más adelante, cuando llegue el loop closure: componer, pose relativa (`new * old⁻¹`), transformar por lotes.
2. **`Frustum`** (`geometry/frustum.hpp` + `.cpp`): 8 esquinas (profundidad mín/máx del frame) → AABB (fase rápida)
   + 6 planos (fase precisa) → qué puntos ve la cámara. Interfaz por lotes.
3. **`PointPixelMatcher`**: proyecta, lee profundidad, empareja si |Δz| < 3 cm. Única pieza con interfaz
   (CPU hoy, CUDA mañana). Necesita decidir antes el tipo de imagen.
4. Con 1-3: `VanillaMapper` en C++ (`mapping/`), comparado con la salida de referencia del Python.

## Decisiones pendientes (cuando toque)
- Tipo de imagen (profundidad, RGB): `Image<T>` propio o `cv::Mat`. Lo necesita el Matcher.
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
- Posibles bugs del Python detectados (sin verificar; chips abiertos en la sesión): limpieza de keyframes borrados
  en `ovo.py`, y `slam_module` desconocido que cae en silencio a poses GT.

## Diario de la tesis
El submódulo `tfm/` no está inicializado en el Windows local, así que la entrada del diario de esta sesión
no se ha escrito en `tfm/diario/2026-09-30.md`. El resumen de decisiones está en este fichero, en los ADRs y en
`CLAUDE.md`; pasarlo al diario cuando `tfm/` esté disponible.
