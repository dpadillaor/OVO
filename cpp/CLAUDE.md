# OVO C++ (`cpp/`)

Port de OVO a C++. Objetivos: mejor arquitectura, más rapidez (threads) y ejercicio de C++.
Primera meta: reproducir el mapeo geométrico de `VanillaMapper` (poses GT + desproyección de profundidad).

## Visión (condiciona la arquitectura desde ya)
- La semántica corre en un **thread paralelo a ORB-SLAM3**; a largo plazo será un módulo de un **SLAM propio**.
  El SLAM debe ser intercambiable sin tocar la semántica.
- **Geometría y semántica son dueñas de sus datos** y se comunican por avisos (eventos): la geometría
  publica puntos nuevos/movidos/borrados; la semántica es dueña de instancias y etiquetas punto→instancia.
- **Submapas semánticos** (como el Atlas de ORB-SLAM3): un **detector de drift** decidirá abrir un
  submapa semántico nuevo; más tarde los submapas se fusionan (fusión de instancias entre submapas).
- El detector de drift es **dominio** (decide). La telemetría solo **observa** (puede perder datos).

**Modo de trabajo:** David escribe el código de dominio; Claude pregunta, guía, explica y revisa.
Claude no genera código ni estructura sin que se le pida explícitamente (a veces David pide "hazlo tú":
tests, esqueletos, refactors). Claude redacta los `.puml` y mantiene el CMake.
Explicar **un concepto cada vez**, con ejemplos y analogías con Python; si David se pierde, parar y simplificar.

**Metodología de pair programming (cada pieza nueva).** Antes de escribir código, Claude presenta una **ficha** corta
y nada más (sin ensayos, sin alternativas salvo que haya una decisión real):
1. **Fichero:** ruta y para qué sirve (una frase).
2. **Qué contiene:** tipos y funciones, cada una con **inputs → output** y qué hace (una línea).
3. **Estado interno:** miembros / variables internas y por qué.
4. **Decisiones abiertas:** solo las que bloquean, con recomendación.
5. **Tareas:** lista numerada de pasos pequeños (quién escribe cada uno: David o Claude), incluidos tests.
Luego se avanza tarea a tarea: David escribe (o pide "hazlo tú"), Claude revisa y compila.

**Estado y próximos pasos:** ver `docs/progress.md` (leer al empezar cada sesión).

## Arquitectura
Hexagonal (ports & adapters) + SOLID.
- Dominio en el centro: lógica pura, sin I/O, sin GPU, sin SLAM, sin dataset.
- Puertos = interfaces definidas por lo que el dominio necesita.
- Adaptadores = implementaciones concretas (GT, ORB-SLAM3, PLY, mocks).
- Las dependencias apuntan hacia dentro: el dominio nunca incluye un adaptador.

## Prioridades: correcto > rápido > eficiente en memoria
Regla: primero correcto, luego medir (Tracy, benchmarks), luego optimizar lo que diga el profiler.
- **Seguro:** un solo escritor por dato; datos compartidos entre threads = snapshots inmutables
  (`shared_ptr<const T>`); asserts en Debug, sin coste en Release.
- **Rápido:** puntos en struct-of-arrays contiguos; etiquetas como `vector` indexado por `PointId`;
  templates/concepts en el camino caliente, `virtual` solo en fronteras (SLAM, I/O); `std::span` y move, sin copias.
- **Preparado para GPU (sin escribir GPU aún):** (1) datos en arrays contiguos por campo (SoA), subir a GPU = una copia;
  (2) operaciones caras con interfaz por lotes (bloques de puntos), implementables en CPU o CUDA;
  (3) matemática de un punto pura, header-only, sin estado ni memoria dinámica (ladrillo reutilizable en kernels);
  (4) nada de reservas, virtuales ni excepciones dentro del bucle caliente. El bucle por lotes llama al ladrillo y el compilador vectoriza (SIMD).
- **Contenedores:** no copiar las estructuras de Python. Se eligen caso a caso según el patrón de acceso
  (ids densos → `vector` indexado; `std::map` no es la opción por defecto). Decidir midiendo.
- **Eficiente:** bloques de puntos por keyframe (un LC solo copia lo que se mueve); tipos ajustados
  (`float` xyz, `uint8` color).
- La primera versión CPU puede ser más lenta que Python (GPU) en frustum/proyección: es esperado.

## Estructura de `core/` por niveles
Un nivel solo usa los de abajo, nunca los de arriba. Carpeta = namespace (`common/` vive en `ovo::core`).
1. `common/`: ladrillos básicos (`StrongId`). No depende de nada.
2. `geometry/`: matemática pura sin estado (cámara, pose, frustum, matcher). `ovo::core::geometry`.
3. `mapping/` y `semantics/` (futuro): el sistema, con estado. `semantics` usa `mapping`, nunca al revés.
Los tests replican la estructura (`tests/unit/core/<nivel>/test_<fichero>.cpp`).
Cuando llegue `semantics/`, valorar una librería CMake por nivel para que el compilador imponga las fronteras.

## Modelo de dominio
Ver `docs/domain-model.md` (entidades, valores, dueños, link por `PointId`).

## Documentación de arquitectura (`cpp/docs/`)
Excepción a la regla del raíz: los diagramas del port C++ viven en `cpp/docs/`, no en `tfm/diagrams/`.
Lo maduro se copia a `tfm/` como material de tesis.
- Diagramas PlantUML por niveles C4 (contexto, contenedores, componentes) + dinámicos; versiones as-is (Python) y to-be (C++). No se renderizan.
- ADRs: un markdown por decisión (contexto, opciones, decisión, por qué).

## Lenguaje
- Estándar: C++23 (g++ 13 / MinGW en Windows; Linux más adelante).
- Sin `new`/`delete` a mano ni punteros crudos que posean memoria.
  Orden de preferencia: valor > `unique_ptr` > `shared_ptr` (solo propiedad compartida real).
  Puntero crudo o referencia = observador sin propiedad.
- RAII, rule of zero, const-correctness, `[[nodiscard]]`, `noexcept` en lo que no puede lanzar.
- Tipos fuertes solo donde el error es fácil y silencioso (ids, dirección de la pose). Puntos/vectores: Eigen
  a pelo, con el sistema de referencia en el nombre (`pointCam`, `pointWorld`).
- `float` para geometría (precisión suficiente, mitad de memoria, GPU). Eigen por `const&`. Nunca dejar un
  vector de Eigen sin inicializar (no se inicializa a cero).
- **Header vs `.cpp`:** en el header lo obligatorio (templates, `constexpr`) y las funciones pequeñas del camino
  caliente (inlining + SIMD + CUDA). En `.cpp` las grandes, las no calientes y las que necesitan includes pesados.
- Píxeles: `(u, v)` = (columna, fila); el centro del píxel está en la coordenada entera (como el `meshgrid` del
  Python al desproyectar). Coordenada → píxel = redondear al más cercano (`std::lround`, con signo), luego límites.
- Ids con tipo fuerte (`FrameId`, `PointId`, `InstanceId`), no `int`.
- Inyección de dependencias por constructor; sin singletons ni estado global.

## Errores (tres niveles)
| Situación | Ejemplo | Herramienta |
|---|---|---|
| Fallo esperable del mundo real | pose NaN del SLAM, frame sin profundidad | `std::expected` (opcional), con `[[nodiscard]]` |
| Dato inválido al arrancar | cámara con `fx = 0` | excepción (validar en el constructor: "valida en la frontera, confía dentro") |
| Bug del programador (precondición) | `project` con `z <= 0` | `assert` + comentario de precondición; death test en `...DeathTest` |
- Nunca `*r` sobre un `expected` sin comprobar antes.
- Validar floats con comparaciones "en positivo" (`x > 0 && x < inf`) para rechazar NaN.

## Estilo y build
- **Código en inglés:** comentarios, mensajes de error y strings del código (C++ y CMake). La documentación (`docs/`, este fichero) sigue en español.
- Naming: `PascalCase` para tipos, `camelCase` para funciones/métodos/variables, miembros privados con sufijo `_`.
- Build: CMake + Ninja (toolchain local: MinGW g++ 13.2, CMake 3.29). Claude escribe y mantiene el CMake; David lo revisa.
- Tests: GoogleTest + GoogleMock (mocks de puertos). Una suite por clase y operación (`PinholeCamera`, `PinholeCameraProject`...); death tests en suites `...DeathTest`. **TDD:** test primero (rojo → verde → refactor).
  Incluir casos límite (0, negativos, NaN, ±inf). Un test rompe una sola cosa.
- Tipos de test, cada uno con su ejecutable: `tests/unit/` (rápidos, sin datos, siempre),
  `tests/integration/` (adaptadores reales, datos), `tests/regression/` (C++ vs salida de referencia de Python);
  `benchmarks/` aparte (rendimiento).

## Pendiente de decidir
- Pertenencia punto↔instancia: única fuente de verdad (punto→instancia o instancia→puntos).
- Polimorfismo de puertos: `virtual` vs `concepts` (tendencia: `virtual` en fronteras, `concepts` dentro).
- Descomposición de `VanillaMapper` en dominio / puertos / adaptadores.
- Telemetría: puerto de observabilidad (eventos, métricas, trazas) con adaptadores (JSONL, Rerun, wandb, Tracy).
- Gestión de errores detallada (errores también como eventos de telemetría).

## Herramientas
- ASan/TSan no funcionan con MinGW: los bugs de concurrencia se validan en Linux.
