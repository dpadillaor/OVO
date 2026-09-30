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

**Modo de trabajo:** el código lo escribe David. Claude pregunta, guía y revisa;
no genera código ni estructura sin que se le pida explícitamente.
Excepción: Claude redacta los diagramas `.puml` a partir de lo que David decide; David los visualiza y corrige.

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
Los tests replican la estructura (`tests/core/<nivel>/test_<fichero>.cpp`).
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
- RAII, rule of zero, const-correctness, `[[nodiscard]]`.
- Ids con tipo fuerte (`FrameId`, `PointId`, `InstanceId`), no `int`.
- Inyección de dependencias por constructor; sin singletons ni estado global.

## Errores
- `std::expected` opcional para fallos esperables (pose NaN, frame vacío); siempre con `[[nodiscard]]`.
- Excepciones para errores fatales (config inválida al arrancar).
- Nunca `*r` sobre un `expected` sin comprobar antes.

## Estilo y build
- Naming: `PascalCase` para tipos, `camelCase` para funciones/métodos/variables, miembros privados con sufijo `_`.
- Build: CMake + Ninja (toolchain local: MinGW g++ 13.2, CMake 3.29). Claude escribe y mantiene el CMake; David lo revisa.
- Tests: GoogleTest + GoogleMock (mocks de puertos).

## Pendiente de decidir
- Pertenencia punto↔instancia: única fuente de verdad (punto→instancia o instancia→puntos).
- Polimorfismo de puertos: `virtual` vs `concepts` (tendencia: `virtual` en fronteras, `concepts` dentro).
- Descomposición de `VanillaMapper` en dominio / puertos / adaptadores.
- Telemetría: puerto de observabilidad (eventos, métricas, trazas) con adaptadores (JSONL, Rerun, wandb, Tracy).
- Gestión de errores detallada (errores también como eventos de telemetría).

## Herramientas
- ASan/TSan no funcionan con MinGW: los bugs de concurrencia se validan en Linux.
