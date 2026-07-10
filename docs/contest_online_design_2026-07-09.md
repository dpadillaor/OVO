# Contest → online: diseño (consolidación 2026-07-09)

**Rama:** `study/contest-realtime-online` (desde `study/point-instance-assignment`).
**Estado:** pensamiento de diseño consolidado. **Aún NO se ha tocado código.** Próximo paso:
requisitos técnicos de R1, plumbing, arquitectura (SOLID/SRP), validación.

Documento de continuidad. Recoge el razonamiento de la sesión de diseño; no es un spec de
implementación todavía. Lectura previa recomendada: `docs/contest_timing_analysis.html` (§9
rediseño real-time R1–R5, del que esto parte y al que afina) y `docs/contest_firmness_redesign_2026-07-07.md`
(discriminador vigente).

---

## 0. El reframe base
Hoy contest = **acumular continuo (hot) → decidir todo de golpe (cold, batch)**. Online = tejer
la decisión en el bucle, trabajo **acotado por frame**. Clave del salto: **casi todo contest ya es
online; solo la DECISIÓN es batch.**

## 1. Las dos mitades
| Mitad | Qué hace | Estado |
|---|---|---|
| **ACUMULAR** (`record_grab/claim/sighting`, `_track_objects`) | store gota a gota, cada KF | **ya es online**, solo optimizable (GPU) |
| **DECIDIR** (`report()`, cadencia de fusión) | big-bang: deriva TODAS las PairFeatures desde cero → clasifica → aplica | **lo único batch** — esto se rediseña |

→ "Contest online" = hacer que las **métricas derivadas también goteen**, no reescribir contest.

## 2. Las 3 preguntas que el batch resolvía implícito (= R1/R2/R3 del timing doc)
1. **¿Sobre QUÉ decides?** → solo lo que cambió este KF (**R1**).
2. **¿CUÁNDO decides?** → cuando cruza umbral, edge-triggered (**R2**) — *pendiente de pensar*.
3. **¿CUÁNTO por frame?** → presupuesto N/frame (**R3**) — *pendiente de pensar*.

## 3. Decisiones de diseño tomadas

### a) RECOMPUTAR, no mantener vivo
Mantener agregados con `+=/-=` obliga a **deshacer contribuciones pasadas** (reversión):
- `exclusivity` de (D,C) cambia cuando entra un challenger nuevo C' sobre un punto ya contado;
- la media `persistence` cambia cuando sube `claims[p]` (necesitas el valor viejo para restarlo).

Eso exige memoria por punto + undo → drift silencioso. **Recomputar** desde los puntos lee la
verdad actual, cero undo, cero drift. Cómputo extra despreciable → robustez gratis.

### b) Unidad sucia = DEFENDER, no par
Radio de impacto de un robo `grab(p,C)`: cambia persistencia de p → firm del par (D,C) →
`focus` de **TODOS** los pares de D (comparten `n_disputed(D)`) → `exclusivity` de otros (D,C').
Todo acotado a **los pares de un defender**. `size(D)` NO cambia por un robo. →
`dirty_defenders.add(owner(p))` en el hot path (O(1)); recomputar `pairs_for_defender(D)`.

### c) Niveles: punto = verdad, par/instancia = recomputado
Cadena `punto → (agregar) → par → (normalizar por instancia) → feature`.
- **Nivel PUNTO** (store): `grabs[p][C]`, `claims[p]`, `sightings[p]`. Único vivo/incremental.
- **Nivel PAR** (agregado desde puntos): `firm_points` (cuenta), `total_grabs` (suma),
  `persistence` (media), `exclusive_points` (cuenta), `split_points`.
- **Nivel INSTANCIA**: `size(D)` (del mapa), `n_disputed(D)`.
- **Cruce par×instancia**: `containment = firm_points/size(D)`, `focus = firm_points/n_disputed(D)`.

Par e instancia se **recomputan** por defender sucio. Es `aggregator.pairs()` de hoy **con alcance
a un defender** — no hay mecanismo incremental nuevo frágil.

### d) Frontera GPU/CPU
Dos cargas opuestas: **acumular** (gorda, uniforme → GPU, `scatter_add`; es R5) vs
**recomputar+decidir** (pequeña, dispersa, branchy → CPU). GPU frena lo pequeño e irregular
(overhead de kernel launch + sync). Diseño: store en GPU, bajar a CPU **solo los escalares de los
defenders sucios** (poquitos), no los 2M puntos como hoy.

## 4. Calibración de coste
- Cómputo online = **ruido** (~10³–10⁴ ops/KF, microsegundos vs SAM). Total ≈ batch, repartido,
  sin pico.
- Hoy `report()` ya es ~0.1s (la geometría kNN de 1.5s del timing doc está **BORRADA**).
- Riesgo de subir el total: recomputar un defender caliente muchas veces → lo evita R2 (coalesce
  por dirty set: 50 robos → 1 recompute).
- **El coste real nunca fue cómputo:** RAM (store ~300MB de dicts) + sync `.cpu()` por KF. Online
  no los empeora; R5 los mata.

## 5. Lo genuinamente NUEVO (el nudo, pendiente de profundizar)
**Bucle de realimentación tracking↔contest.** Hoy contest decide sobre mapa congelado. Online, un
MERGE cambia `points_ins_ids` (el owner) → re-entra en el tracking del KF siguiente → cambia qué
cuenta como "robo" → cambia la evidencia futura. Contest se influye a sí mismo vía tracking.
Probablemente bueno (correcciones tempranas mejoran tracking posterior), pero es lo que no existe hoy.

## 6. Estrategia de validación (shadow testing)
- El **batch es el ORÁCULO**, no "lo viejo a quitar".
- **Fase sombra:** online calcula en paralelo, batch manda (decide y actúa). Cada report: comparar
  features online vs batch → **deben coincidir bit a bit** (por construcción: misma fórmula, alcance
  distinto). Discrepancia = bug de invalidación localizado → **caza exactamente el "olvidé marcar sucio"**.
- **Fase viva:** solo cuando la sombra cuadra siempre → online decide y actúa. Ahí diverge del batch
  **a propósito** (feedback §5), y deja de ser comparable end-to-end.

## 7. Naturaleza de R1–R5
Son **bocetos de una línea** del timing doc §9, no specs. R1 textual ofrece dos caminos (mantener
vivo / marcar sucio) sin elegir; esta sesión lo desempata: **recompute, grano defender, punto=verdad**.
- **R4** (geometría async) = **moot** (seam/color kNN borrados en el rediseño del discriminador).
- **R5** = solo la acumulación del store a GPU.

## 8. Pendiente de pensar (siguiente sesión)
- **Requisitos técnicos de R1 + plumbing + arquitectura (SOLID/SRP)**: quién marca sucio, quién
  mantiene el dirty set, quién recomputa, dónde vive todo. Separar responsabilidades sin acoplar al
  hot path.
- **R2**: cuándo un defender sucio dispara decisión de verdad (umbral / edge-trigger).
- **Denominadores vivos**: cómo tener `size(D)` y `n_disputed(D)` frescos sin recomputar enteros
  (¿contador vivo mantenido por el SLAM/objects? ¿recompute al ensuciar?).
- **Invalidación en eventos raros**: merge/split aplicado cambia `size` → ensucia todos los pares de
  esa instancia (radio grande pero raro). Crecimiento del mapa sube `size(D)`.
- **Feedback loop** (§5) a fondo.

## 9. Bugs de drift encontrados en el código actual (si se toca)
- `ovo/entities/ovo.py:685-690`: el filtro `split_mode` busca `"parcial"`/`"dominancia"` en
  `v.reason`, pero el discriminador nuevo emite reasons `"focus (...) -> split"` /
  `"por-par (...) -> split"` → solo `contest_split_mode="all"` aplica splits (los manifests usan `all`).
- `ovo/entities/contest/manager.py:8` docstring dice "el Actuator se enchufa en la fase siguiente" —
  desactualizado: `_apply_contest_merges` (`ovo.py:642`) ya existe y aplica.

---

## 10. Requisitos R1 (inventario) — modo sombra, sin cambio de comportamiento

### 10.1 Contrato de invalidación (el núcleo)
Regla: *todo evento que cambie las features de un defender lo marca sucio.* Con `reverse_containment`
**muerto para decidir** (verificado: `discriminator.classify` no lo usa; solo se loguea), la
invalidación por `size` queda local a D (D como defender, no como challenger).

Verificado en código: `grabbed = assigned_idx[contested] ⊆ assigned_idx = claimed`. Todo robo es un
claim → **un solo marcado por máscara** cubre grab y claim (el claim leal también importa: sube
`claims[p]` → baja `persistence` de pares donde p ya estaba disputado).

| Evento | Marcar sucio | Dato (disponible) |
|---|---|---|
| **asignación bajo máscara** (`_track_objects`, funde grab+claim) | `points_ins_ids[assigned_idx].unique()` | ya en mano ✓ |
| punto nuevo asignado a D (birth/crecimiento) | D (`map_ins_id`) | ✓ (coincide con ser assigned-owner; ver R4) |
| `on_merge(t,s)` | t + owners de los grabs de s; **drop s** | `_by_grabber[s]` + owner_lookup ✓ |
| `on_remove(i)` | owners de los grabs de i; **drop i** | `_by_grabber[i]` ✓ |
| split aplicado D→C | D, C | `split_points` ✓ |
| `prune_to_live` | owners afectados / refresh | live set ✓ |

Refinamiento (optimización, no v1): filtrar owners de puntos **que están en el store** → set más
chico. v1 = cut rápido (`assigned_idx.unique()`, correcto/superset, derroche acotado).

### 10.2 Componentes (SRP) — ver §11.3 para el mapa completo con las clases de resolución
1. **`DirtyDefenders`** — set sucio: `mark/drop/take`. Contenedor tonto; lo alimentan los call sites.
2. **`IncrementalAggregator`** — `features_for(defenders) → {D:[PairFeatures]}`, **puro**; reusa
   `_owner_lookup`/`_is_firm`/`_finalize`.
3. **`FeatureCache`** — `{D:[PairFeatures]}` vivo.
4. **`OnlineContestCoordinator`** — orquesta `take→recompute→cache→expose`; sin lógica propia.
5. **`ShadowValidator`** — diff batch↔online.
6. **`ContestDiscriminator`** — sin cambios (ya puro por defender).

### 10.3 Denominadores vivos
- `size(D)`: recomputar en el refresh con **un** `unique(return_counts=True)` (todos a la vez);
  leer solo los sucios. Sin contador mantenido → menos estado, menos drift.
- `n_disputed(D)`: se recomputa escaneando los disputados de D. Sin estado extra.

### 10.4 Config e instrumentación
- Flag `contest_agg = batch | online_shadow | online_live` (default `batch`, cero cambio).
- Buckets `t_refresh`, `n_dirty` (patrón `t_contest_*`). Si el store baja a GPU → `cuda.Event`.

---

## 11. Riesgos, contratos y su resolución con clases (revisión senior 2026-07-09)

Veredicto de la revisión: **el diseño de fondo no tiene fallos de concepto.** Los problemas son de
**infra-especificación** y se resuelven poniendo cada responsabilidad en su clase. Anotados para no
descubrirlos "en 2 semanas".

### 11.1 CRÍTICOS
- **R1 · "O(all)→O(changed)" es falso sin índice `owner→puntos`.** Sin él, `features_for` hace
  `_owner_lookup` sobre TODOS los disputados cada refresh → mismo coste dominante que el batch, solo
  **repartido, no reducido**. Reducir de verdad exige un índice `owner→{puntos disputados}` (hermano
  de `_by_grabber`). **Decisión pendiente:** construirlo (más estado) vs aceptar "solo repartir" en
  v1 (en GT el recompute es barato). No vender "reducido" cuando aún no lo es.
- **R2 · Invariante frágil y SIN guarda:** *el owner de un punto solo cambia por `on_merge`/
  `on_remove`/split.* Verificado hoy en código, pero implícito. Cualquier futuro escritor de
  `points_ins_ids` que no notifique → dirty set podrido en silencio → decisión mala e invisible.
- **R3 · Shadow NO es bit-exact:** ints exactos, pero `persistence` (media) difiere en el último ULP
  por orden de suma. Assert exacto → falsas alarmas. Necesita tolerancia float desde el día 1.

### 11.2 MEDIOS
- **R4 · `size(D)` growth funciona por coincidencia** (la instancia que crece = mayoritaria de la
  máscara = assigned-owner). Cierto hoy, acoplado al tracking. → test explícito.
- **R5 · `prune` infra-especificado.** Irrelevante en GT (append-only, sin poda); riesgo en ORB-SLAM real.
- **R6 · La fase VIVA no tiene oráculo.** Shadow valida *features* (R1), no el bucle vivo (R2/R3+feedback).
  Lo vivo se valida por AP/visual. No engañarse: "shadow OK" cubre el plumbing, no el sistema online entero.
- **R7 · Dónde engancha el refresh** (paso mapping vs semantic): shadow obliga a report-cadence; vivo
  puede moverlo (cambia timing del feedback).

### 11.3 Resolución con clases (SOLID) — el mapa que llevamos a la fase de plumbing
**Insight:** R1+R2+R4+R5 son el mismo problema ("mantener `owner→puntos disputados` coherente") →
una sola clase los absorbe.

| Clase | Única responsabilidad | Resuelve | Estado |
|---|---|---|---|
| `ContestStore` | conteos por punto (grabs/claims/sightings) | — | existe |
| `DisputedOwnerIndex` | `owner→puntos disputados` coherente vía `on_assign/on_merge/on_remove/on_split/prune` | **R1** (acceso directo O(cambiado)), **R2** (choke-point único = invariante encapsulada), **R4** (assign como evento de 1ª clase, no coincidencia), **R5** (`prune` un solo sitio) | **nueva** |
| `DirtyDefenders` | set sucio `mark/drop/take` | plumbing | **nueva** (o fusionada con el index) |
| `IncrementalAggregator` | recomputar features de un set de defenders, **puro** | R1 (consumidor) | extiende `ContestAggregator` |
| `FeatureCache` | `{D:[PairFeatures]}` vivo | — | **nueva** |
| `ShadowValidator` | diff batch↔online, `atol` floats / exacto ints | **R3**, **R6** (docstring honesto: valida features, no comportamiento) | **nueva** |
| `OnlineContestCoordinator` | orquesta take→recompute→cache→expose (sin lógica propia) | — | **nueva** (o modo del manager) |
| `RefreshPolicy` (Strategy) | *cuándo* refrescar, desacoplado del *qué* | **R7** | **nueva** |
| `ContestDiscriminator` | decidir por defender | — | existe |

Principios: cada clase **S**RP; aggregator/discriminator **O**pen-closed (ya lo son). **Sin Protocols
/ interfaces por ahora** — YAGNI: cada pieza tiene UNA implementación, así que se usan como clases
concretas. La abstracción (Protocol/ABC) se extrae **el día que aparezca la 2ª implementación** (p.ej.
inyectar otro aggregator), no antes. Todo vive en `ovo/entities/contest/` como módulo limpio; toca OVO
por una **costura mínima** (el hot path llama `index.on_assign(...)` con los owners que ya tiene en
mano). El acoplamiento viejo de OVO NO se hereda: lo nuevo se construye limpio.

---

## 12. Contratos y dataclasses (definición)

Reusar lo existente en `types.py`: `PairFeatures`, `Verdict`, `Decision`, `InsId`, `PointId`.
**Sin Protocols** (YAGNI, §11.3): clases concretas + dataclasses inmutables + invariantes documentadas.

### 12.1 Dataclasses nuevas (frozen, value objects)

**ESENCIALES**
```python
@dataclass(frozen=True)
class FeatureTolerance:
    """R3: ints exactos, floats con atol (persistence/media difiere en ULP por orden de suma)."""
    atol: float = 1e-6
    FLOAT_FIELDS = ("containment", "reverse_containment", "persistence", "focus", "exclusivity")
    INT_FIELDS = ("firm_points", "total_grabs")

@dataclass(frozen=True)
class Mismatch:
    defender: InsId
    challenger: InsId
    field: str                 # nombre del campo de PairFeatures
    batch: float
    online: float
    kind: str = "value"        # "value" | "missing_online" | "missing_batch"
```

**RECOMENDADO**
```python
@dataclass(frozen=True)
class ComparisonReport:
    report_idx: int
    n_pairs_batch: int
    n_pairs_online: int
    mismatches: tuple          # tuple[Mismatch, ...]
    @property
    def ok(self) -> bool:
        return not self.mismatches
```

**OPCIONAL (instrumentación)**
```python
@dataclass(frozen=True)
class RefreshStats:
    n_dirty: int               # defenders tomados este refresh
    n_pairs: int               # pares recomputados
    t_seconds: float
```

### 12.2 Clases concretas — interfaz pública (impl en fase plumbing)
- **`DisputedOwnerIndex`** — `owner→puntos disputados` coherente. Único mutador de propiedad (guarda R2).
  - consulta: `points_of(defender) -> Set[PointId]` (alimenta el recompute, R1).
  - eventos (la "puerta única"): `on_assign(defender, point_ids)`, `on_merge(target, source)`,
    `on_remove(ins)`, `on_split(defender, challenger, point_ids)`, `prune(live)`.
- **`DirtyDefenders`** — `mark(defender)`, `drop(defender)`, `take() -> set` (+ clear). (Fusionable en el index.)
- **`IncrementalAggregator`** (extiende `ContestAggregator`) — `features_for(defenders, point_ids,
  points_ins_ids) -> Dict[InsId, List[PairFeatures]]`, **puro**.
- **`FeatureCache`** — `update(defender, pairs)`, `get(defender)`, `drop(defender)`, `all()`.
- **`ShadowValidator`** — `compare(batch: dict, online: dict, tol: FeatureTolerance) -> ComparisonReport`.
- **`OnlineContestCoordinator`** — orquesta `take→recompute→cache→expose`; sin lógica propia.

> El "contrato `OwnershipEvents`" **no es un Protocol**: es el conjunto de métodos-evento de
> `DisputedOwnerIndex` + la invariante documentada + el shadow test. Esa es la garantía de R2.

### 12.3 Invariantes (contrato defendido por tests + asserts en modo debug)
| Componente | Invariante |
|---|---|
| `ContestStore` | conteos monótonos no decrecientes (grabs/claims/sightings solo `++`) |
| `DisputedOwnerIndex` | `p ∈ points_of(owner(p))` ∀ p disputado; coherente con `points_ins_ids` entre eventos |
| `IncrementalAggregator` | **puro**: no muta store/map/index |
| `FeatureCache` | ∀ D no-sucio: `cache[D] == aggregator.features_for({D})` ← lo que el shadow comprueba |

---

## 13. Plan de implementación R1 (EJECUTABLE / RESUMIBLE)

> Un agente que retome debe poder continuar SOLO con esta sección + §10–§12. Marca el checklist
> §13.7 conforme avanzas. Decisiones de diseño ya tomadas (delegadas por el usuario): están LOCKED.

### 13.0 Scope y entorno
**Scope: SOLO R1 — agregador online incremental en modo SOMBRA.** El batch sigue decidiendo y
actuando (CERO cambio de comportamiento con default). **NO** R2/R3/live, **NO** aplicar decisiones
online. Objetivo doble:
1. **Correctitud**: features del agregador online == features del batch (ints exactos, floats `atol=1e-6`).
2. **Tiempo**: medir t(online) vs t(batch) por report (instrumentación TEMPORAL, se elimina, §13.6).

Entorno (verificado): envs conda `ovo` y `ovo2`. Tests: `conda run -n ovo python -m pytest
tests/unit/test_contest_online_*.py` (pytest.ini, estilo `unittest.TestCase`, ver
`tests/unit/test_contest_aggregator.py`). Runner de experimentos: `ovo2`,
`scripts/run_experiments_batch.py --manifest <m> [--preview]`.

Run de validación (corto): manifest replay desde substrato `raw-baseline-full_32ac4`, escenas
**office0** (rápida) → **office3** → **room2** (muchos splits, buen estrés). Plantilla:
`manifests/20260708_GT_CLIP_contest-perpair.yaml` + añadir `semantic.contest.agg_mode: online_shadow`.
AP no importa aquí (validamos el agregador, no la decisión); mirar el `shadow_report` y el timing.

### 13.1 Organización de ficheros (LOCKED)
Todo en `ovo/entities/contest/`:
- `types.py` — **crece**: `FeatureTolerance`, `Mismatch`, `ComparisonReport`, `RefreshStats` (§12.1).
- `online.py` — **nuevo**: `DirtyDefenders` (set), `IncrementalAggregator(ContestAggregator)`,
  `FeatureCache`, `OnlineContestCoordinator` (recibe los eventos de propiedad, marca sucio y orquesta
  take→recompute→cache; expone las features online).
- `shadow.py` — **nuevo**: `ShadowValidator` (`compare(batch, online, tol) -> ComparisonReport`).

**RECONCILIACIÓN v1 (importante):** `DisputedOwnerIndex` (owner→puntos persistente) **se DIFIERE**
junto a la optimización de "reducir" (§11.1). En v1 el coordinator NO mantiene índice persistente:
maneja los eventos (marca sucio) y `features_for` hace "bucket global + finalize dirty" (O(all
contested) por refresh). El coordinator captura, en eventos estructurales (merge/remove), los puntos
afectados en un `_pending` y resuelve sus owners en el `refresh()` (que tiene `points_ins_ids`). Como
`features_for` calcula igual que el batch (mismo `_accumulate`/`_finalize`) y solo filtra defenders,
las features son **idénticas por construcción** → el shadow valida sobre todo la **completitud del
dirty set**, no la aritmética. Consecuencia de timing esperada: **online ≈ batch** (reparte, no
reduce); si el timing lo confirma, el índice `owner→puntos` es la mejora siguiente.
- `manager.py` — **editar**: construye el coordinator si `agg_mode != batch`; en `report()`, tras
  calcular los pares batch, si sombra: `coordinator.refresh()` + `ShadowValidator.compare` + dump.
- `ovo.py` — **editar**: hooks `on_assign` (en `_track_objects`) y `on_split` (en apply); enrutar los
  `on_merge`/`on_remove` existentes también al coordinator.

`IncrementalAggregator` reusa `ContestAggregator._owner_lookup/_is_firm/_finalize`; añade
`features_for(defenders, point_ids, points_ins_ids)` que restringe `_finalize` a los defenders pedidos.
**Decisión R1 (§11.1):** v1 usa "bucket global + finalize solo sucios" (un `_owner_lookup` global +
agrupar por owner + finalizar sucios) → REPARTE, no reduce. Índice `owner→puntos` para O(changed) real
= optimización posterior SOLO si el timing (§13.0.2) muestra que hace falta. Documentarlo así, sin
vender "reducido".

### 13.2 Config (LOCKED)
En `semantic.contest` (lo lee `ContestManager` como `cfg.get(...)`; verificado: `ovo.py:82`
`ContestManager(config)` recibe el sub-config semantic, igual que `contest_fusion`):
- `agg_mode: batch | online_shadow` (default **`batch`** → cero cambio). (`online_live` = futuro, no R1.)
- `atol: 1e-6`.
Shadow se ancla dentro de `report()`, que solo corre si `contest_fusion != off` → el run de validación
necesita `contest_fusion: only` (o observe) + `agg_mode: online_shadow`.

### 13.3 Costura / event sites + verificación de fugas (contrato R2)
| Evento | Sitio | Acción |
|---|---|---|
| `on_assign(owner_ids, point_ids)` | `ovo.py:_track_objects` (~L373, tras `record_claim`) | marcar sucios `points_ins_ids[assigned_idx].unique()`; añadir puntos al index bajo su owner |
| `on_merge(t,s)` | `_apply_contest_merges:672`, `_fuse_overlapping_instances:753` (ya llaman `contest.on_merge`) | enrutar también a coordinator (mover bucket s→t, marcar t, drop s) |
| `on_remove(i)` | `_remove_missing_instances:555` (ya llama) | enrutar a coordinator (drop i, marcar owners de sus grabs) |
| `on_split(D,C,pts)` | `_apply_contest_merges` rama split (~L707) | marcar D,C; mover puntos en el index |
| `prune(live)` | `report()` / `store.prune_to_live` | index.prune(live) |
**Guarda R2:** `grep -n "points_ins_ids\[" ovo/entities/ovo.py` + revisar cada escritura de owner →
confirmar que todas notifican. Si aparece una sin notificar → fuga → añadir al contrato o al sitio.

### 13.4 Orden de implementación (cada paso VERIFICABLE por su cuenta)
1. **Dataclasses** en `types.py` + `tests/unit/test_contest_online_types.py` (construcción, `ok`).
2. **`DisputedOwnerIndex`** (+ `DirtyDefenders`) + test: invariante `p ∈ points_of(owner(p))` bajo
   assign/merge/remove/split/prune; `take()` vacía.
3. **`IncrementalAggregator.features_for`** + test: en un fixture pequeño, `features_for(all)` ==
   `ContestAggregator.pairs()` agrupado (mismo resultado, alcance total). Luego `features_for(subset)`
   == subconjunto.
4. **`FeatureCache`** + **`OnlineContestCoordinator`** (eventos → index+dirty; `refresh()` →
   `take` → `features_for(dirty)` → `cache.update`) + test del ciclo.
5. **`ShadowValidator.compare`** + test: tolerancia float, exacto int, lados faltantes (`missing_*`).
6. **Wiring**: `manager.report()` modo sombra (batch + refresh + compare + dump) + hooks en `ovo.py`
   (`on_assign`/`on_split`, enrutado merge/remove). Config `agg_mode`.
7. **Logger**: dump `shadow_report.csv` en `logger/contest/` (routing ya existe); stats `n_dirty`.
8. **[TEMP]** instrumentación de tiempo `t_online`/`t_batch` (ver §13.6 — se elimina al final).
9. **Run office0** (`agg_mode: online_shadow`) → **0 mismatches** (dentro de tol). Si hay → localizar
   defender, arreglar invalidación, repetir. Registrar timing.
10. **Run room2** (estrés splits) → 0 mismatches. Registrar timing.
11. **Quitar** la instrumentación TEMP (§13.6). Verificar tests verdes + un último run sombra limpio.
12. Actualizar §13.7 (checklist) y §13.8 (resultados de timing) en este doc.

### 13.5 Criterios de aceptación
- Tests unitarios de todos los pasos: **verdes**.
- Shadow office0 **y** room2: **0 mismatches** (int exacto, float `atol=1e-6`).
- **No behavior change**: con `agg_mode=batch` (default), veredictos idénticos a antes (el path batch
  no se toca; comprobar que un run `batch` da el mismo `contest_verdicts.csv` que el commit base).
- Timing medido y anotado (§13.8), instrumentación temporal **eliminada** (grep limpio, §13.6).
- `ShadowValidator` + modo `online_shadow` = **permanentes** (son el test de regresión del agregador).
  Solo la instrumentación de TIEMPO es temporal.

### 13.6 Instrumentación TEMPORAL a ELIMINAR al terminar
Marcar cada una con comentario `# TEMP-TIMING (remove)`:
- timers `t_online`/`t_batch` alrededor de `coordinator.refresh()` y del `aggregator.pairs()` batch en
  `report()`.
- cualquier `print`/log de comparación de tiempos y cualquier CSV/artefacto de timing en scratchpad.
Al acabar: `grep -rn "TEMP-TIMING" ovo/ studies/` → debe salir vacío. El shadow y `agg_mode` se quedan.

### 13.7 Checklist de resumción (ACTUALIZAR conforme se avanza)
- [x] 1. Dataclasses (`types.py`) + test (`test_contest_online.py`)
- [x] 2. DirtyDefenders + test (índice persistente `DisputedOwnerIndex` DIFERIDO, v1 sin él — §13.1)
- [x] 3. IncrementalAggregator.features_for + test (== batch, exacto)
- [x] 4. FeatureCache + OnlineContestCoordinator + test (ciclo)
- [x] 5. ShadowValidator + test (tolerancia float / lados faltantes)
- [x] 6. Wiring manager.report + hooks ovo.py (`note_assignment`/`on_split`, ruteo merge/remove) + config `agg_mode`
- [x] 7. dump `shadow_report.csv` + timing csv en output_dir; `shadow_ok` property
- [x] 8. [TEMP-TIMING] timers t_online/t_batch en report()
- [x] 9. Run office0 shadow == **0 mismatches** (153 pairs, 1 report; `20260709_GT_CLIP_online-shadow_98472`)
- [x] 10. Run office3 + room2 shadow == **0 mismatches** (jumps; `20260709_GTJump-J3-T0p4-R15p0_..._0de5e`)
       + **4 tests cross-report** deterministas (merge/split/remove entre reports) porque el live no lo alcanza
- [x] 11. Quitar TEMP-TIMING (grep limpio, `shadow_timing.csv` borrados, `RefreshStats.t_seconds` fuera); tests verdes
- [x] 12. Doc actualizado (§13.7 + §13.8). Run de confirmación post-limpieza: `[shadow] report 1: OK
       (153 pairs, n_dirty=133)`, sin timing en print, sin `shadow_timing.csv`. **R1 shadow COMPLETO.**

**Estado unit tests (2026-07-09):** **42/42 verdes** (`test_contest_online.py` 23: dataclasses, dirty,
cache, incremental==batch, ciclo coordinator, shadow tolerancia, **4 cross-report**; +
`test_contest_store`/`test_contest_aggregator`/`test_logger_contest_routing`). Arreglado de paso el
drift `min_count`→`min_grabs` en `test_contest_aggregator.py` (§13.9) + un assert obsoleto
(`test_no_pair_when_no_claims`: claims=0 → punto ignorado, no par).

**HALLAZGO (cadencia de report en GT):** `report()` corre **UNA vez al final** en GT (`close_loops:true`),
y **los jumps NO añaden reports** (solo meten drift; la fusión sigue siendo única al cierre). → el run
LIVE valida el path de **1 report** (acumulación completa, datos reales, `on_remove` sí se ejerce antes
del report) pero NO el **cross-report** (staleness por merge/split aplicados entre reports). Ese hueco lo
cubren los **4 tests de integración cross-report** (`TestCrossReport`): conducen el coordinator por
varios reports con merge/split/remove aplicados entre medias y exigen `all_features()==batch.pairs()`.
Cobertura combinada: live = 1-report sobre datos reales (office0/office3/room2, 0 mismatches);
integración = cross-report determinista. Para un futuro live REAL multi-report (R2/R3) el shadow ya está
listo y volvería a ser el oráculo.

### 13.8 Resultados de timing y correctitud
- **office0** (GT limpio, 1 report): **0 mismatches**, 153 pairs, n_dirty=133.
  `t_online=107.6ms` vs `t_batch=113.2ms` → **online ≈ batch** (confirma "reparte, no reduce": el
  online hace el mismo `_accumulate` global que el batch; el índice `owner→puntos` reduciría esto).
  1 solo report (GT limpio no estresa cross-report → ver paso 10 con jumps).
- **office3** (jumps, 1 report): **0 mismatches**, 460 pairs, n_dirty=293. `t_online=218ms` vs `t_batch=234ms`.
- **room2** (jumps, 1 report): **0 mismatches**, 230 pairs, n_dirty=177. `t_online=360ms` vs `t_batch=360ms`.
  (timing medido con instrumentación TEMPORAL, ya ELIMINADA — §13.6.)

**Lectura**: 1 report con casi todos los defenders sucios (p.ej. 133/153, 293/460) → el online recomputa
prácticamente todo → **timing ≈ batch, ESPERADO**. El win de tiempo real necesita (a) refresh por-KF
(dirty pequeño por refresh, R3) y/o (b) índice `owner→puntos` (reduce el `_accumulate`, §11.1). R1
shadow demuestra CORRECTITUD del plumbing (0 mismatches en 3 escenas + 4 tests cross-report); el timing
confirma que v1 **reparte, no reduce** (por diseño) → el índice `owner→puntos` es la mejora siguiente si
se quiere reducir tiempo, pero solo tiene sentido con refresh por-KF (R2/R3), no a cadencia de report.

### 13.9 Gotchas conocidos
- **`tests/unit/test_contest_aggregator.py` usa `ContestAggregator(min_count=1)`** — kwarg viejo
  (renombrado a `min_grabs`). Test pre-existente con drift; NO es de este trabajo, pero si peta al
  correr la suite, es esto (no tu cambio). Anotarlo, no arreglar salvo que estorbe.
- Bug `split_mode` reasons (`ovo.py:685-690`) y docstring viejo (`manager.py:8`) — §9. No tocar en R1.
- Config path: `ContestManager` lee `config["contest"]`; el manifest anida en `semantic.contest`. OVO
  recibe el sub-config semantic → coherente. Verificar al cablear `agg_mode`.
- Commits: **no commitear sin que el usuario lo pida** (regla vigente). Dejar working tree limpio +
  este doc como ancla de resumción.

---

## Apéndice · ubicaciones verificadas (2026-07-09, commit 0adcb9f, working tree limpio)
- Bucle por frame: `ovomapping.py::run` (L588) → `_run_mapping_and_fusion_step` (L380, gate `map_every`,
  cold path/`report` vive aquí) → `_run_semantic_step` (L298, gate `segment_every`, hot path). Mapping
  corre ANTES que semantic en el mismo frame.
- Hot path: `ovo.py::_track_objects` (L335), bucle por máscara; `record_sighting` (L351, 1/KF),
  `record_claim` (L373, por máscara), `record_grab` (L376, subconjunto contested).
- Cold path: `manager.py::report` (L77); trigger vía `update_map` (`ovo.py:586`), gated por
  `slam.map_updated` (en GT/SimulatedSLAM se pone en jumps/loop-closures).
- Aplicar: `ovo.py::_apply_contest_merges` (L642); hooks coherencia `contest.on_merge`/`on_remove`.
- Store: dicts por id permanente (`store.py`). `pcd_ids` en VanillaMapper = `arange` append-only,
  nunca reordenado ni podado → en GT id permanente == posición (habilita R5).
