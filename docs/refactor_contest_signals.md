# Refactor brief — Señales del hot path del contest y arreglo del denominador de `persistence`

> **Audiencia:** agente de implementación. Este documento es autocontenido: incluye contexto, motivación,
> ubicaciones exactas en el código (verificadas 2026-07-02), diseño de datos, cambios fichero a fichero,
> fases, criterios de aceptación y riesgos. Léelo entero antes de tocar código.
>
> **Regla de oro:** las líneas citadas son del estado a 2026-07-02. Antes de editar, **re-verifica** con grep
> por nombre de función/símbolo — pueden haberse movido.

---

## 0-STATUS · Qué está implementado (2026-07-06)

**HECHO y verificado end-to-end** (office0, GT, contest only+split; 19 tests unitarios pasan):

- **Arreglo del denominador de persistence** — `store.claims_of(p)` como denominador. Acotada [0,1] (max=1.000, 0 >1).
- **Contadores por punto (dicts en `ContestStore`, id permanente):** `_grabs` (robos, ya existía), `_claims`
  (reclamos totales = denominador), `_sightings` (P3 = veces matcheado). Con record/prune/serialización.
- **Telemetría Tier 2 por KF** — `n_matched, n_pre_assign, n_used, n_orphans, n_births, n_robos` (gated en `config["log"]`).
- **Organización de salida** — ver §14-LAYOUT.

**NO hecho (pendiente, ver §10 y §15-PERF):**
- Migración de dicts a **tensores GPU** (O3): hoy `_claims`/`_sightings` son dicts Python (~300MB en office0). Tensores
  = ~16MB + sin `.tolist()`/loop por KF. Es la optimización clave para online real-time.
- **Gates** de coste: `record_sighting`/`record_claim` corren aunque `contest_fusion=off`; `sightings` (P3) no tiene
  consumidor de decisión aún.
- **P6 completo** (dict `{punto:{inst:n}}` por-instancia): se implementó solo el **escalar** `_claims` (= suma), que
  basta para el denominador. El desglose por-instancia no se necesita todavía.

---

## 1. Contexto (qué es el contest)

El **contest** (`ovo/entities/contest/`) decide fusiones/splits de instancias 3D observando **qué instancia le roba
puntos a cuál** a lo largo de los keyframes. Piezas:

- `store.py` — acumula, por punto disputado, cuántos KFs lo reclamó cada "ganador". Hoy **asimétrico**: solo
  guarda robos.
- `aggregator.py` — deriva features por par `(perdedor L, ganador W)`: `containment`, `reverse_containment`,
  `total_grabs`, `persistence`, `focus`, `firm_points`, `split_points`.
- `discriminator.py` — árbol de decisión sobre esas features → `MERGE_CONTAINMENT` / `SPLIT` / `NO_ACTION` / `DEFER`.
- `manager.py` — fachada (`record`, `report`, `on_merge`, `on_remove`).

**Roles (por keyframe, por punto):**
- **Ganador (W)** = instancia de la máscara SAM que cubre el punto ahora.
- **Perdedor (L)** = instancia dueña actual del punto.
- Si `W != L` → **robo** → se registra.

---

## 2. Motivación — el bug del denominador de `persistence`

`persistence` mide si el robo de W es dominante o un parpadeo. Hoy (`aggregator.py:82`):

```
persistence(L,W) = (KFs en que la máscara de W robó el punto)   ← record, reloj SEMÁNTICO
                   ──────────────────────────────────────────
                   (veces que el punto se vio: point_obs)        ← pcd_obs, reloj GEOMÉTRICO
```

**Dos causas independientes hacen que el denominador esté inflado / mal escalado:**

1. **Cadencia distinta.** `pcd_obs` sube en el paso de *mapping* (cada `map_every`=5 frames + movimiento).
   `record` sube en el paso *semántico* (cada `segment_every`=10 frames). El denominador tic-tac ~2× más rápido
   → cuenta momentos donde SAM ni corrió (imposible robar).
2. **Máscaras.** `pcd_obs` es visibilidad **geométrica**; no sabe de máscaras. SAM no es perfecto: a veces el punto
   se ve pero no cae bajo ninguna máscara. Esos momentos inflan el denominador sin haber sido oportunidad de robo.

**Consecuencia:** techo estructural `< 1` (en este run map÷5/seg÷10 ~0.5), y el umbral `min_persist_split=0.1`
del discriminador **significa cosas distintas según config y trayectoria**. No transfiere si cambian las cadencias.

**Estado (2026-07-06): IMPLEMENTADO + VERIFICADO end-to-end.** (Ver §0-STATUS y §13-REFERENCIA abajo.)

Verificación del bug (office0): se cruzó `contest.json` (robos/punto) con `pcd_obs`. Resultado **opuesto a la
hipótesis y peor**: `max(robos/pcd_obs) = 9.0`, persistence **NO acotada en [0,1]** (podía dar >1), no un ceiling<1.
Causa: `pcd_obs` (mapping) gateado por movimiento → sube poco; `record` (semántico) sube cada `segment_every` sin
importar movimiento → numerador > denominador.

Tras el arreglo (verificado end-to-end): persistence en [0,1], **max=1.000, 0 valores >1**. 19 tests unitarios pasan.
Sin regresión (6 fallos pre-existentes en test_batch_runner, ajenos).

---

## 3. Objetivo del refactor

1. **Arreglar el denominador** de `persistence` haciéndolo intrínseco y en el mismo reloj (semántico, mask-aware).
2. **Capturar señales nuevas** del hot path, separadas en dos tiers:
   - **Tier 1 (por punto)** — alimentan el contest. Persisten, indexadas por id permanente.
   - **Tier 2 (por KF)** — telemetría. Escalares al logger, no son estado.
3. Dejar el hot path **preparado** para la futura versión real-time incremental (no implementarla aquí).

---

## 4. El hot path — dónde ocurre todo (verificado)

Cadena: bucle principal → `_run_semantic_step` (cada `segment_every`) → `detect_and_track_objects` →
`_match_and_track_instances` → `_track_objects`.

### `_match_and_track_instances` (`ovo/entities/ovo.py:276`)
```
:305  frustum_mask            → puntos en el cono de visión
:311  matched_points_idxs, matches = match_3d_points_to_2d_pixels(...)
        matched_points_idxs = índices (dentro del frustum) de los puntos que reproyectan bien (VISTOS)
        matches             = (N,2) píxel (x,y) de cada uno
:317  matched_seg_idxs = seg_map[matches[:,1], matches[:,0]]   → id de máscara SAM (o -1 si ninguna)
:319  frustum_points_ids, frustum_points_ins_ids = points_ids[frustum_mask], points_ins_ids[frustum_mask]
:320  → _track_objects(...)
```

### `_track_objects` (`ovo/entities/ovo.py:335`)
Bucle **por máscara** (`for map_idx in range(seg_map.max()+1)`, `:350`):
```
:352  map_points = matched_points_idxs[matched_seg_idxs == map_idx]   # puntos bajo esta máscara
:353  if len(map_points) > track_th:                                  # máscara con suficientes puntos
:355    assigned_mask = points_ins_ids[map_points] > -1               # cuáles YA tienen instancia
:358    if assigned_mask.sum() > track_th:                            # RAMA A: instancia conocida
:359      map_ins_id = torch.mode(...).values.item()                  # ganador = instancia mayoritaria
:361      contested = points_ins_ids[assigned_idx] != map_ins_id      # ★ ROBO: dueño != ganador
:363      self.contest.record_grab(points_ids[assigned_idx[contested]], map_ins_id)
:370    elif len(unassigned_points_ids) > track_th:                   # RAMA B: instancia NUEVA
:372      self.next_ins_id += 1                                       # nacimiento
:377    if map_ins_id > -1:                                           # asignación efectiva
:379      points_ins_ids[map_points[~assigned_mask]] = map_ins_id
```

**Nota importante:** en `_track_objects`, `matched_points_idxs` indexa dentro del **frustum**, y
`points_ins_ids`/`points_ids` son los del frustum. Cualquier contador nuevo debe usar `points_ids[...]`
(= **id permanente**) como clave, no la posición.

---

## 5. La partición de los puntos matcheados (estructura organizadora)

Cada punto de `matched_points_idxs` cae en **exactamente una** hoja. Todas las señales son hojas de este árbol:

```
matcheados (n_matched)
├── YA ASIGNADOS  (points_ins_ids > -1)            → n_pre_assign
│   ├── bajo máscara usada
│   │   ├── LEAL   (dueño_máscara == su instancia)          → P6 (claim leal)
│   │   └── ROBO   (dueño_máscara != su instancia)          → P6 (claim robo) + record + n_robos
│   └── sin máscara usada  → HUÉRFANO-ASIGNADO (3a)         → P4
└── SIN ASIGNAR   (points_ins_ids == -1)           → n_pre_unassign
    ├── bajo máscara usada → NACE/CRECE (se le asigna)
    └── sin máscara usada  → HUÉRFANO-LIBRE (3b)            → derivado
```

**Diseño clave:** implementar **una función de partición** que clasifica los matcheados en estas hojas una vez
por KF. De ahí caen: los escalares Tier 2 (logger) y los eventos Tier 1 (por punto). No instrumentar cada señal
por separado.

---

## 6. Señales a capturar (tabla final)

### Tier 1 — por PUNTO (persisten, id permanente, alimentan contest)

| ID | Señal | Qué cuenta | Almacenamiento | Coste |
|----|-------|-----------|----------------|-------|
| **P6** | store simétrico `{punto:{inst:n}}` | reclamos **bajo máscara** (leal + robo), +1 por dueño de máscara | dict keyed por id permanente (ver §7) | alto |
| **P3** | `sightings` — veces matcheado (escalar) | +1 cada vez que reproyecta bien (1+2+3) | tensor hermano de `pcd_obs` (posición) | bajo |
| **P4** | `orphan_assigned` — huérfano-asignado 3a (escalar) | +1 si matcheado, `ins>-1`, sin máscara usada | tensor hermano de `pcd_obs` (posición) | bajo |
| P5 | `pcd_obs` *(ya existe)* | visibilidad geométrica | `vanilla_mapper.py:25` | — |

### Tier 2 — por KF (escalares → `logger.stats`, telemetría)

| ID | Señal | Qué mide |
|----|-------|----------|
| K1 | `n_matched` | puntos en matching este KF |
| K2 | `n_pre_assign` | de esos, ya tenían instancia (madurez del mapa) |
| K7 | `n_used` | máscaras que asignaron algo |
| K8 | `n_orphans` | matcheados sin máscara usada (calidad SAM) |
| K9 | `n_births` | instancias nuevas creadas |
| K10 | `n_robos` | eventos de robo (contested) este KF |

### Derivadas — NO se almacenan, se calculan al vuelo

| Derivada | Fórmula |
|----------|---------|
| `n_pre_unassign` | `n_matched − n_pre_assign` |
| cobertura SAM | `1 − n_orphans / n_matched` |
| tasa de disputa | `n_robos / n_pre_assign` |
| fiabilidad del punto | `suma(P6[pt]) / P3[pt]` |
| huérfano-libre (3b) | `P3[pt] − suma(P6[pt]) − P4[pt]` |
| **persistence(L→W)** | `P6[pt][W] / suma(P6[pt])` |

### Fuera (con motivo)
- P1 robos / P2 leales sueltos → **absorbidas por P6**.
- `n_sam`, `n_over_th`, "máscaras aire" → redundantes con `n_used` (añadir luego si hace falta el ratio).
- grado de disputa → derivable de P6. recencia → pausada. Señales por instancia (fragilidad/hub) → pausadas
  (fragilidad ya en P4).

---

## 7. Almacenamiento

### 7.1 Escalares P3, P4 → tensores hermanos de `pcd_obs`

`pcd_obs` es un tensor por punto **indexado por posición**, mantenido por el SLAM junto a `pcd`, `pcd_ids`,
`pcd_normals`, `pcd_colors` (crecen/podan/reordenan juntos). Añadir P3/P4 como hermanos:

- **Dónde:** `ovo/slam/vanilla_mapper.py` — declarar junto a `pcd_obs` (`:25`), inicializar en el `vstack` de
  puntos nuevos (`:101`), exponer con un getter tipo `get_point_observations` (`:120`). Heredado por
  `SimulatedSLAM` automáticamente.
- **Incremento:** `scatter_add_` vectorizado sobre posiciones (como `pcd_obs` en `:71`). Pero P3/P4 se
  incrementan en el paso **semántico** (con las máscaras), no en `map()`. Opciones:
  - (a) El paso semántico calcula los deltas por posición y llama a un método del SLAM
    `add_semantic_obs(positions, orphan_mask)` que hace el `scatter_add_`.
  - (b) O el semántico actualiza directamente los tensores expuestos por referencia.
- **Ventajas:** vectorizado/GPU; poda/reorder **gratis** (el SLAM ya lo hace); **instancia-agnósticos** →
  `on_merge`/`on_remove` NO los tocan; el aggregator los lee con la **misma** `searchsorted` (`_owner_lookup`)
  que ya usa para `point_obs`.

### 7.2 P6 (sparse 2D) → clave = id permanente

**Ahora (batch, recomendado para empezar):** extender el `ContestStore` actual a **simétrico**.
- Hoy `store.record_grab(points, grabber)` (`store.py:28`) solo registra robos. Hay que registrar **también los leales**
  (cuando `dueño_máscara == instancia_del_punto`).
- Estructura ya existente sirve: `_grabs: {punto:{inst:n}}` (`store.py:23`) + `_by_grabber` (`:25`).
- `on_merge`/`on_remove`/`prune_to_live` (`:46`,`:58`,`:65`) ya funcionan; verificar que siguen coherentes con
  las entradas leales.
- Coste: memoria crece (antes solo disputados; ahora también leales). Aceptable para batch.

**Futuro (real-time incremental, NO en este refactor):** COO en tensores `pt_slot[]`, `inst_id[]`, `count[]` con
hash `(pt,inst)→fila`; agregación por `segment_sum`; merge = remap columna `inst_id` + coalesce. Documentar como
trabajo futuro (fase R1 del rediseño real-time), no implementar.

### 7.3 Puente posición ↔ id permanente
El mapa expone `pcd_ids` (id permanente en cada posición). El aggregator ya cruza id→posición con `searchsorted`
sobre ids ordenados (`aggregator.py:_owner_lookup`). Escalares (posición) y P6 (id) se juntan ahí sin código nuevo.

### 7.4 Tier 2 K → logger
Escalares por KF a `logger.stats` vía `log_ovo_stats` (`ovo/entities/logger.py`), como los `t_contest_*`.
No son estado, no persisten como tensores.

---

## 8. Cambios fichero a fichero

### 8.1 `ovo/slam/vanilla_mapper.py`
- Declarar `self.pcd_sightings` (P3) y `self.pcd_orphan` (P4) junto a `pcd_obs` (`:25`), int32, misma forma.
- Añadirlos al `vstack` de puntos nuevos (`:101`, inicializar a 0 para puntos nuevos) y al `map_dict`
  save/load (`:157`, `:167`) para checkpoints.
- Getters `get_point_sightings()`, `get_point_orphan()`.
- Método `add_semantic_obs(positions, orphan_positions)` que hace `scatter_add_` sobre esos tensores.
- **Verificar** que `SimulatedSLAM` (`ovo/slam/simulated/slam.py`) hereda sin romper (usa `super().map`), y que
  su rama de jump (`:180`) no necesita ajuste (P3/P4 se tocan en el semántico, no en map()).

### 8.2 `ovo/entities/ovo.py` — `_track_objects` (`:335`) y `_match_and_track_instances` (`:276`)
- Implementar la **función de partición** sobre `matched_points_idxs` / `matched_seg_idxs` / `points_ins_ids`.
- Calcular por KF: `n_matched`, `n_pre_assign`, `n_used`, `n_orphans`, `n_births`, `n_robos`.
- Marcar por posición: cuáles matchearon (P3 +1) y cuáles son huérfano-asignado (P4 +1) → llamar
  `slam.add_semantic_obs(...)`.
- Registrar leales además de robos → llamar a un `record` extendido (ver 8.3).
- Emitir K1,K2,K7,K8,K9,K10 al logger (solo si `config["log"]`).
- **Cuidado:** capturar `n_pre_assign` y el estado de asignación **ANTES** de que el bucle mute
  `points_ins_ids` (`:379`).

### 8.3 `ovo/entities/contest/store.py` + `manager.py`
- Extender `record` (o añadir `record_claim`) para aceptar leales, no solo robos → store simétrico.
- Verificar `on_merge`/`on_remove`/`prune_to_live`/`to_dict`/`from_dict` con entradas leales.

### 8.4 `ovo/entities/contest/aggregator.py` — el arreglo del denominador
- Hoy `persistence_sum[(a,w)] += c / tobs` con `tobs` = `point_obs` (`:82`).
- **Cambiar** a denominador intrínseco de P6: `persistence(L,W) = P6[pt][W] / suma(P6[pt])`.
- El aggregator ya tiene el store; con P6 simétrico, `suma(P6[pt])` = suma de todos los reclamos del punto.
- **Decisión abierta:** denominador `1+2` (solo P6, sin huérfanos — recomendado) vs `1+2+3` (usar P3).
  Por defecto **1+2** (no castiga fallos de SAM). Dejar el otro calculable para comparar.
- Mantener `point_obs` como referencia/telemetría, ya no como denominador.

### 8.5 `ovo/entities/logger.py`
- Confirmar que las K entran por `log_ovo_stats`; añadir claves a la lista de stats si hace falta (`:31`).

---

## 9. Verificación (antes y después)

### 9.1 Confirmar el bug (barato, datos existentes)
Cruzar `contest.json` (robos por punto) con `pcd_obs` del checkpoint `ovo_map.ckpt`. Calcular
`max(max_robos / pcd_obs)` sobre todos los puntos:
- si ni los más robados superan ~0.5-0.6 → techo `<1` → **bug confirmado**.
- si alguno llega a ~1 → los relojes se alinean por movimiento → **cosmético**.

### 9.2 Confirmar el arreglo (tras implementar)
- Un punto robado **siempre que se segmenta** debe dar `persistence ≈ 1` con el denominador P6 (imposible antes).
- Comparar distribución de `persistence` antes/después; ver **cuántos veredictos cambian** (un arreglo que no
  mueve decisiones es cosmético — reportarlo).
- Correr el experimento de referencia (office0, GT, contest only+split) y comparar `contest_verdicts.csv`.

### 9.3 No romper
- `t_contest_*` (timing) sigue funcionando.
- `observe` mode sigue sin mutar el mapa.
- Checkpoints save/load consistentes con los tensores nuevos.
- Tests: `tests/unit/test_ovo_*.py`, `test_fusion_*` deben pasar.

---

## 10. Fases sugeridas (orden de implementación)

1. **Verificar el bug** (9.1) — no toques código si sale cosmético; re-evalúa prioridad.
2. **Tensores P3/P4** en el mapper + `add_semantic_obs` (8.1). Sin usarlos aún.
3. **Función de partición** en `_track_objects` (8.2): emitir K1,K2,K7,K8,K9,K10 al logger + incrementar P3/P4.
   Verificar telemetría con una corrida.
4. **Store simétrico** (8.3): registrar leales.
5. **Arreglo del denominador** en aggregator (8.4) usando P6.
6. **Verificar el arreglo** (9.2): distribución de persistence + veredictos cambiados.

Cada fase es un commit verificable de forma independiente.

---

## 11. Decisiones abiertas (confirmar con el humano antes de cerrar)

| # | Decisión | Recomendación por defecto |
|---|----------|---------------------------|
| D1 | Denominador `1+2` (P6) vs `1+2+3` (P6+P3) | **1+2** (no castiga fallos de SAM) |
| D2 | P6 dict simétrico ahora vs COO tensores | **dict ahora**; COO solo en fase real-time |
| D3 | Incluir `n_sam` en Tier 2 | fuera ahora; añadir si se quiere el ratio `n_used/n_sam` |
| D4 | P3/P4 en el SLAM (posición) vs en el contest (id) | **SLAM/posición** (hermanos de `pcd_obs`, prune-free) |

---

## 12. Riesgos y notas

- **Memoria (P6 simétrico):** guardar leales de todos los puntos vistos crece el store. Monitorizar; si explota,
  considerar contador escalar de "reclamos totales por punto" en vez del dict completo.
- **Medición GPU:** si en el futuro se quitan los `.cpu()`/`.tolist()` (kNN-GPU, store GPU-resident), el
  cronometraje con `time.time()` deja de valer → usar `torch.cuda.Event`. (Ver informe de tiempos.)
- **Clave permanente:** todo lo por-punto debe usar id permanente (`points_ids`), nunca posición cruda, salvo los
  tensores hermanos del SLAM que el propio SLAM mantiene alineados.
- **Cadencias:** `map_every` y `segment_every` son config (5 y 10 en el run de referencia). El arreglo de P6 las
  vuelve irrelevantes para persistence (mismo reloj semántico), que es justo el objetivo.
- **RE-CALIBRAR `min_persist_split` (0.1):** estaba ajustado contra la persistence rota (desacotada, podía dar >1).
  Con el arreglo, persistence se acota en [0,1] y baja en general (denominador ahora incluye leales). El umbral 0.1
  probablemente deje de discriminar igual → **validar visualmente** los splits/merges y re-tunear si hace falta.
- **Coste hot path:** `record_claim` se llama para TODOS los puntos asignados bajo máscara (muchos más que los
  robos), con un `.cpu().tolist()` por máscara. Vigilar coste del paso semántico; si molesta, mover a tensor
  GPU-resident (fase real-time).

---

## 13-REFERENCIA · Señales del contest (implementadas)

### Primarias — se registran directas en el hot path (id permanente)

| Señal | Estructura | De dónde sale | Alimenta |
|-------|-----------|---------------|----------|
| **grabs** (robos) | `_grabs {punto:{grabber:n}}` | `record_grab()` en `ovo.py:_track_objects` (~L376), puntos *contested* | total_grabs, firm_points, containment, focus |
| **claims** | `_claims {punto:n}` | `record_claim()` (~L373), todo asignado bajo máscara | **denominador de persistence** |
| **sightings** (P3) | `_sightings {punto:n}` | `record_sighting()` (~L351), todo matcheado | fiabilidad, P4 |
| **size** (tamaño instancia) | del mapa | `points_ins_ids.unique` (`aggregator.py:50`) | denominador de containment (nd, nch) |

`grabs/claims/sightings` viven en `ContestStore` → `fusion/contest/contest.json` (claves `grabs`/`claims`/`sightings`).

### Secundarias / derivadas — el aggregator las calcula por par; el discriminator las consume

| Señal | Fórmula | Para qué |
|-------|---------|----------|
| containment | `firm_points / nd` | banda strong(≥high)/partial/baja |
| reverse_containment | `firm(ch→def) / nch` | frontera simétrica/bidireccional |
| total_grabs | `Σ c` | gate de ruido (`min_mass`) |
| **persistence** | `media_p( c / claims(p) )` — [0,1] (`aggregator.py`) | gate de parpadeo (`min_persist_split`) |
| focus | `firm_points / disputed_total(def)` | activar rama dominancia |
| split_points | ids de esos puntos | aplicar el SPLIT |

### Derivadas de diagnóstico (sin consumidor de decisión aún)

| Señal | Fórmula | Para qué |
|-------|---------|----------|
| P4 huérfano-asignado | `sightings − claims` | fallos de SAM sobre punto asignado (fragilidad) |
| fiabilidad | `claims / sightings` | consistencia de cobertura del punto |

### Telemetría Tier 2 (por KF, `logger/contest/`, NO decide)
`n_matched, n_pre_assign, n_used, n_orphans, n_births, n_robos`. Derivadas: `cobertura=1−n_orphans/n_matched`,
`tasa_disputa=n_robos/n_pre_assign`.

---

## 14-LAYOUT · Ficheros de salida (implementado)

```
{scene}/
├── fusion/contest/          datos del contest
│   ├── contest.json           estado por punto: {"grabs":{...}, "claims":{...}, "sightings":{...}}
│   └── contest_verdicts.csv   veredictos por report
└── logger/contest/          series por KF
    ├── t_contest_*.log         timing (8 buckets)
    └── n_matched.log, n_pre_assign.log, n_used.log,
        n_orphans.log, n_births.log, n_robos.log   telemetría Tier 2
```
`fusion_decisions.csv` se queda en `fusion/` (fusión clásica, no contest). Routing en `logger.py`
(`_is_contest_stat` → `logger/contest/`); dirs creados en `Logger.__init__`; dump paths en `ovomapping.py`
(set_output_dir → `fusion/contest`). Lectores `Study_seg/*.py` toman ruta por CLI → no afectados.

---

## 15-PERF · Impacto en rendimiento online (medido)

| Qué | ¿Online? | Detalle |
|-----|----------|---------|
| Ficheros `.log` (disco) | **NO** | `write_stats` una vez al final, no por frame |
| Telemetría Tier 2 | **poco** | gated en `config["log"]` (default true); unos `.item()`/KF |
| **record_claim + record_sighting** | **CPU pequeño / RAM notable** | siempre activo (solo respeta `contest.enabled`) |

- **CPU:** medido en `t_obj` (envuelve `_match_and_track_instances`→`_track_objects`): 43.0→43.8 ms/KF, dentro del
  ruido de warmup. `record_sighting` procesa ~200k pts/KF con `.cpu().tolist()`+loop dict, pero corre cada
  `segment_every` sobre un paso ya de ~44 ms.
- **RAM:** dicts crecen a ~2M (sightings) + 1.4M (claims) ≈ **300MB** en office0. No dio OOM. **Es el coste real.**

**Mitigaciones para real-time (pendientes):**
1. Migrar `_claims`/`_sightings` a **tensores GPU** (id permanente == posición, verificado: el SLAM solo appende):
   `scatter_add`, sin `.tolist()`/loop, ~16MB en vez de 300MB. (Deviación de O3 documentada — hoy son dicts.)
2. **Gate** `record_sighting`/`record_claim` en `contest_fusion != off` → runs clásicos pagan cero.
3. `sightings` (P3) sin consumidor de decisión → tras flag de diagnóstico (off por defecto online).
4. `config["log"]=false` apaga la telemetría Tier 2.

---

## Referencias
- Informe de tiempos y diseño del contest: `docs/contest_timing_analysis.html` (§"Hallazgo abierto" → RESUELTO).
- Memoria de proyecto: `contest-timing-2026-07-02`.
- Tests: `tests/unit/test_contest_store.py`, `test_contest_aggregator.py`, `test_logger_contest_routing.py`.
