# HANDOFF — Sesión contest (merge/split de instancias)

> Informe para el próximo agente. Lee esto entero antes de tocar nada. Explica el
> mecanismo desde cero, lo que hicimos esta sesión, y qué queda. Fecha: 2026-06-15.
> Rama/worktree: `study/point-instance-assignment`.

---

## 0. Cómo trabaja el usuario (LÉELO)

- Va **poco a poco**. Pide conceptos explicados **paso a paso**, con ejemplos
  concretos y números. Si vas rápido o abstracto, te para ("no lo entiendo",
  "ve más lento"). Responde con ejemplos pequeños y dibujos ASCII.
- **Valida visualmente**: tiene un visor (Rerun) y mira las instancias reales.
  Cuando dice "la 93 debería fusionar" es porque la ha visto. Fíate de su juicio
  visual sobre la verdad de cada caso.
- Quiere **análisis honesto y crítico** — "para bien y para mal". Si algo está
  mal o es un parche, díselo. Reconoce tus errores (esta sesión confundí IDs
  entre runs y me corrigió; los IDs de instancia NO son estables entre runs).
- Decide él los umbrales y el rumbo. Tú propones, mides, recomiendas.
- Idioma: español.

---

## 1. Qué es el mecanismo "contest"

OVO construye un mapa 3D semántico con instancias. Cada punto 3D pertenece a una
instancia (su dueño). El **contest** recupera una señal 2D que se descartaba: cuando
un punto cae bajo la **máscara** de OTRA instancia (no su dueño), eso es un
**conflicto**. El contest registra esos conflictos y decide qué hacer:

- **MERGE**: el perdedor es un fragmento del ganador → fusiónalos (borra el perdedor).
- **SPLIT**: el perdedor tiene un trozo que es del ganador → corta ese trozo, dáselo.
- **NO_ACTION**: ruido, frontera real, o evidencia floja → no tocar.
- **DEFER_TO_FUSION**: simétrico → que lo decida la fusión geométrica.

Vive AL LADO de la fusión geométrica existente, no dentro. Config del run actual:
`contest_fusion: only` (el contest es la ÚNICA fusión; la geométrica se apaga) y
`contest_split_mode: partial` (solo se aplican splits "parcial"; los de
"dominancia" se loguean pero NO se ejecutan).

### Arquitectura (`ovo/entities/contest/`)
- `store.py` — `ContestStore`: registro crudo `{punto: {ganador: nº_KFs}}`. Camino
  caliente (cada KF) + ciclo de vida (on_merge/on_remove/prune).
- `aggregator.py` — `ContestAggregator`: deriva features normalizadas por par a
  partir del store + estado vivo del mapa. Puro.
- `discriminator.py` — `ContestDiscriminator`: features → Verdict. La casuística.
- `manager.py` — `ContestManager`: fachada. OVO lo engancha en 4 sitios (record,
  on_merge, on_remove, report). `report()` clasifica + resuelve por par + vuelca CSV.

### Las features (entiéndelas bien — son el corazón)
Para un par direccional **A→W** (A=perdedor, W=ganador):

| feature | fórmula | qué dice |
|---|---|---|
| `containment` | chunk / \|A\| | qué fracción de A está dentro de W. **DIRECCIONAL.** Métrica reina. |
| `reverse_containment` | el mismo en sentido W→A | ≈0 = unidireccional (fragmento limpio); alto = bidireccional (mutuo) |
| `strong_points` | nº de puntos de A que vieron a W | numerador de containment |
| `mass` | Σ KFs sobre esos puntos | peso crudo. Solo filtro de ruido (`mass>=50`). NO normalizado |
| `persistence` | media(obs_bajo_W / obs_totales del punto) | estabilidad temporal. Bajo = parpadeo |
| `focus` | strong_points / puntos_disputados_totales_de_A | ¿la disputa se concentra en UN ganador? |

Clave conceptual: **containment mide el TAMAÑO del solape; focus mide si está
CONCENTRADO en un rival.** `mass`/`strong` mismo numerador, distinto denominador
que `containment`/`focus`. Hay un doc lento de esto en `Study_seg/concepto_masa.md`
y `Study_seg/casos_contencion.md`.

---

## 2. El flujo de decisión ACTUAL (tras esta sesión)

Diagrama canónico: **`Study_seg/flujo_decision_contest.puml`** (mantenlo al día).
Resumen del recorrido de cada perdedor en `discriminator.classify()`:

```
1. mass < 50 (todos los pares)         -> NO_ACTION "ruido"
2. cont >= high(0.70)  [STRONG]:
     - 1 ganador y rev>=0.70           -> DEFER_TO_FUSION (simétrico)  [muerto en office3]
     - todos misma raíz                -> MERGE
     - >=2 raíces distintas            -> NO_ACTION "frontera real"
3. 0.5 <= cont < 0.70  [PARTIAL]:
     - sim(A,W) >= 0.81                -> MERGE   (descriptor: mismo objeto)
     - rev >= 0.5                      -> NO_ACTION (bidireccional/frontera)
     - persistence < 0.1               -> NO_ACTION (parpadeo)
     - si no                           -> SPLIT "parcial"
4. cont 0.05-0.5 y focus>=0.7  [DOMINANCIA]:
     - sim(A,W) >= 0.81                -> MERGE   (descriptor: mismo objeto adyacente)
     - si no                           -> SPLIT "dominancia"
5. resto                               -> NO_ACTION "borde"
```

Luego **`_resolve_pairs`** (en manager) decide UNA vez por par no-ordenado {A,B}
juntando los dos sentidos (ver §3). Sustituye al viejo `_reconcile`.

---

## 3. Qué hicimos esta sesión (cronológico, con el porqué)

Partimos del baseline (run `..._6e7c8`) que tenía: `high=0.8`, decisión
per-perdedor independiente, y `_reconcile` como parche. Analizando office3 con el
usuario (que validó casos en el visor) salieron problemas y los arreglamos:

### (A) Decisión por PAR en vez de por perdedor — `_resolve_pairs`
**Problema:** un par físico {A,B} genera 2 veredictos independientes (A→B y B→A)
que pueden contradecirse. La PARED 63/151 (dos mitades del mismo objeto) salía
SPLIT en ambos sentidos → se robaban puntos → mapa "super mezclado". `_reconcile`
solo cubría MERGE-vs-SPLIT, no SPLIT-vs-SPLIT.
**Fix:** `manager._reconcile` → `manager._resolve_pairs` + `_join`. Decide por par
con los dos containments:
- algún MERGE → MERGE en dirección de mayor contención
- ambos SPLIT + max(cont)>=0.5 → MERGE (casi-merge, la pared)
- ambos SPLIT + max(cont)<0.5 → NO_ACTION (frontera, ej. silla)
- un solo SPLIT → se mantiene
**NOTA arquitectónica honesta:** `classify` sigue siendo per-perdedor (necesario
para la lógica multi-ganador de *frontera real*); `_resolve_pairs` es la autoridad
final por par. Es un híbrido, NO el `classify(par)` puro que llegamos a discutir.
La frontera-real (≥2 raíces) es multi-ganador y no se puede expresar pairwise.

### (B) Bajar merge a `high=0.70`
**Por qué:** la pared (cont 0.794) se quedaba a 0.006 del merge. El usuario pidió
bajar. Hecho. (Default en código.)

### (C) Guards en partial: reverse + persistencia
**Por qué:** la rama partial emitía SPLIT solo por geometría (cruda, "salvaje").
Añadimos: `rev>=0.5` → no split (bidireccional); `persistence<0.1` → no split
(parpadeo). El usuario eligió estas dos.

### (D) Quitar guard de persistencia del MERGE
**Por qué:** al bajar high, el guard de persistencia (que metimos en merge) frenó
12 merges de contención ALTA (0.82–1.00) con persist baja → regresión. El usuario
dijo "quita la persistencia de momento" (solo del merge; en partial se queda).

### (E) DESCRIPTOR como desempate en partial — la idea grande del usuario
**Problema:** en la franja 0.5–0.70 la geometría NO separa "fragmento a fusionar"
de "trozo a cortar" (caso 93→43: cont 0.55 unidireccional, claramente fragmento,
pero iba a split). **Regla del usuario:** "en la franja 50–70 entra el descriptor;
si es igual fusionamos, si no spliteamos."
**Fix:** `classify(loser, pairs, sim=callable)`. `sim(a,w)` = cosine de
`clip_feature` (lo construye `ovo.py:_contest_sim`). En partial: `sim>=sim_merge
(0.81)` → MERGE; si no → guards → SPLIT.

### (F) Extender el descriptor a DOMINANCIA — último cambio
**Problema:** 185→43 y 77→79 son el MISMO objeto adyacente (sim 0.92/0.81) pero con
contención baja (banda dominancia) → split, porque el descriptor solo estaba en
partial. **Fix:** misma lógica en la rama dominancia: `sim>=0.81` → MERGE; si no →
SPLIT.
**Validación:** los 5 splits que sobreviven tienen sim 0.76–0.79 (<0.81), incluido
el blob 89→103 (sim 0.76 → split, correcto). El umbral 0.81 cae justo en el hueco:
mismos-objeto 0.81–0.92, distintos 0.76–0.79.

---

## 4. Estado del código (archivos tocados)

- `ovo/entities/contest/discriminator.py`:
  - `__init__` nuevos params: `high=0.7`, `max_rev_split=0.5`,
    `min_persist_split=0.1`, `sim_merge=0.81`.
  - `classify(self, loser, pairs, sim=None)` — `sim` es callable `(a,w)->float|None`.
  - rama partial: descriptor → reverse guard → persistence guard → split.
  - rama dominancia: descriptor → split.
- `ovo/entities/contest/manager.py`:
  - `_reconcile` ELIMINADO → `_resolve_pairs` + `_join`.
  - `report(..., sim=None)` reenvía sim a classify.
  - wiring de los params nuevos desde `cfg` (bloque `contest:` de la config).
- `ovo/entities/ovo.py`:
  - `_contest_sim(a, w)` = cosine de `clip_feature` de las instancias; se pasa a
    `report(...)`. OJO: clip_feature puede ser None para instancias nuevas →
    devuelve None → cae a split (backward compatible).

**Config (bloque `contest:` bajo `semantic`)**: `high, low, min_mass,
min_split_cont, max_rev_split, min_persist_split, sim_merge`. Más arriba (semantic):
`contest_fusion: only|both|observe`, `contest_split_mode: partial|dominance|all|off`.

**SIN TESTS**: no hay tests unitarios del discriminador/resolver. Sería bueno
añadirlos (los casos pared/silla/blob/93/185 son fixtures naturales).

---

## 5. Herramientas construidas (`Study_seg/`, stdlib, sin deps)

Skill invocable: `/contest-data` (`.claude/skills/contest-data/SKILL.md`).

- `Study_seg/contest_csv.py` — analiza `contest_verdicts.csv`:
  - `summary CSV` · `pair CSV ID...` · `filter CSV --decision X --sort feat --top N`
    · `compare CSV_A CSV_B` (diff de veredictos entre runs — el más útil)
- `Study_seg/contest_json.py` — analiza `contest.json` (store crudo):
  - `summary` · `winner ID...` · `point PID...` · `hot --by winners|count` ·
    `pairmass A W` · `seen ID...` (proxy de veces-vista = max KF-count)

Para sacar similitud de descriptor o tamaños reales hay que ir al
`ovo_map.ckpt` (`map_params`: xyz/obj_ids/ids/obs; `ovo_map_params`:
`ins3d_<id>_clip_feature`). Ejemplos de extracción a lo largo de esta sesión.

---

## 6. Los runs (office3, escena única)

| id | qué | leer |
|---|---|---|
| `20260615_GT_CLIP_baseline-contest_6e7c8` | mecanismo viejo (reconcile, high=0.8) | baseline |
| `..._contest-split-partial-pia_5c701` | resolve por par + gate persistencia en merge | (revertido) |
| `..._contest-split-partial-pia_0f899` | sin gate de merge | |
| `..._contest-split-partial-pia_f5053` | descriptor en partial | |
| **`..._contest-split-partial-pia_66419`** | **descriptor en partial + dominancia — ÚLTIMO** | **ESTE** |

> **ÚLTIMO RUN A MIRAR:**
> `data/output/Replica/20260615_GT_CLIP_contest-split-partial-pia_66419/office3/`
> Visor: `rerun data/output/Replica/20260615_GT_CLIP_contest-split-partial-pia_66419/office3/rerun.rrd`

Manifest: `manifests/20260615_GT_CLIP_contest-split-partial-pia.yaml`
(stages: `[run]` solamente — ver §8 punto crítico).

### Efecto medido (veredictos, baseline → último)
| decisión | baseline | último (66419) |
|---|---|---|
| MERGE | 45 | **85** |
| SPLIT | 53 | **5** |
| NO_ACTION | 95 | 100 |

Casos validados con el usuario que ahora funcionan: pared 63/151 (MERGE), 93→43
(MERGE), 77→79 (MERGE), 185→43 (MERGE). Blob 89→103 sigue SPLIT (sim 0.76, correcto).

---

## 7. Trabajo futuro / gaps ABIERTOS (priorizado)

1. **⚠️ VALIDAR EL OVER-MERGE de dominancia.** El descriptor en dominancia metió
   +32 merges. Algunos dudosos: **31→22 mergeó** (sim>=0.81) pero el usuario dijo
   en el baseline que NO debían unirse. CLIP puede confundir muebles distintos
   (silla~mesa, "furniture"). HAY QUE EYEBALL los 31 merges nuevos en el visor
   (run 66419) antes de dar por bueno. Lista en `compare f5053 66419`.
2. **Recompute / 2ª pasada (caso 95).** 95 está 88.6% contenido en 78 pero salió
   NO_ACTION "frontera real" porque en el report sus puntos estaban repartidos
   entre 78+33+81 (3 instancias) que se fusionaron DESPUÉS en el mismo batch. Sin
   re-evaluación, 95 se queda fuera. Fix: aplicar merges → recomputar → re-evaluar.
3. **Union-find NO conectado** (`root=identidad` en el discriminador). La
   frontera-real cuenta IDs distintos, no objetos. Si pasáramos el find real de la
   fusión, 78/33/81 contarían como 1 raíz. Relacionado con #2.
4. **Feature que falta: cobertura = chunk/|W|** (chunk dividido por el tamaño del
   GANADOR, no de A). Distinguiría blob (cobertura≈1) de frontera (cobertura baja)
   sin descriptor. Discutida, no implementada. El `size` ya está en el aggregator.
5. **Medir CALIDAD real (mIoU/AP), no solo veredictos.** Todos los runs fueron
   `stages: [run]`. NUNCA corrimos `segment`+`eval`. No sabemos si los cambios
   mejoran el mapa de verdad. Añade `stages: [run, segment, eval, eval_instances]`
   y compara métricas vs baseline. **Esto es lo más importante que falta.**
6. **Solo 1 report** con `contest_fusion: only`. El "espera" (persistence guard) no
   tiene segunda oportunidad → en la práctica = NO_ACTION permanente. Y no hay
   acumulación temporal.
7. **DEFER_TO_FUSION muerto** en office3 (nunca dispara: requiere bidireccional
   alto, no se da con tamaños dispares).
8. **Solo office3.** Probar otras escenas Replica.
9. **Sin tests** del discriminador/resolver.
10. `Study_seg/flujo_decision_contest.md` tiene un mermaid embebido VIEJO (el .mmd
    suelto se borró). Sincronízalo con el .puml o bórralo.

---

## 8. Cómo continuar

```bash
# comparar último run con baseline
python3 Study_seg/contest_csv.py compare \
  data/output/Replica/20260615_GT_CLIP_baseline-contest_6e7c8/office3/contest_verdicts.csv \
  data/output/Replica/20260615_GT_CLIP_contest-split-partial-pia_66419/office3/contest_verdicts.csv

# inspeccionar un caso
python3 Study_seg/contest_csv.py pair <CSV> 31 22

# relanzar experimento (env ovo2)
conda run -n ovo2 python scripts/run_experiments_batch.py \
  --manifest manifests/20260615_GT_CLIP_contest-split-partial-pia.yaml --verbose
```

**Sugerencia de primer paso para el próximo agente:** eyeball con el usuario los
31 merges nuevos de dominancia (#1) — sobre todo 31→22 — para confirmar que el
descriptor no está sobre-fusionando. Y montar un run con `eval` (#5) para medir
impacto real en mIoU/AP. Todo lo demás (recompute, union-find, cobertura) son
mejoras estructurales que vienen después.
```
