# Contest — rediseño de firmeza + evaluador vs GT (traspaso 2026-07-07)

Documento de continuidad. Un agente que empiece de cero debería poder retomar el trabajo
solo con esto. Cubre: qué se cambió y por qué, hallazgos con números, la herramienta nueva
de evaluación, y el trabajo abierto.

---

## 0. Contexto y objetivo

El **mecanismo contest** (merge/split discriminator para mapeo semántico 3D de instancias)
estaba **sobreajustado a office4** (única escena de tuning previa). Regresaba en room1
(−15.4% AP_agnostic) y room2 (−6.0%). Objetivo: **simplificar el discriminador para que
generalice sin entrenar en todas las escenas**.

**Split train/held-out fijado** (memoria `contest-tuning-split-2026-07-07`):
- **Tuning** (se mira al ajustar): `office0, office3, office4, room2`. (2026-07-08: room2↔room1.)
- **Held-out** (solo para validar generalización): `office1, office2, room0, room1`.

Dataset: Replica, 8 escenas. Métrica: **AP_agnostic** (class-agnostic instance AP).
Valores en decimal [0,1].

---

## 1. Anatomía del contest (para entender el resto)

Flujo, todo en `ovo/entities/contest/`:

1. **Camino caliente** (cada KF, `ovo/entities/ovo.py::_track_objects`): registra en
   `ContestStore` (`store.py`) por punto: `record_sighting` / `record_claim` / `record_grab`.
   - Tres relojes: `sightings ⊇ claims ⊇ Σgrabs`. `grab` = punto cae bajo máscara de una
     instancia que NO es su dueño actual.
2. **Cadencia de fusión** (`manager.py::report`):
   - `ContestAggregator.pairs()` (`aggregator.py`) → convierte el store crudo en
     `PairFeatures` por par (defender, challenger).
   - `ContestDiscriminator.classify()` (`discriminator.py`) → un `Verdict` por defender.
   - resolve / reeval_frontier / clean_transfer → aplica al mapa.
3. **Sustrato de análisis** = `pre_fusion.ckpt` (owners PRE-fusión, lo que el contest vio),
   NO `ovo_map.ckpt` (post-fusión).

### Vocabulario de features (`types.py::PairFeatures`)
- `containment = firm_points / |defender|` — fracción del defender contenida en el challenger.
- `reverse_containment` — al revés.
- `firm_points` — nº de puntos del defender robados "con firmeza" por el challenger. **El átomo.**
- `persistence` — media de `grabs/claims` por punto (estabilidad temporal).
- `focus = firm_points / puntos disputados del defender`.
- `exclusivity` — puntos firmes sin otro challenger / firm_points.

### Bandas del discriminador (`discriminator.py::classify`) — ESTADO 2026-07-08 (post barrera focus)
Gate de masa (`total_grabs < min_mass=50` → NO_ACTION ruido), luego por containment:
- **strong** (`c ≥ high=0.6` **Y** `pers ≥ min_persist_strong=0.5`): 1 objeto→MERGE ·
  ≥2 objetos→NO_ACTION. (Sub-rama simétrica `reverse≥high`→DEFER **borrada**.)
  Reasons: `mostly contained in 1 object` · `mostly contained in several, no clear owner`.
- **partial** (`low=0.4 ≤ c < 0.6`): **regla nueva (fragment)** — MERGE si
  `pers ≥ partial_merge_persist(0.6)` **Y** `excl ≥ partial_merge_excl(0.8)` **Y** 1 raíz;
  si no, NO_ACTION. **Sin descriptor, sin split.** Reason: `fragment (...) -> merge`.
- **focused** (`c < 0.4`, `focus ≥ 0.7`): **barrera de firmeza** (2026-07-08) — SPLIT si
  `excl ≥ min_split_excl(0.9)` **Y** `pers ≥ min_split_persist(0.6)`; si no, NO_ACTION débil.
  **Sustituye a los guards geométricos seam/color (kNN caro), ya BORRADOS.** Reasons:
  `dominancia (...) -> split` · `dominancia débil (...) -> no split`.
- **borde** (fallthrough): NO_ACTION.

**SPLIT solo lo produce focused** (partial no splitea). Los guards seam/color y el guard muerto
`min_persist_split` se han eliminado del código (discriminator, manager, engine, ovo.py). La banda
ya no llama a `seam()`/`color()` → el kNN de normales/color NO se computa en producción.

---

## 2. EL CAMBIO PRINCIPAL: firmeza relativa (persistencia en el átomo)

### El problema
`firm_points` es el átomo de TODO (containment/focus/exclusivity se construyen contándolos).
El gate de firmeza original (`aggregator.py`) era **absoluto**: un punto contaba si el
challenger lo robó `≥ min_count (5)` KFs, **ciego a la persistencia**. Un punto robado 5/100
(roce marginal en zona transitada) contaba idéntico a uno robado 5/5 (del challenger de
verdad). El roce se colaba en el átomo e inflaba containment.

Había un guard de par `min_persist_split=0.1` en el discriminador para limpiar esto a
posteriori — pero **estaba muerto**: medimos la distribución y NINGÚN par en las bandas
partial/focused caía por debajo de 0.1 (min observado 0.124). Umbral calibrado en una escala
vieja (denominador `point_obs`, ya sustituido por `claims`) que ya no existe.

### La solución (idea del usuario, validada)
Mover la persistencia **al gate de firmeza**, no como guard de par. Un punto es firme si:

```
grabs >= min_grabs (5)   AND   grabs/claims >= firm_tau (0.30)
```

Evidencia (no es ruido de pocos KFs) **Y** compromiso (no es roce de refilón). Así
containment/focus/exclusivity se vuelven persistencia-aware gratis, en el origen.

### Cómo elegimos firm_tau=0.30
1. Medimos la distribución de `grabs/claims` por punto sobre 2.16M robos, 8 escenas
   (`scratchpad/point_persist_dist.py`). Los firmes actuales (c≥5) tienen mediana 0.64,
   pero cola baja (~20% < 0.2 = el leak). room1 (la que regresaba) tenía los firmes más
   grazy (mediana 0.24 vs 0.83 de office3) → señal de que `tau` la recortaría selectivamente.
2. Barrido `tau ∈ {0, 0.25, 0.30, 0.35, 0.40, 0.50}` por replay desde los mismos ckpts.

**Resultado (AP_agnostic global vs raw):**

| tau | global | Δ% | room1 | room2 | office4 | room0 |
|---|---|---|---|---|---|---|
| 0 (contest viejo) | 0.2406 | +3.3% | **−15.4%** ❌ | **−6.0%** ❌ | +25.8% | +9.3% |
| **0.25 / 0.30** | **0.2517** | **+8.0%** | **+4.9%** ✅ | **+2.6%** ✅ | +25.8% | +9.3% |
| 0.35 | 0.2502 | +7.4% | +1.5% | +2.6% | +25.8% | +9.3% |
| 0.40 / 0.50 | 0.2466 | +5.8% | +1.5% | +2.6% | +16.5% ↓ | +6.7% ↓ |

- `tau=0.25` y `0.30` **empatan** (meseta: ningún punto firme decisivo cae en [0.25,0.30)).
- **Duplica la ganancia** (+3.3% → +8.0%), **arregla las dos regresiones** (room1, room2),
  no toca los wins (office4, room0), **generaliza** (held-out media +5.5% vs +3.5%).
- `tau ≥ 0.40` erosiona office4/room0 → no subir de ahí.
- **Elegido `firm_tau=0.30`** (centro-alto de meseta, más conservador que 0.25, lejos de 0.40).

### Código del cambio
`ovo/entities/contest/aggregator.py::_is_firm`:
```python
def _is_firm(self, grabs: int, persistence: float) -> bool:
    return grabs >= self.min_grabs and persistence >= self.firm_tau
```
- `firm_tau=0` → `persistence>=0` siempre cierto → se reduce a solo evidencia (comportamiento legacy).
- **Default `firm_tau=0.30`** fijado en 3 sitios: `aggregator.py::__init__`,
  `manager.py:32` (`cfg.get("firm_tau", 0.30)`), `query/engine.py` (mismo default, para que
  la query tool coincida con producción).
- Renombrado `min_count` → **`min_grabs`** en todo (aggregator, manager, engine, cli, tui,
  types, manifests). `ovo.yaml` no tiene bloque contest → default por código.

### Refactor de aggregator.py (sin cambio de comportamiento, verificado escena a escena)
`pairs()` pasó de un método monolítico con 6 defaultdicts en paralelo a orquestador SRP:
- `_sizes()` — tamaño por instancia.
- `_contested()` — puntos disputados + dueño actual.
- `_is_firm(grabs, persistence)` — el gate (aquí vive la tesis).
- `_accumulate()` — bucle, llena un `_PairEvidence` por par (clase que encapsula sus
  contadores: `add_grab`, `add_firm`, `add_persist`).
- `_finalize()` — normaliza a `PairFeatures`.
- Diccionario clave: `evidence_by_pair: {(defender, challenger): _PairEvidence}`.
- Nombres descriptivos en todo (point/defender/challenger/grabs/persistence, sin `p/d/c`).

**IMPORTANTE:** el guard muerto `min_persist_split=0.1` en `discriminator.py` sigue ahí
(no lo quitamos aún). Con firm_tau la persistencia ya filtra en el átomo → ese guard es
doblemente redundante. **Candidato a borrar en la simplificación** (ver §5).

---

## 3. HERRAMIENTA NUEVA: evaluador de decisiones vs GT

`studies/contest_metrics/eval/` — califica la CALIDAD de cada decisión del contest contra
ground truth. Skill: `.claude/skills/contest-eval/SKILL.md`.

### Por qué, y la ventaja sobre fusion_metrics
`studies/fusion_metrics/` califica un CSV fijo de decisiones → probar un cambio exige
re-correr el experimento, y **solo evalúa merges de instancia entera** (no splits de
subconjunto). Nuestro evaluador se apoya en `ContestProbe` (`query/engine.py`), que
**recalcula veredictos desde el sustrato bajo cualquier config** → **contrafactual: cambiar
un gate y re-calificar SIN re-run**. Y maneja splits (subconjuntos).

### El primitivo point-set (unifica merge y split) — `eval/grade.py`
> Toda decisión reasigna un conjunto de puntos `S` de defender `D` a challenger `C`.
> `g(X)` = GT dominante sobre `X` (mayoría, void fuera). `belongs = g(S)==g(C)`. `moved` = MERGE|SPLIT.
> `moved&belongs=TP · moved&¬belongs=FP · ¬moved&belongs=FN · ¬moved&¬belongs=TN`. Sin GT medible → SKIP.
> **MERGE**: S = todos los puntos del defender. **SPLIT**: S = `split_points`. Merge es el caso S=todo-D.

### Piezas (SRP, calcado de fusion_metrics/core)
- `eval/gt.py` — `SceneGT` + `load_scene_gt`: proyecta GT (malla) por punto vía KDTree
  (vecino más cercano OVO→vértice GT). Assert de alineación vértices==etiquetas. `dominant_over_rows/ids`.
  - GT: `data/input/Datasets/Replica/{scene}_mesh.ply` + `instance_gt/{scene}.txt`.
    Codificación `id = clase*1000 + instancia`, `id<=0 = void`. (Open3D avisa de caras pero
    los vértices cargan completos — usamos solo vértices.)
- `eval/grade.py` — `grade_move`, `confusion`, `rates`, enum `Grade{TP,FP,FN,TN,SKIP}`, dataclass `GradedDecision`. Puro, cero I/O.
- `eval/pipeline.py` — `grade_scene`/`grade_scenes`: probe → GT → grade. `SCENE_GROUPS`
  (tuning/held/all). Incluye **barrido de missed-transfers** (pares firmes cuyo trozo GT =
  challenger ≠ defender y NO se movió — el FN que el nivel-veredicto no ve por winner-take-all).
- `eval/report.py` — tablas puras (confusión, por-gate, missed) + `to_payload` (--json).
- `cli.py` — dominio `eval` (`_parse_overrides` ahora tolera strings, p.ej. `contest_split_mode=all`).

### Uso
```bash
conda run -n ovo python -m studies.contest_metrics eval --exp <id> --scenes tuning
conda run -n ovo python -m studies.contest_metrics eval --exp <id> --scenes tuning --set firm_tau=0.4
conda run -n ovo python -m studies.contest_metrics eval --exp <id> --scene office4 --json
```

### Hallazgos (tuning, firm_tau=0.30, sobre ckpts de 32ac4)
POOLED: `TP=68 FP=5 prec=0.932` · **missed transfers: 315**.

Por gate:
| gate | TP | FP | lectura |
|---|---|---|---|
| contención alta, 1 raíz (merge) | 57 | 3 | núcleo sólido |
| **parcial (split)** | **2** | **2** | **cara o cruz — señal basura, candidato #1 a cortar** |
| parcial + descriptor igual (merge) | 6 | 0 | limpio |
| dominancia (focused split) | 3 | 0 | limpio en tuning |

**Dos conclusiones fuertes:**
1. El gate **"parcial" es ruido** (50% FP). Cortarlo o rediseñarlo.
2. **315 missed transfers** = el contest es **demasiado conservador** en splits. Explica el
   **AP_50 plano** (los splits casi no ocurren → no afinan límites). room1: chunks de
   700/657/613 pts sin transferir. El winner-take-all + umbrales altos dejan casi todo sobre la mesa.
3. office4 tiene **1 FP concreto**: `def 92 → ch 82` (merge "contención alta, 1 raíz"),
   GT(92)=87004 ≠ GT(82)=10 → over-merge. Investigable con `query pair`.

**Caveat honesto:** "missed" cerca de bordes de instancia puede tener sangrado GT; los
grandes (cientos de pts) son reales. FN=0 en el nivel-veredicto es artefacto (winner-take-all
+ borde sin challenger→SKIP), no virtud — por eso el missed-scan es aparte.

---

## 4. Inventario de experimentos y comandos

Envs: **`ovo2`** para el runner (`scripts/run_experiments_batch.py`), **`ovo`** para las tools
(`studies.contest_metrics`, y `textual` para la TUI está en ovo2).

Experimentos (en `data/output/Replica/`):
- `20260707_GT_CLIP_raw-baseline-full_32ac4` — mapa CRUDO (classic_fusion:false, sin ruido),
  **guarda los pre_fusion.ckpt** en `data/checkpoints/Replica/…_32ac4/{scene}/`. Sustrato de todo.
- `20260707_GT_CLIP_contest-only-full_fcfe8` — contest tau=0 (viejo).
- `…_contest-tau025_7fee4`, `…_tau030_7862e`, `…_tau035_ea3b0`, `…_tau040_bef5a`, `…_tau050_4091c` — barrido (replay).

Scripts de análisis en scratchpad (efímeros, referencia):
`persist_dist.py` (distribución por par/banda), `point_persist_dist.py` (por punto),
`compare_sweep.py` (AP 7 bandas tuning/held), `eval_phase1.py` (prototipo del evaluador).

Comparar AP entre dos exps: leer `data/output/Replica/<exp>/replica/instance_ap_<scene>.txt`
(línea `AP_agnostic, <val>`).

### Gotchas (críticos)
- **El runner fuerza `noise_enabled: true`** ante cualquier bloque `noise:` no-jump
  (`run_experiments_batch.py:192`). Para GT limpio hay que poner `noise_enabled: false` explícito.
- **rerun `fusion`/`loop_closure` NO graban sin evento de loop closure** → rrd de 91KB.
  Para GT limpio usar **`rerun_mode: stream`** (rrd ~890MB, graba todo).
- `close_loops: true` en GT limpio solo dispara la fusión final; el `pre_fusion.ckpt` se
  guarda justo antes.

---

## 5. TRABAJO ABIERTO / intenciones futuras

Objetivo global sigue siendo **simplificar el discriminador** (`discriminator.py`) para
que generalice. Con el evaluador vs GT ya tenemos el instrumento para hacerlo con datos.

### RESUMEN CONFIG FINAL (2026-07-08)
```
strong:  c ≥ high(0.6)  → firm=[pers≥0.5]: 1 raíz MERGE / ≥2 NO_ACTION / sin firme NO_ACTION
         (pertenencia SOLO por containment; la persistencia decide DENTRO, no se fuga a focused)
partial: low(0.4) ≤ c < 0.6, pers ≥ 0.6 Y excl ≥ 0.8 Y 1 raíz → MERGE (fragment); sin split
focused: c < high, focus ≥ 0.7 (y no capturado por strong/partial) →
         BARRERA excl ≥ 0.9 Y pers ≥ 0.6 → SPLIT / si no NO_ACTION débil (2026-07-08, sin seam/color)
firm_tau=0.30, min_grabs=5, min_mass=50, min_split_focus=0.7, min_split_excl=0.9, min_split_persist=0.6
umbrales del discriminador ahora en `@dataclass ContestThresholds` (discriminator.py), from_cfg (§7.10)
```
BORRADO: rama DEFER (simétrica), guards partial (bidireccional `max_rev_split`, inestable
`min_persist_split`), param `sim_merge` (descriptor fuera de partial), **guards focused seam/color
(kNN normales+color) y `max_seam_angle`/`min_persist_split` (2026-07-08)** — ver §7.7.
Runs de respaldo: `contest-strong-v2` (high=0.6+piso), `contest-partial-fragment` (guards),
`contest-focus-barrier` (barrera, config actual). AP_agnostic media: raw 0.2330 · tau030 0.2518
(+8.0%) · partial 0.2481 (+6.5%) · **barrier 0.2477 (+6.3%)**. Barrera AP-neutral vs guards
(idéntica 7/8, room1 −0.003) → guards seam/color no ganaban AP, fuera gratis + sin kNN (§7.9).
Split train/held: tuning=office0/office3/office4/**room2**,
held=office1/office2/room0/**room1** (room2↔room1 cambiadas 2026-07-08).

### Escala de decisiones (lección de la sesión)
1. **Calidad-decisión (eval TP/FP) ≠ AP.** El eval es proxy; la verdad es re-correr y medir AP.
2. **No combinar cambios sin re-medir la interacción** (high×piso se solaparon y ocultaron culpa).
3. **La proyección GT miente con challengers impuros** (dominante global). Validar dudas en el
   visor `debug_merge_decisions` (fusion_metrics). Varios "FP" de strong/partial = artefacto GT.

### HECHO (2026-07-07, sesión de afinado de strong)
- **Auditoría de strong** (exp `…contest-tau030-viz_68083`, 8 escenas): 94 TP / 9 FP.
  Los 9 FP son casi todos defendibles (gigante que absorbe objeto chico encajado, o dos
  instancias de la misma clase pegadas). **Strong MERGE 1-raíz se da por bueno, no se toca.**
  Los FP se agrupan en 2 tipos: (A) `excl≈0` gigante sucio (office2→50, room1 19→22);
  (B) `excl≈1` contención limpia + identidad errónea (office4 92→82, room1 65→22). No urge guard.
- **BORRADA la sub-rama simétrica `reverse_containment≥high → DEFER_TO_FUSION`**
  (`discriminator.py`, antes líneas 61-67). Razón doble: (1) empírica — 0/104 disparos en
  8 escenas, `reverse≈0.00` siempre porque el merge es chico-dentro-de-grande → estructural;
  (2) diseño — dos instancias totalmente superpuestas = fallo de **tracking**, no de
  segmentación → responsabilidad de otro mecanismo, no del discriminador. Verificado con
  `eval --scenes all`: confusión **idéntica** (TP=115 FP=13 FN=1 TN=1). Enum `DEFER_TO_FUSION`
  se conserva (tipo válido reusable), pero ya no lo produce nadie.
- **Piso de persistencia en strong (`min_persist_strong=0.5`)** — para contar como dueño
  "strong" un challenger necesita `containment ≥ high` **Y** `persistence ≥ 0.5` (agarra cada
  punto ≥ la mitad de las veces = mayoría). Filtro en la lista strong (`discriminator.py`),
  wired en manager + engine + `_thresholds`. **Dos efectos con una sola regla:**
  1. **Arregla el caso office3 def 91** (trozo de mesa entre 2 challengers strong): 103 (mesa,
     dueño real, `pers=0.67`) y 89 (silla **mal segmentada** solapada, `pers=0.34`). El piso
     descarta a 89 (tibio) → 91 queda con 1 solo dueño → MERGE 91→103, **TP** por GT. Esto
     **sustituye** al tie-break ad-hoc `frontier_persist_ratio=1.5` (n=1, calibrado a ojo) que
     se probó antes y se descartó — el piso es principiado, no un umbral tallado.
  2. **Guardarraíl de robustez**: un merge respaldado por `pers<0.5` (dueño que agarra en
     minoría) no es fiable → se rechaza. Cae 1 merge así: room0 68→16 (`pers=0.39`, GT
     76009→76009, era TP) → ahora NO_ACTION (FN). **Coste asumido a propósito**: 0.39 es
     genuinamente flojo, no fiarse es lo correcto.
  `eval --scenes all`: **TP=115 FP=13 FN=2** (vs 116/13/1 del tie-break descartado). Swap neto:
  +91 (frontera resuelta) −68 (merge tibio rechazado). El gate `mostly contained in 1 object`
  mantiene 94 TP/9 FP (entra 91, sale 68). **NO se descartó ningún FP** (los 9 siguen — son
  artefacto del grader, ver abajo), pero tampoco era el objetivo: esto es securización, no caza-FP.
- **`high` bajado 0.7 → 0.6** (decisión: lado conservador). Con el piso de persistencia puesto,
  bajar high **no es neutro** (el barrido previo que decía "neutro" era SIN piso): pares en
  `[0.6,0.7)` con `pers<0.5` (p.ej. office4 33→22 pers 0.46, room0 115→83 pers 0.32) entran a
  strong pero el piso los expulsa → no se fusionan (antes los rescataba parcial+descriptor).
  `eval`: **TP 115→113, FP 13 igual, FN 2→3**. Se pierden 2 merges de baja persistencia
  **a propósito**: el usuario prefiere no fusionar con evidencia tibia (recuperarlos más adelante
  con otra señal). Efecto secundario bueno: encoge el gate parcial (franja ahora `[0.5,0.6)`).
  Lección: **no combinar cambios sin re-medir su interacción** (piso × high se solapan).
- **`high` bajado a 0.6 (decisión final, lado conservador).** El A/B por AP lo confirmó:
  `high=0.6` cuesta AP (v2: office4 0.275 vs 0.297, room1 0.352 vs 0.361) porque saca pares del
  gate parcial-descriptor y el piso los expulsa de strong → huérfanos (cont≥0.6 pero pers<0.5 no
  caben en `[0.5,0.6)`). v3 (high=0.7+piso) recupera todo → **el high era el culpable, el piso es
  gratis (solo room0 −0.002)**. El usuario **elige 0.6 igualmente**: no fusionar con evidencia
  tibia > +AP. **Sigue +6.3% vs raw** (office4 +16.5%, todas las escenas positivas). Runs: v2
  (`contest-strong-v2`) respalda la config; v3 borrado (era solo para aislar).
- **Guards muertos de la banda parcial BORRADOS** (bidireccional `rev≥max_rev_split` + inestable
  `persist<min_persist_split`). Medido: **0/21 disparos** en 8 escenas (el inestable estructural-
  mente muerto por firm_tau; el bidireccional vacío en Replica). La banda parcial queda binaria:
  `sim≥0.81→MERGE · else→SPLIT`. Param `max_rev_split` **eliminado** (era su único uso);
  `min_persist_split` se conserva (aún lo usa la banda focused, también muerto ahí → pendiente).
  `eval`: confusión **idéntica** (TP=113 FP=13 FN=3).
- **Reasons de strong reescritos a inglés** (vocabulario). Eje único "cuán contenido + hay dueño":
  `mostly contained in 1 object (cont) -> merge` · `mostly contained in several, no clear owner -> no action`.
  Actualizado el filtro `_reeval_frontier` (`manager.py`, matchea `"mostly contained in several"`) y
  docstrings. `report.py::gate_key` agrupa por prefijo → confusión idéntica. Resto de bandas
  (parcial/dominancia/borde) siguen en español — migración pendiente.
- **Hallazgo: los 9 "FP" de strong son en gran parte artefacto del grader.** El grader compara
  `g(defender)` vs el **dominante GLOBAL del challenger**. Cuando el challenger está corrupto
  (instancia mezclada), ese dominante miente. Caso disecado: office2 **50** = 60% GT69 (suelo) +
  **39% GT87008** (medio objeto absorbido). Merges 100→50 y 103→50 (GT87008) se marcan FP porque
  `g(50)=69`, pero localmente donde tocan 50 **es** 87008 (visor debug_merge_decisions: verde =
  mismo objeto). Son **merges razonables** (consolidan el objeto 87008 disperso), no over-merges.
  `exclusivity≈0` en ellos = zona con 2 gigantes (50 y 95) robando los mismos puntos, no señal de
  mal merge. **Pendiente (§eval v2):** calificar merge contra GT de la región de solape, no del
  challenger global → los FP fantasma desaparecerían.

### HECHO (2026-07-08, sesión de rediseño de la banda parcial)
- **BANDA PARCIAL REDISEÑADA — de descriptor a firmeza.** Análisis: en `[0.4,0.6)` el descriptor
  (sim) es mal juez — pierde 4 merges de mismo-objeto con sim 0.75-0.78, y con sim alto se deja
  engañar (office0 127→13 diff con sim 0.88). **Persistencia separa mucho mejor** (validado con
  inspección visual del usuario: 10 merges pers∈[0.62,1.0] vs 25→0 "enchufe/pared" pers 0.49).
  Nueva regla: MERGE si `pers ≥ 0.6 Y excl ≥ 0.8 Y 1 raíz`; si no, NO_ACTION. **Sin split, sin
  descriptor.** Params `partial_merge_persist=0.6`, `partial_merge_excl=0.8`; `low` 0.5→0.4;
  `sim_merge` eliminado. Las 3 señales cubren fallos distintos: **excl** mata zonas multi-disputadas
  (mismo punto varios grabbers, office2→50), **raíz única** mata reparto por mitades (dos dueños
  exclusivos), **pers** valida que el dueño es real (enchufe 25→0 fuera). Simulación tuning: 10/10
  correctos (los 2 "FP" GT = artefacto, visualmente SAME); held: 3 TP / 0 FP. Counterfactual eval:
  TP 113→**116**, FP 13→**12**.
- **AP del rediseño (`contest-partial-fragment`, replay 32ac4): NEUTRO (+0.2% vs v2).** office4
  +0.005, room2 +0.003, office3 +0.001 (tuning ↑), room0 −0.005 (held ↓). Los fragments son
  merges correctos pero **trozos pequeños** → no mueven AP de instancia. **Valor = simplificación +
  robustez, no AP.** Sigue +6.5% vs raw. Se **mantiene** (código más limpio, no cuesta AP).
- **room0 −0.005 diseccionado:** los 3 merges nuevos son correctos (SAME). La pérdida = (a) quitar
  el split parcial (transferencias TP menores) y (b) **10→70** (objeto 98060, dos trozos grandes,
  merge REAL) que la vieja fusionaba por descriptor (sim 0.88) y la nueva rechaza (pers 0.54<0.6).
  **El descriptor tenía señal complementaria.** Opción pendiente (no implementada, usuario dijo
  dejar): descriptor como 2ª vía de merge — `(pers≥0.6 Y excl≥0.8) O (sim≥0.81 Y excl≥0.8)`.
  Recupera 10→70 (sim 0.88) sin meter 118→4 (sim 0.78, challenger impuro 46/44). Discrimina exacto.
- **`low` NO baja de 0.4 (medido).** A `low=0.35` la franja `[0.35,0.4)` da **2 TP / 3 FP** en
  tuning: la regla se rompe porque a baja contención un par muy persistente puede ser un **vecino
  pegado de la misma clase** (FP: 93026→93024, 93008→93056, pers 0.63-0.94), no un fragmento.
  pers/excl dejan de discriminar. **0.4 es el suelo** (donde "fragmento" aún manda sobre "vecino").

- **STRONG reestructurado — persistencia DENTRO, no como filtro de pertenencia.** Antes
  `strong = [cont≥high Y pers≥0.5]` → un par `cont≥0.6, pers<0.5` quedaba fuera de strong Y de
  partial (`[0.4,0.6)`, cont demasiado alto) → **se fugaba a focused** (que solo mira focus). Feo.
  Ahora: `strong = [cont≥high]` (pertenencia solo por containment); dentro, `firm = [pers≥0.5]`:
  1 raíz firme→MERGE, ≥2→NO_ACTION, sin dueño firme→**NO_ACTION "mostly contained but no firm
  owner"** (no se fuga; algo mayormente contenido no es un trozo que transferir). Los 4 huérfanos
  (office4 33→22, room0 68→16/115→83, room1 80→33) son **todos same-object** (merge ideal, bloqueado
  por piso). Eval: TP 116→115 (−1: focused rescataba 1 con split casual), **FN 3→1**, F1 0.939→0.947.
  Compromiso aceptado: diseño limpio > 1 split casual. El piso bloqueando same-object de baja
  persistencia (estos + room0 10→70) queda como palanca aparte (bajar piso / descriptor 2ª-vía).

### HECHO (2026-07-07, sesión de afinado de strong) [detalle abajo]

Orden sugerido:
1. ~~**Cortar/rediseñar el gate "parcial"**~~ **HECHO** (ver arriba: rediseñado a fragment por
   firmeza, sin split). Queda: opción descriptor-2ª-vía-de-merge (pendiente, recupera 10→70).
2. ~~**Borrar el guard muerto `min_persist_split=0.1`** de partial~~ **HECHO en partial** (junto
   al bidireccional; ver HECHO). **Pendiente en focused** (`discriminator.py:~125`): mismo guard,
   también muerto (0/27 con firm_tau) → borrar y verificar `eval`. Si queda huérfano el param
   `min_persist_split`, eliminarlo también.
2b. **Descriptor como 2ª vía de MERGE en partial** (analizado, NO implementado — usuario dejó para
   luego). `(pers≥0.6 Y excl≥0.8) O (sim≥0.81 Y excl≥0.8)` + 1 raíz. Recupera room0 10→70
   (objeto 98060, merge real, pers 0.54 pero sim 0.88) sin meter 118→4 (sim 0.78, challenger
   impuro). El descriptor tiene señal complementaria a la persistencia (cazan casos distintos).
3. **Los 315 missed transfers** + **split por firmeza**: ahora **solo focused produce split**
   (partial ya no). El split sigue tímido. Al quitar el split de partial se perdieron transferencias
   TP menores (room0). Idea: un **split por firmeza** simétrico a la regla fragment — transferir un
   trozo (no instancia entera) si es persistente+exclusivo hacia otro dueño. Ataca AP_50 plano.
4. **Reconciliar `clean_transfer`** (`clean_transfer.py`, flag `clean_transfer`, default off):
   usa persistencia 0.5 + exclusividad 0.7 por par — otra señal de persistencia descoordinada
   del `firm_tau=0.30` del átomo. Decidir si sobrevive o se subsume.
5. **Simplificar bandas**: el aparato de guards geométricos (seam_angle, color 2-sentidos) de
   la banda focused está tallado en office3/4. Medir con `eval` cuánto aporta cada guard
   (contrafactual `--set max_seam_angle=…`) y cortar lo que no generalice.
6. **eval v2 — grade local + missed-moves como FN real**: (a) el grade de MERGE compara
   `g(defender)` vs dominante **global** del challenger → miente con challengers corruptos
   (ver Hallazgo en HECHO: los 9 "FP" de strong son casi todos artefacto de esto). Calificar
   contra el GT de la **región de solape** (los firm_points), no del challenger entero. (b) el
   barrido missed es informativo aparte → integrarlo como FN en la confusión (grade a nivel par,
   no solo veredicto ganador). Ambos = mismo refactor: bajar el grade al nivel de conjunto-de-puntos.
7. ~~**DEFER_TO_FUSION**: decidir bucket propio.~~ **Moot** — la rama que lo producía se
   borró (ver HECHO). El enum queda sin productor; ningún veredicto es DEFER ya.
8. **viz para eval**: donut de confusión, barras de impacto por gate (studies/contest_metrics/viz).

Regla de oro: **tunear en `--scenes tuning`, confirmar en `--scenes held`**. Nunca ajustar
umbrales mirando held.

---

## 7. SIGUIENTE ETAPA — rediseño del SPLIT (banda focused) — ARRANQUE INMEDIATO

### 7.0 El giro de eje (leer primero)
El discriminador tiene dos mitades espejo:
- **MERGE (HECHO)** — eje **containment**. "¿El defender cabe DENTRO de un dueño?" → absórbelo.
  Lo cubren strong (c≥0.6) + partial fragment (0.4-0.6). Ambos por firmeza (pers/excl), sin descriptor.
- **SPLIT (SIGUIENTE)** — eje **focus**. "¿El defender es un objeto grande que RETIENE un trozo de
  un vecino?" `focus = firm / puntos_disputados_del_defender`. containment bajo (el trozo es chico
  vs el defender entero) pero focus alto (ese trozo va limpio a un vecino) → **corta el trozo y
  dáselo** (transfer). Ej: una mesa que arrastra puntos de una silla → transfiere esos puntos a la silla.

### 7.1 Diagnóstico (medido esta sesión, exp `20260707_GT_CLIP_contest-tau030-viz_68083`)
- La banda **focused produce 4 splits en 8 escenas, los 4 TP** (precisión perfecta) + 2 NO_ACTION.
  Por escena: office0/office3/office4/room0 = 1 split; office1/office2 = 1 no_action; room1/room2 = 0.
- **643 missed transfers** (trozos firmes cuyo GT = un challenger ≠ defender, y NO se movieron).
  → focused resuelve ~0.6% de las oportunidades. **No falla por imprecisión, falla por OMISIÓN.**
- Causa: **winner-take-all por defender** (cada defender elige UN challenger) + **guards geométricos
  overfit** (seam_angle 15°, color 2-sentidos, min_persist_split 0.1 — tallados a mano en office3/4).
  → explica el **AP_50 plano** (los límites no se afinan porque casi no hay transferencias).

### 7.2 El plan (aplicar al split lo mismo que al merge)
1. **Split POR PAR, no por defender.** Hoy `focused = max(pairs, key=focus)` → un solo challenger
   por defender. Cambiar a: transferir CADA trozo firme+exclusivo a su dueño real, aunque el
   defender tenga varios vecinos mordiéndolo. Ataca directamente los 643.
2. **Tirar los guards geométricos** (seam/color) y decidir por **firmeza del trozo**: `pers ≥ X Y
   excl ≥ Y` de los `split_points` que se transfieren. Trozo persistente+exclusivo hacia un vecino
   = transferencia legítima; roce tibio = no. Mismo eje que la regla `fragment`.
3. **Guard de raíz/misma-clase**: OJO — a baja contención los vecinos de la MISMA CLASE (2 sillas,
   2 objetos 93xxx) se roban mutuamente con pers alta (visto en el barrido `low<0.4`: FP 93026→93024,
   93008→93056 con pers 0.63-0.94). El split debe evitar transferir entre instancias que son objetos
   distintos de la misma clase. Señal a explorar: el trozo cambia de GT? (no tenemos GT en producción)
   → quizá geometría mínima o el propio focus + tamaño del trozo.

### 7.3 PRIMER ANÁLISIS a correr (antes de tocar código)
Distribución de los 643 missed: **tamaño (firm_points), persistence, exclusivity**, tuning vs held.
Objetivo: ver qué umbral de firmeza separa transferencias reales (trozos grandes, persistentes,
exclusivos) del ruido de borde (trozos chicos). El scan de missed está en
`studies/contest_metrics/eval/pipeline.py::_scan_missed` — reutilizar su lógica:
```python
# criterio actual del scan (pipeline.py):
#   para cada par firme f no movido: gt_chunk=dom(f.split_points), gt_ch=dom(challenger),
#   gt_def=dom(defender); cuenta si gt_chunk==gt_ch != gt_def.
# CAVEAT: usa dominante por MAYORÍA -> trozos de borde con 51% se cuelan. Añadir PUREZA
#   (fracción del trozo que es ese GT) y filtrar por >=0.9 para separar reales de sangrado.
```
Patrón de script (env `ovo`): `ContestProbe.from_experiment(exp, scene)` da `p._pairs` (PairFeatures
con `.split_points`, `.persistence`, `.exclusivity`, `.firm_points`); `load_scene_gt(scene, p._map)`
da `gt.gt_labels` (alineado con `p._map.points_ins_ids`) y `gt.dominant_over_ids(split_points)`.

### 7.4 Herramientas / comandos (cheat-sheet)
- **eval** (calidad decisión + missed): `conda run -n ovo python -m studies.contest_metrics eval
  --exp <id> --scenes all|tuning|held` · `--set K=V` (contrafactual sin re-run) · `--json`.
  Reporta confusión por gate + "missed transfers totales".
- **query** (una decisión en vivo): `... query pair A B --exp <id> --scene <sc>` (ambos sentidos +
  señales + veredicto resuelto + umbrales) · `query list` · `query branch` · `query point <p>`.
- **GT visual** (validar dudas, artefactos de proyección): `cd studies/fusion_metrics && python -m
  scripts.debug_merge_decisions --exp_path <ABS ruta output> --scene <sc> --z_max 1.5`, teclear `def ch`.
  rojo=A azul=B, verde=mismo GT. Ids = obj_id del ckpt (casan con ContestProbe).
- **experimento AP** (la verdad): manifest replay desde sustrato 32ac4 (ver
  `manifests/20260708_GT_CLIP_contest-partial-fragment.yaml` de plantilla) →
  `conda run -n ovo2 python scripts/run_experiments_batch.py --manifest <m> [--preview]` →
  AP en `data/output/Replica/<exp>/replica/instance_ap_<scene>.txt` (línea `AP_agnostic, <val>`).
  Baseline raw = `…raw-baseline-full_32ac4` (sin contest). Comparar por escena, tuning vs held.
- **Sustrato**: los `pre_fusion.ckpt` viven en `data/checkpoints/Replica/…_32ac4/{scene}/`.

### 7.5 Reglas de oro (no repetir errores de esta sesión)
- **Calidad-decisión (eval TP/FP) ≠ AP** → toda hipótesis se valida con replay + AP, no solo eval.
- **No combinar cambios** sin medir su interacción por separado (high×piso nos mordió).
- **Tunear en tuning (office0/office3/office4/room2), confirmar en held (office1/office2/room0/room1).**
- **La proyección GT miente con challengers impuros** → dudas al visor.
- Env: `ovo` para tools (studies.contest_metrics), `ovo2` para el runner.

### 7.6 RAMA APARCADA (2026-07-08): firmeza-del-par vs focus — decisión pendiente
Exploramos saltar de **focus** (por-defender) a **firmeza del par** (firm/excl/pers del trozo) y
lo aparcamos para volver a estudiar **focus primero**. Lo medido, para no reconstruir:

- **Embudo dominancia (banda focused), 8 escenas** (`scratchpad/dominancia_funnel.py`): solo
  **6 defenders entran, 4 SPLIT, 2 NO_ACTION (guard color)**. persist y seam **nunca dispararon**
  (guards muertos). room2 = 0 splits (max focus 0.60 < 0.7). La puerta `focus≥0.7` es el cuello.
- **Por qué casi no dispara** (`scratchpad/dominancia_pairs.py`): `focus = firm_par / TODO lo
  disputado del defender`. Un defender mordido por varios vecinos reparte → focus minúsculo
  (mediana ~0.07, p90 ~0.2-0.3). focus≥0.7 solo lo alcanza un objeto disputado por UN challenger.
  → los 4 que disparan son single-challenger-dominated; el resto (646 missed) nunca llega.
- **Dos ejes = dos problemas distintos:**
  - **focus** (actual) → problema de **recall** (dispara 4×/8 escenas, precisión perfecta).
  - **firmeza del par** → problema de **precisión** (office4: 11 candidatas con firm≥50/excl≥0.9/
    pers≥0.45, solo **6/11 GT-buenas** = ~55%). Cambia recall por precisión.
- **office4 embudo por-par** (`scratchpad/office4_funnel.py`): 253 con split → cont<0.4 → masa≥50
  (213) → firm≥50 (50) → excl≥0.9 (35) → pers≥0.45 (**11**, 6 GT-buenas). pur/leaves usan GT (solo
  validación, NO producción). Umbrales arbitrarios; bajarlos (firm≥5 pers≥0.3) explota el conteo.
- **El seam guard NO discrimina** (`scratchpad/seam_check.py`): giro de normal 1-12° para TODOS
  (fumada y buenas). Lógico: el trozo se elige como los puntos que el challenger roba → por
  construcción se apoyan en su superficie → siempre alineados. Solo cazaría cojín-sobre-sofá.
- **Caso 30→45 office4** (la duda que decide todo): trozo 297 pts, firm 297 pers 0.65 excl 1.0,
  z[-1.16,-1.08] (a ras de suelo), **normal horizontal (nz 0.09) = superficie vertical**. GT dice
  trozo=pared 93056, def=suelo 40007. Usuario lo leyó "fumada suelo/pared" a ojo pero geometría +
  GT sugieren que son puntos de la BASE de la pared mal asignados al suelo → transfer podría ser
  correcto. **No resuelto** — hace falta ver los 297 pts concretos en visor, no las instancias
  enteras. Herramienta lista: `scratchpad/export_chunk_ply.py <def> <ch>` → PLY (def rojo, ch azul,
  trozo amarillo, contexto gris).
- **PREGUNTA ABIERTA que decide la rama:** ¿los 646 missed son transferencias reales (→ firmeza
  del par + trabajar el guard estructural) o mayormente smears/mask-bleed (→ focus tenía razón)?
  Se responde validando en visor una muestra de las 11 de office4 con su PLY.

### 7.7 HECHO (2026-07-08): guards focused → barrera de firmeza (excl/pers), fuera seam/color
Sustituidos los guards geométricos de la banda focused por una barrera barata sobre señales que
ya están en `PairFeatures`. Motivo: el kNN de normales (seam) + color es caro, y midiendo su
impacto real resultó peor que la barrera.

- **Embudo focus, 8 escenas** (`scratchpad/focus_funnel.py`, `focus_entrants.py`): solo **6
  defenders entran** (focus≥0.7). Con guards: 4 SPLIT, 2 NO por color, 0 por seam, 0 por persist.
  **seam nunca disparó** (giros 3-9°, umbral 15°) = peso muerto. Los 2 kills de color:
  - office1 `45→43`: firm 258, excl 1.0, pers 0.94. color lo mató por pelos (ΔE_ch 1.7 vs def 1.1).
    GT dice BUENO → **color tiraba una transferencia real**.
  - office2 `95→50`: firm 19143, excl 1.0, **pers 0.49**. color lo mató con margen (10 vs 2.5).
    GT sucio (pur 0.69) dice malo. La barrera lo mata igual por pers<0.6.
- **Barrera elegida (usuario): `excl ≥ 0.9` Y `pers ≥ 0.6`.** Sobre los 6, reproduce las 4 buenas,
  **recupera office1 45→43** (color la mataba) y **mata office2 95→50** (pers 0.49). Estrictamente
  ≥ que los guards en lo medido, y sin kNN. Verificado (`scratchpad/verify_new.py`): 5 SPLIT + 1
  NO_ACTION débil, reasons correctos. Eval office4 OK (dominancia 1 TP / 0 FP).
- **Aviso de margen:** el rey office4 `82→17` tiene pers **0.63**, a 0.03 del corte 0.6. Si en otra
  config el mejor split cae <0.6, la barrera lo mata. `pers≥0.5` sería más seguro; `≥0.6` es más
  agresivo con office2. Trade elegido: 0.6.
- **Código tocado:** `discriminator.py` (params `min_split_excl=0.9`/`min_split_persist=0.6`, borrados
  `min_persist_split`/`max_seam_angle`, classify sin seam/color), `manager.py` (report/classify sin
  seam/color, cfg nuevos), `query/engine.py` (cfg + _thresholds + classify), `ovo.py` (borradas
  closures `_contest_seam`/`_contest_color`, report solo con sim). `callbacks.py` conserva seam/color
  (el query tool los define pero ya no se llaman) — candidato a limpiar.
### 7.8 Barrido del umbral focus (`min_split_focus`) — DECISIÓN: queda en 0.7 (2026-07-08)
Antes de tocar el umbral de entrada de la banda, se parametrizó `focus≥0.7` (antes hardcodeado)
como `min_split_focus` y se barrió 0.7→0.2 con el classify REAL (barrera excl/pers dentro).
`scratchpad/focus_sweep.py` (conteo) + `focus_recovers.py` (qué entra en cada escalón).
- **La barrera sola NO salva la precisión al bajar focus.** SPLITs / GT-buenos / GT-malos:
  tuning 0.7→3/2/1(=82 miente) · 0.6→4/3/1 · 0.5→6/3/3 · 0.4→11/5/6 · 0.2→18/10/8.
  held 0.7→2/2/0 · 0.6→2/2/0 · 0.5→2/2/0 · 0.4→4/3/1 · 0.2→13/5/8.
- **0.6 recupera 1 solo split**: `room2 2→32`, firm **9 pts** (masa ridícula, no mueve AP). Held
  no gana nada. **0.5** ya mete 2 GT-malos en tuning. ≤0.4 = régimen firmeza-del-par (~55% prec).
- **Conclusión: 0.7 se queda.** Bajarlo no da masa buena sin meter FP; el techo de focus es real
  (las buenas gordas viven en focus 0.05-0.2 mezcladas con smears). Recuperarlas exige otro eje.

### 7.9 AP MEDIDO (replay 32ac4) — la barrera es AP-NEUTRAL vs guards (2026-07-08)
Run `manifests/20260708_GT_CLIP_contest-focus-barrier.yaml` (= partial-fragment + params barrera).
AP_agnostic media, 8 escenas:

| config | media | vs raw |
|---|---|---|
| raw (sin contest) | 0.2330 | — |
| tau030 (viejo, guards) | 0.2518 | +8.0% |
| partial-fragment (guards) | 0.2481 | +6.5% |
| **barrier (excl/pers, sin kNN)** | **0.2477** | **+6.3%** |

- **barrier ≈ partial: idénticas en 7/8 escenas, solo `room1` −0.003** (0.352→0.349). Cambio global
  −0.04% = ruido. **Los guards seam/color NO estaban ganando AP** → quitarlos sale gratis y ahorra
  el kNN de normales+color en producción. Win de cómputo, neutro en calidad. (Esperado: splits pocos
  y de poca masa.)
- **tau030 (0.2518) sigue > barrier (0.2477).** Ese ~0.4% ya se sacrificó en el rediseño de partial
  (lado conservador); la barrera lo mantiene, no lo recupera. Objetivo AP-puro → tau030; objetivo
  simplicidad+robustez+coste → barrier.
- **room1 −0.003 SIN CERRAR:** en el sustrato 32ac4 room1 tiene un focused entrant que flipa al
  quitar guards (en el sustrato tau030-viz room1 tenía 0 entrants → por eso el análisis previo no lo
  vio). Herramienta: `scratchpad/room1_diff.py` (cargar desde raw-baseline-full_32ac4). Trivial pero
  es held → mirar en visor si se quiere cerrar.

### 7.10 Refactor: `ContestThresholds` dataclass (2026-07-08, sin cambio de comportamiento)
Los 9 umbrales del discriminador pasan de params sueltos del `__init__` (defaults triplicados en
discriminator/manager/engine) a un `@dataclass(frozen=True) ContestThresholds` en `discriminator.py`,
agrupado por banda, con `from_cfg(cfg)` (ignora claves desconocidas). `ContestDiscriminator.__init__`
= 2 args (`th`, `root`). `classify` usa `self.th.X`. manager/engine construyen con
`ContestThresholds.from_cfg(cfg)`; `_thresholds()` de engine = `asdict(d.th)`. **Defaults en 1 sitio.**
Verificado idéntico en los 6 entrants (`scratchpad/verify_new.py`).

### 7.11 Run desde 0 para VISUALIZAR el estado actual (barrera)
`manifests/20260708_GT_CLIP_contest-barrier-viz.yaml` (clon de `contest-tau030-viz`, config barrera,
SLAM completo + `save_rrd`, GT limpio `noise_enabled:false`). Genera pre_fusion.ckpt fresco + `.rrd`
para el visor. Env `ovo2`. Comparable 1:1 con el rrd de tau030-viz para ver los splits en vivo.

### 7.12 SIGUIENTE ESTUDIO — recuperar las masas grandes que focus omite (ARRANQUE INMEDIATO)
Con la banda focus ya cerrada (barrera excl/pers, umbral 0.7 fijo, AP-neutral), el frente abierto es
**la omisión**: focus tiene recall ~3% (§7.8) y deja fuera transferencias GT-limpias **enormes**. No
es cosmético — hay masa real de AP sobre la mesa.

**EL FINDING (medido, con números duros):**
- `room2` hace **0 splits** (max focus 0.60 < 0.7) pero tiene DOS transferencias GT-limpias gigantes
  ignoradas: `7→0` firm **10361** (pur 0.99), `79→46` firm **9496** (pur 1.00), + `76→72` firm 3243.
  Esto explica el AP plano de room2. (`scratchpad/dominancia_candidates.py`.)
- `office4`: solo dispara 1, pierde `11→15` (firm 522, pers 0.92, pur 1.0), `30→45` (297), `82→75` (311)...
- Causa raíz (§7.8): `focus = firm_par / TODO lo disputado del defender` colapsa cuando varios vecinos
  muerden al defender → las buenas viven en focus 0.05-0.2, **mezcladas con smears**, y bajar el umbral
  mete FP (probado en el barrido). focus no puede separarlas → **hace falta otro eje**.

**EL PLAN (§7.2, sin cambios):** split **POR PAR** (no winner-take-all por defender) + barrera de
firmeza del trozo (pers/excl, ya validada en focused). Ataca directamente los ~646 missed.

**EL BLOQUEO REAL — falta un guard estructural, y ya sabemos qué NO vale:**
- **seam (giro de normal): NO discrimina** (`scratchpad/seam_check.py`). Giros 1-12° para TODOS
  (buenas y smears). El trozo se elige como los puntos que el challenger roba → por construcción se
  apoya en su superficie → siempre alineado. Muerto salvo cojín-sobre-sofá.
- **color: caro (kNN) y ya borrado**; además tampoco separaba el caso duro.
- **planitud del trozo sola: NO basta.** El smear `30→45` es lámina fina (espesor-z 0.039m) PERO el
  TP `82→17` también es superficie plana horizontal (nz 0.99). Flatness no distingue.
- Queda por explorar: **coherencia 3D del trozo** (PCA: ¿blob real vs lámina-smear?), **tamaño del
  trozo vs challenger**, **trap misma-clase** (2 sillas/objetos 93xxx se roban con pers alta a baja
  contención — §7.2.3; en producción no hay clase, señal a inventar).

**LA PREGUNTA QUE DECIDE GO/NO-GO (hacerla PRIMERO, antes de tocar código):**
¿Los 646 missed son transferencias reales o smears/mask-bleed? Caso testigo **sin resolver**:
`office4 30→45` — trozo firme (firm 297, pers 0.65, excl 1.0) a ras de suelo (z -1.11) con normal
HORIZONTAL (nz 0.09 = superficie vertical). GT dice pared 93056, def suelo 40007. Puede ser (a) smear
de máscara de pared sobre franja de suelo = FALSO, o (b) puntos de la BASE de la pared mal asignados
al suelo = transferencia CORRECTA. No se resolvió a ojo porque el visor pinta instancias enteras, no
el trozo. **Primer paso: validar en visor los 297 puntos concretos** con
`scratchpad/export_chunk_ply.py <def> <ch>` (PLY: def rojo, ch azul, trozo amarillo, contexto gris) —
empezar por las gordas de room2 (`7 0`, `79 46`) y el ambiguo `30 45`. Si son reales → el eje
firmeza-del-par vale y el trabajo es el guard estructural. Si son smears → focus tenía razón y hay que
buscar el AP en otro lado.

**Herramientas listas (todas en `scratchpad/`, env `ovo`, PYTHONPATH=repo):**
`dominancia_candidates.py` (candidatas GT-limpias por escena), `office4_funnel.py` (embudo por-par),
`export_chunk_ply.py` (ver el trozo), `seam_check.py`/`probe_30_45.py` (geometría del trozo),
`focus_sweep.py` (efecto del umbral). rrd para visor: run `contest-barrier-viz` (§7.11).
Rutas: sustrato `data/checkpoints/Replica/…raw-baseline-full_32ac4/{scene}/pre_fusion.ckpt`;
`ContestProbe.from_experiment(exp, scene)` da `_pairs` (PairFeatures) + `_map` + `_cb` (seam/color aún
disponibles para análisis aunque el discriminador ya no los llame).

### 7.13 EN CURSO (2026-07-08 tarde-noche): rediseño SPLIT por-par — arquitectura + validación visual

**A. GO/NO-GO RESUELTO = GO.** Los miss NO son smears: son transferencias reales. Validado en visor
(herramienta nueva, abajo) sobre office4. Casos confirmados a ojo:
- `30→45` (el testigo ambiguo suelo/pared): **REAL** — refinamiento de suelo.
- `11→15`, `17→99`, `36→30`, `70→30`, `121→30`: refinamientos de suelo reales.
- `75→68`: mata puntos flotantes dentro de otra instancia. `77→70`, `26→15`, etc: micro-refinamientos.
- **`103→99` y `101→99` son BUENOS pese a pur 0.65-0.66** → **la pureza GT MIENTE** (dice mezcla, el ojo
  ve refinamiento preciso de objeto chico). LECCIÓN: el guard NO debe apoyarse en pureza. Los low-pur
  que íbamos a descartar son oro.
- Conclusión: `pers/excl` (sin guard geométrico) ya separa lo bueno de la basura en office4; la basura
  cayó por pers/excl bajo, no por geometría. `82→75` queda **APARCADO para el final** (dudoso: ¿75 es
  objeto legítimo o fragmento espurio? señales buenas —firm 311, pers 0.83, excl 0.93— pero a discutir).

**B. LA LIMITACIÓN RAÍZ era la firma `classify -> Verdict`** (un veredicto por defender). Colapsaba los
N pares del defender en 1. Análisis del flujo:
- Entra a `classify`: un defender + TODOS sus pares de conflicto (uno por challenger que le roba firme).
  Cada `PairFeatures` YA trae el trozo entero (`split_points`, firm, pers, excl). La info por-par está
  en la ENTRADA; se perdía en la SALIDA.
- **MERGE y SPLIT son de naturaleza distinta:** MERGE = del defender (uno, terminal, el defender
  desaparece → winner-take-all correcto). SPLIT = del par/trozo (0..N, el defender sobrevive y puede
  soltar varios trozos a varios vecinos → winner-take-all INCORRECTO).
- `focus = firm/n_disputed` es la ÚNICA señal contaminada por la concurrencia del defender (mete los
  puntos de los OTROS pares en el denominador). Y `focus≥0.7` admite **≤1 par matemáticamente** (si uno
  se lleva el 70%, a los demás les queda ≤30%). Por eso el `max(focus)` era cosmético: la puerta focus
  YA es winner-take-all intrínseco. Quitar el `max` no da más candidatos; hay que quitar focus de la
  puerta y filtrar por señal LOCAL al par (pers/excl).

**C. REFACTOR HECHO Y VALIDADO (behaviour-preserving, AP replay idéntico 0.2478 vs 0.2477).**
`classify -> List[Verdict]`. MERGE: early-exit, `return [uno]`. SPLIT: `_split_verdicts()` recorre pares
y emite `[0..N]`. Puerta AÚN `focus≥0.7` (admite ≤1 → misma salida, forma nueva). Cambiar la banda =
1 línea (la puerta). Ficheros: `discriminator.py` (`_split_verdicts` nuevo), `manager.py` (append→
extend), `query/engine.py` (`classify_all` + `_verdict_toward`; `explain`/`branch` adaptados),
`eval/pipeline.py` (itera `classify_all`). Manifest replay: `contest-focus-barrier` (re-lanzado).

**D. DISJUNCIÓN GARANTIZADA POR `excl`.** Los trozos hermanos de un mismo defender NO se pisan si excl
alta: `82→17` (43686 pts, excl 0.99) y `82→75` (311 pts, excl 0.93) → **0 solape**. `excl≥y` ya hace los
trozos disjuntos por construcción → **el split por-par NO necesita lógica de conflicto entre hermanos**;
se pueden aplicar varios splits por defender la misma ronda sin colisión. Elimina el `clean_transfer`
parche (era esto mismo hecho por fuera, con umbrales distintos).

**E. HERRAMIENTA NUEVA — visor 3D interactivo `contest inspect`** (`studies/contest_metrics/viz/
inspect3d.py`, dominio CLI `inspect`). Espejo de la de fusión: escena gris + corte de techo (`--z-max`)
+ REPL por terminal. Query `A B` / `A->B` / `A→B` → pinta **def rojo · ch azul · TROZO verde (gordo,
×6 jitter)** + señales + veredicto. Cmds: `show <id>`, `pairs <def>`, `scene`, `q`. Uso:
`python -m studies.contest_metrics inspect --exp 20260708_GT_CLIP_contest-barrier-viz_66656 --scene
office4 --z-max 1.5`. (verde = lo que el azul le roba al rojo = split_points a transferir.)

**F. ANÁLISIS MISS TODAS LAS TUNING** (sustrato `contest-barrier-viz`, filtro `pers>0.5 Y excl>0.75`,
corte masa firm≥50). Patrón consistente: pers/excl deja set chico (22-29), la masa vive en 1-6 pares,
resto migajas (firm 1-45, pur~1.0, refinamientos reales pero masa nula):
- `office0`: 1 gordo (`43→41` firm 96) + 21 migajas.
- `office3`: 2 gordos (`49→44` firm 76 pur 1.0; `115→109` firm 70 **pur 0.67** low-pur) + 27 migajas.
- `office4`: 4 gordos (`11→15` 522, `82→75` 311, `30→45` 297, `17→99` 137) + 19 migajas.
- `room2`: 6 gordos, **la BALLENA `79→46` firm 9496** (pur 1.0) + `76→72` 3243, `76→74` 1296, `76→36`
  438, `37→32` 315, `9→6` 257 + 20 migajas. → room2 es donde está el AP (explica su AP plano).
- Nota `room2 7→0`: firm 10361 (era candidato en tau030) pero pers **0.40** < 0.5 → NO pasa el filtro
  pers en el sustrato barrier; classify(7)=NO_ACTION. (Antes se citaba como gorda; el filtro lo excluye.)

**SIGUIENTE PASO CONCRETO:** fijar la barrera por-par (`firm_points≥N` + `pers≥x` + `excl≥y`, SIN guard
geométrico — pers/excl basta en tuning) cambiando la puerta de `_split_verdicts` (hoy `focus≥0.7`), y
medir AP (replay). Umbral masa a calibrar: firm≥50 deja 1-6/escena; la cola no mueve AP. Validar en
held (office1/office2/room0/room1) que pers/excl no mete FP sin el guard. Herramientas: `contest
inspect` (visor), `eval` (grade + missed), `focus_partition.py`/`focus_partition` scratchpad.

### 7.14 HECHO (2026-07-08 noche): banda SPLIT por-par implementada (dos bandas conviven)

**Validación visual completa (4/4 tuning) → GO firme.** Detalle por caso en
`docs/contest_split_visual_inspection_2026-07-08.md`. Los splits por-par son transferencias reales
(refina suelo/techo/objetos chicos/flotantes). La ballena room2 `79→46` (firm 9496) validada a ojo.
Hallazgos: (a) **pur miente por 2 razones** — GT proyectado mal etiqueta (office4 103/101→99), o el
trozo tiene **cola sucia real** recortable (office3 115→109); ambos net-positivos. (b) **split-que-es-
merge** (office0 43→41, 61→56): containment bajo → cae a split; transferir el trozo es fix parcial OK
(idea aparcada: escalar split→merge). (c) `82→75` aparcado (¿75 objeto legítimo?).

**DISEÑO FINAL — el SPLIT tiene DOS bandas que conviven** (ambas emiten a `List[Verdict]`, dedup por
challenger, focus prioridad). NO se reemplazó focus; se le SUMÓ por-par (el sentido de la firma lista):
- **focus (sniper, intacto):** `focus≥0.7 Y excl≥min_focus_excl(0.9) Y pers≥min_focus_persist(0.6)`.
  ≤1 par por matemáticas, CUALQUIER tamaño → rescata los splits limpios PEQUEÑOS.
- **por-par (nueva):** `firm≥min_split_firm(50) Y excl≥min_split_excl(0.8) Y pers≥min_split_persist(0.5)`.
  0..N por defender → rescata las GORDAS que focus omite (denominador `n_disputed` contaminado por
  concurrencia). Ej: `79→46` focus 0.22 (focus la mataba) → entra por-par.
- Un defender puede sacar `[focus→A + por-par→B + por-par→C…]` (challengers distintos). Ej office4
  `82`: `focus→17` (firm 43686) **Y** `por-par→75` (firm 311) — lo que la firma vieja `->Verdict` no
  podía. `defenders_multi` >0 en office3/office4/room2.

**BARRIDOS que fijaron los umbrales por-par** (sobre los 49 candidatos gordos firm≥50 excl>0.75 de
tuning; substrato barrier-viz):
- **excl (eje discriminante):** buenos apretados en ~1.0 (11/12 en 0.97-1.00; solo `17→99`=0.82);
  basura en ~0 (96→82=0.03). Subir excl casi GRATIS (a 0.9 solo cae `17→99`). Elegido **0.8** (mantiene
  los 12 validados). excl hace el trabajo que creíamos de focus, pero LOCAL al trozo (no contaminado).
- **pers (solo suelo):** buenos dispersos 0.55-1.0. Subir MATA buenos: 0.55 pierde `9→6`; 0.6 pierde
  `76→36`/`37→32`/`17→99`; **0.65 mata la BALLENA `79→46`**(pers 0.64). Elegido **0.5** (no sacrifica
  validados; la basura ya cayó por firm/excl, no por pers).
- **masa:** el gate viejo `min_mass=50` es sobre `total_grabs` (volumen de robos), NO tamaño de trozo:
  `12→4` firm=1 pero total_grabs=64 → pasaba. Por eso `min_split_firm=50` es un gate NUEVO sobre
  `firm_points` (tamaño real). Las migajas <50 quedan para un **refiner de máscaras externo futuro**
  (muchas, validadas buenas, pero fiabilidad sin ruido sin probar → fuera del contest por ahora).

**CÓDIGO:** `_split_verdicts` reescrito estilo merge (bandas como comprehensions + combinar):
`focus=[...]`, `perpair=[...]`, `verdicts=focus + perpair(dedup)`. Helper `_split(defender,p,kind,sim)`.
`ContestThresholds`: borrados `min_split_cont`; `min_split_focus` conservado; añadidos `min_focus_excl`,
`min_focus_persist`, `min_split_firm`; `min_split_excl` 0.9→0.8, `min_split_persist` 0.6→0.5.
Manifest `contest-perpair.yaml`. **AP replay: media 0.2495 (+0.7% vs barrier; room2 +0.009).**

**BUG arreglado (destapado por la banda por-par):** `manager._resolve_pairs` (y `_reeval_frontier`)
usaban `self.discriminator.low`/`.high` — referencias STALE del refactor a dataclass (ahora viven en
`.th.low`/`.th.high`). No petaba antes porque `_join` solo se llama cuando un par {a,b} tiene split en
LOS DOS sentidos; con la banda por-par (más splits) sí ocurre → `AttributeError` a mitad de office4.
Fix: `self.discriminator.th.low`/`.th.high`. (El replay behaviour-preserving §7.13 no lo destapó
porque focus≥0.7 casi nunca da splits recíprocos.)

**AP MEDIDO (replay perpair vs barrier):** office0 +0.003, office3 +0.002, **room2 +0.009** (la ballena
`79→46` paga), office4 +0.000, held (office1/2, room0/1) +0.000 → **cero regresiones**. Media
0.2477→**0.2495** (+0.7%). Sigue < tau030 (0.2518). office4 no sube pese a 4 splits validados →
**calidad visual ≠ AP** (refinan bordes de instancias existentes, el AP agnóstico apenas lo premia).
Comparativa: raw 0.2330 · barrier 0.2477 · **perpair 0.2495** · tau030 0.2518. Valor del cambio =
arquitectura correcta + room2 + robustez, no salto de AP. clean_transfer OFF en TODA la cadena.

**clean_transfer BORRADO (redundante):** la banda por-par ES clean_transfer bien hecho — misma idea
(pers+excl, complementario a focus) pero DENTRO del discriminador, umbrales unificados, +gate de masa
(firm≥50), excl más estricto (0.8 vs 0.7). Borrados: `clean_transfer.py`, `_add_clean_transfers`, flags
`clean_transfer`/`clean_persist`/`clean_excl` en manager. (Curiosidad: umbrales casi coincidían —
valida que la intuición del parche era buena, solo faltaba integrarla + masa.)

### 7.15 PUNTO DE RETOMA (2026-07-08 noche) — leer esto para continuar

**QUÉ HAY (todo en working tree, SIN COMMIT):**
- **Discriminador `classify -> List[Verdict]`** (merge terminal [1] / split [0..N]). Cascada:
  `ruido → strong(cont≥0.6, MERGE) → partial(0.4-0.6, MERGE) → SPLIT por-par`.
- **SPLIT = 2 bandas que conviven** (`_split_verdicts`), dedup por challenger, focus prioridad:
  - focus (sniper): `focus≥0.7 Y excl≥min_focus_excl(0.9) Y pers≥min_focus_persist(0.6)`. ≤1, cualquier tamaño.
  - por-par: `firm≥min_split_firm(50) Y excl≥min_split_excl(0.8) Y pers≥min_split_persist(0.5)`. 0..N.
- Un defender puede sacar `[focus→A + por-par→B…]` (challengers distintos). Ej office4 82: focus→17 + por-par→75.
- `clean_transfer` BORRADO. Bug `discriminator.th.low/.high` en manager ARREGLADO.

**RESULTADO AP (perpair vs barrier):** media 0.2477→0.2495 (+0.7%); room2 +0.009 (ballena), office0
+0.003, office3 +0.002, resto +0.000, **cero regresiones en held**. Sigue < tau030 0.2518. Calidad
visual ≠ AP (refina bordes, AP agnóstico apenas premia).

**VALIDACIÓN VISUAL:** 4/4 tuning GO, todos los casos en `docs/contest_split_visual_inspection_2026-07-08.md`.

**RUNS/RUTAS:**
- Replay AP: `data/output/Replica/20260708_GT_CLIP_contest-perpair_af54c/` (métrica `AP_agnostic` en
  `replica/instance_ap_<scene>.txt`; NO usar `AP`=0.0, la de record es `AP_agnostic`).
- Viz desde-0 (rrd para visor): `data/output/Replica/20260708_GT_CLIP_contest-perpair-viz_ec060/<scene>/rerun.rrd`.
- Substrato replay: `data/checkpoints/Replica/20260707_GT_CLIP_raw-baseline-full_32ac4/{scene}/pre_fusion.ckpt`.
- Manifests: `contest-perpair.yaml` (replay), `contest-perpair-viz.yaml` (desde 0).

**HERRAMIENTAS:** `python -m studies.contest_metrics inspect --exp <id> --scene <sc> --z-max 1.5`
(visor 3D: `A→B` def rojo/ch azul/trozo verde-gordo + señales + veredicto; `pairs <def>`, `show <id>`).
`eval` (grade + missed). Probe: `ContestProbe.from_experiment(exp, scene)` → `_pairs`/`_by_def`/
`classify_all(d)`/`_verdict_toward(a,b)`.

**ABIERTO (prioridad para el próximo agente):**
1. **Validar HELD en visor** (office1/office2/room0/room1): el +0.000 es neutro-BUENO (no dispara nada
   dañino) vs neutro-por-no-disparar. Usar `inspect` sobre el substrato perpair-viz.
2. **Por qué office4 no sube AP** pese a 4 splits validados buenos → entender qué mide/omite AP_agnostic
   sobre refinamientos de borde (¿el split mejora el mapa pero no la métrica? ¿o el actuator no aplica
   bien?). Verificar en el rrd que los splits se APLICAN (no solo se deciden).
3. **`82→75` — particularidad pendiente** (usuario la tiene, no urge; ver nota en el doc de inspección).
4. **Ideas aparcadas:** (a) refiner de máscaras externo para migajas <50 (muchas, buenas, fuera del
   contest); (b) escalar split→merge cuando el defender es un smear que debería fusionarse entero
   (office0 43→41, 61→56); (c) 82 (pur 0.35) podría pelarse en MÁS trozos.
5. **COMMIT** cuando el usuario lo pida (no hacerlo sin permiso).

**REGLA APRENDIDA:** los smoke tests via probe (`classify_all`) NO ejercen `_resolve_pairs` ni el
actuator de producción → validar SIEMPRE con un run real antes de dar por bueno un cambio de código.

---

## 6. Archivos tocados (resumen)
- `ovo/entities/contest/aggregator.py` — refactor SRP + `_is_firm` relativo + default firm_tau=0.30 + rename min_grabs.
- `ovo/entities/contest/discriminator.py` — **(strong)** borrada rama DEFER + piso
  `min_persist_strong=0.5` + reasons inglés. **(partial)** borrados guards muertos (bidireccional,
  inestable) + rediseño fragment (`partial_merge_persist=0.6`, `partial_merge_excl=0.8`, `low=0.4`,
  `high=0.6`) + borrados params `max_rev_split`, `sim_merge`. **(focused, 2026-07-08)** guards
  seam/color → barrera `excl≥0.9 Y pers≥0.6`; `classify` sin seam/color; `min_split_focus=0.7`
  parametrizado; borrados `min_persist_split`/`max_seam_angle`. **(refactor)** umbrales → `@dataclass
  ContestThresholds` con `from_cfg`; `__init__(th, root)`. **(arquitectura, 2026-07-08 noche)**
  `classify -> List[Verdict]` (MERGE terminal `[uno]`; SPLIT `[0..N]` en `_split_verdicts`). §7.13.
  **(split 2 bandas, 2026-07-08 noche)** focus (sniper, intacto) + por-par (nueva: `firm≥50 excl≥0.8
  pers≥0.5`); dedup por challenger; estilo comprehension + helper `_split`. Thresholds: +`min_focus_excl`
  /`min_focus_persist`/`min_split_firm`, `min_split_excl` 0.9→0.8, `min_split_persist` 0.6→0.5,
  −`min_split_cont`. §7.14.
- `ovo/entities/contest/manager.py`, `.../query/engine.py` — construyen con
  `ContestThresholds.from_cfg(cfg)`; borrados seam/color de `report`/`classify`; `_thresholds`=`asdict`.
  **(arquitectura)** manager `append`→`extend`; engine `classify_all()` + `_verdict_toward()`.
- `studies/contest_metrics/viz/inspect3d.py` — NUEVO visor 3D interactivo (dominio CLI `inspect`):
  escena gris + corte techo + REPL `def→ch` (rojo/azul/verde-gordo) + señales/veredicto. §7.13.E.
- `ovo/entities/ovo.py` — borradas closures `_contest_seam`/`_contest_color`; `report` solo con sim
  (el kNN normales/color ya no se computa).
- `studies/contest_metrics/eval/pipeline.py` — `SCENE_GROUPS` tuning/held con room2↔room1.
- `ovo/entities/contest/types.py` — comentario firm_points (min_grabs).
- `studies/contest_metrics/eval/{__init__,gt,grade,pipeline,report}.py` — NUEVO módulo evaluador.
- `studies/contest_metrics/cli.py` — dominio `eval`, overrides tolerantes a strings.
- `studies/contest_metrics/common/paths.py` — `MESH_ROOT`, `GT_ROOT`.
- `.claude/skills/contest-eval/SKILL.md` — NUEVO skill.
- `.claude/skills/run-experiment/SKILL.md` — tabla contest: min_grabs + firm_tau.
- `manifests/20260707_GT_CLIP_contest-tau*.yaml` — barrido firm_tau.
- `manifests/20260708_GT_CLIP_contest-strong-v2.yaml` — replay high=0.6+piso (respalda config strong).
- `manifests/20260708_GT_CLIP_contest-partial-fragment.yaml` — replay config fragment (guards). AP +6.5% vs raw.
- `manifests/20260708_GT_CLIP_contest-focus-barrier.yaml` — replay barrera (config actual). AP +6.3% vs raw (neutro vs guards).
- `manifests/20260708_GT_CLIP_contest-barrier-viz.yaml` — run desde 0 + rrd para visualizar la barrera.
- `manifests/20260708_GT_CLIP_contest-perpair.yaml` — replay split 2 bandas (focus + por-par). §7.14.
- `manifests/20260708_GT_CLIP_contest-perpair-viz.yaml` — desde 0 + rrd, estado actual (2 bandas). §7.15.
- `docs/contest_split_visual_inspection_2026-07-08.md` — NUEVO: veredictos a ojo por caso (4 tuning).
- `ovo/entities/contest/clean_transfer.py` — **BORRADO** (redundante con la banda por-par). §7.14.
- `ovo/entities/contest/manager.py` — split `extend`; `.th.low/.high` (fix); borrado branch/flags clean_transfer.
- `studies/contest_metrics/viz/inspect3d.py` + `cli.py` (dominio `inspect`) — NUEVO visor 3D. §7.13.E.
