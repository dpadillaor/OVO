# Sesión — señal geométrica (normales) para la rama DOMINANCIA del contest

> Fecha: 2026-06-15. Rama: `study/point-instance-assignment`.
> Qué aprendimos y qué implementamos para separar **fragmento real** (transferir)
> de **objeto en contacto** (no transferir) en la banda de dominancia del contest.
> Lee primero `HANDOFF_contest.md` (mecanismo) y `flujo_decision_contest.puml`.

---

## 0. Punto de partida

La rama **dominancia** (contención baja 0.05–0.5 + focus≥0.7) emitía SPLIT para todo
objeto adyacente. Antes había un intento de MERGE por descriptor (CLIP) que el usuario
**revirtió** esta sesión (over-merge: CLIP no distingue instancias de la misma clase —
3 sillas pegadas → sim alto → fusión falsa). Dominancia quedó SPLIT-only.

Problema observado en el visor (run con `split_mode: all`):
- **89→103** (BUENO): 89 es una silla mal segmentada que se comió un trozo de la
  mesa 103. Ese trozo debe reasignarse a 103. ✅ split correcto.
- **Malos**: el split robaba puntos a objetos **en contacto**:
  - sillas adyacentes (hub 35 tragaba 10 fuentes; 102 ← 88, 105),
  - objetos **encima** de la mesa (131,132,134→103),
  - cojín apoyado en sofá (18→26, 31→22).

`split_mode: partial` NO aplica dominancia (solo la loguea). `all`/`dominance` sí.

---

## 1. Lo que NO funciona (descartado con datos)

**Ninguna feature de máscara 2D separa el bueno de los malos.** El BUENO `89→103` y
el MALO `18→26` son **gemelos**:

| caso          | containment | persistence | focus | sim  |
|---------------|-------------|-------------|-------|------|
| 89→103 BUENO  | 0.128       | 0.229       | 0.989 | 0.76 |
| 18→26  MALO   | 0.132       | 0.228       | 0.834 | 0.86 |

- **persistence** (obs_W / obs_total): no separa (ambos ~0.23).
- **owner_share** (obs_dueño / obs_total, derivada del ckpt): tampoco (89 y 18 ≈ 0.77).
  El dueño SÍ sigue viendo los puntos en los dos casos.
- **cobertura / tamaño** (chunk/|W|, |W|/|A|): separa parte del grupo silla/cojín pero
  **se rompe** con objetos sobre mesa (89 y 131-145 ceden ambos a la mesa 103).
- **descriptor CLIP**: inservible aquí (cojín~sofá sim 0.86, misma clase).

→ La distinción "trozo coplanar de mesa" vs "objeto apoyado encima" es **3D**, no está
en las proyecciones de máscara.

---

## 2. Lo que SÍ funciona — giro de normal en la costura

**Idea (LCCP-like):** en la frontera (costura) chunk↔W, ¿la superficie es continua
(mismo objeto) o tiene un quiebre (objetos distintos)?

- Fragmento real (trozo de mesa) → normales a ambos lados **paralelas** (~8°).
- Objeto en contacto → las normales **giran** (quiebre/surco) → ángulo grande.

**Importante (aprendido peleándonos con ello):**
- El **signo** cóncavo/convexo (LCCP puro) **falla**: depende de orientar las normales
  "hacia afuera" globalmente y no se logra de forma fiable. Usamos solo la **magnitud**
  del giro (`|cos|`), que es invariante a orientación.
- La **magnitud sola** basta porque medimos **solo en la costura** chunk↔W (no en
  bordes sueltos, donde una esquina convexa daría falso positivo).

### Resultados (parche k=15, run `b5f77` con dueños intactos)

```
pair        tag     ang_normal
176->43             2.9°
168->159            7.5°
173->175            7.9°
89->103   BUENO     9.9°   ← fragmento real
36->38             10.0°
──────── hueco ────────
132->103           22.5°
131->103  MALO     26.0°   ← objeto sobre mesa
134->103  MALO     36.3°
31->22    MALO     40.2°   ← cojín
18->26    MALO     61.9°
```

**Hueco limpio 10°↔22°. Umbral = 15°.** Bueno + candidatos < umbral; todos los malos >.

### De dónde salen las normales
- **Del depth**, no PCA. En Replica el depth es **ground-truth (sintético, sin ruido)**
  → normales precisísimas. Resolvió lo que el PCA dejaba dudoso (131,134 pasaron de
  ~13-17° borderline con PCA a 25-29° claros con depth).
- Cálculo barato: rejilla de la imagen (vecinos gratis) → 2 restas + producto cruz.
  ~ms/frame, **<1%** del coste de SAM (381 ms/frame).
- **Parche k=15** para la normal de W (promedio de k vecinos, no el punto del filo) →
  evita apoyarse en el canto ruidoso de W. En Replica-GT ≈ k=1 (no hay ruido que
  promediar); el parche importa con **sensor real**. Coste extra despreciable (mismo
  kd-tree, no recalcula normales).

---

## 3. Lo implementado

### 3.1 Normales en el mapa (como `color`: se fijan al crear el punto)
- `ovo/utils/geometry_utils.py` → **`depth_to_normals(depth, intrinsics)`**: normales
  por píxel en cámara, orientadas hacia la cámara, 0 en bordes/inválidos.
- `ovo/slam/vanilla_mapper.py` → atributo **`pcd_normals`**; se calcula en `map()` del
  depth, se rota a mundo, se apila junto a xyz/color/obs; en `get/set_map_dict` como
  `"normals"` (backward-compat: ckpt viejo → ceros); getter `get_point_normals()`.
- `ovo/slam/simulated/slam.py` → `_transform_pcd_slice` **rota las normales** en el
  loop closure (los `_add_points/_unproject/_append` de simulated están **deprecated**;
  `map()` usa `super().map()` de Vanilla).

### 3.2 Guard de giro-normal en DOMINANCIA (lazy, patrón `sim`)
- `ovo/utils/instance_utils.py` → **`seam_normal_angle(chunk_xyz, chunk_normals,
  w_xyz, w_normals, k=15, dmax=0.05)`**: kNN del chunk a W (open3d KDTree), promedio
  de normal de W con alineación de signo, `|cos|` medio. None si no hay costura.
- `ovo/entities/contest/discriminator.py` → param **`max_seam_angle=15.0`**; rama
  dominancia: tras el guard de persistencia, si `seam(...) > max_seam_angle` → NO_ACTION
  "objeto en contacto". `classify(..., seam=callable)`.
- `ovo/entities/contest/manager.py` → `report(..., seam=None)` reenvía a classify;
  wiring de `max_seam_angle` desde `cfg`.
- `ovo/entities/ovo.py` → callable **`_contest_seam(loser, winner, chunk_ids)`** que
  saca chunk (por point-ids) y puntos de W del mapa y llama `seam_normal_angle`.
  Pasado a `report(..., seam=...)`.
- `ovo/entities/ovomapping.py` → pasa `point_normals=get_point_normals()` a `update_map`.

### 3.3 Guard de PERSISTENCIA en dominancia (resuelve las SILLAS)
- Las sillas adyacentes tienen persistencia **bajísima** (el ganador las roza de
  refilón): hub35 + 102 → persist **0.009–0.06**; el fragmento real 89→103 → **0.229**.
- `discriminator.py` rama dominancia: `persistence < min_persist_split(0.1)` → NO_ACTION.
  Mata todas las sillas, conserva el fragmento. (run `0d9ca`: 25 splits → NO_ACTION,
  todas las sillas; sobreviven cojines/objetos-mesa, que ya filtra el guard de normal.)

### 3.4 Orden de la rama dominancia (final)
```
focus≥0.7 y cont≥0.05 y mass≥50:
  persistence < 0.1            -> NO_ACTION  (parpadeo / sillas adyacentes)
  giro_normal_costura > 15°    -> NO_ACTION  (objeto en contacto: cojín, cosa sobre mesa)
  si no                        -> SPLIT      (fragmento real -> transferir)
```

### 3.5 Visor (Rerun) — normales por instancia
- `rerun_handlers.py` / `rerun_orchestrator.py` / `rerun_contracts.py` / `ovomapping.py`:
  normales **por objeto** en `world/normals/obj_<id>` (flechas), **ocultas por defecto**
  (`EntityBehavior(visible=False)`, override de path concreto — el wildcard `**` no
  matchea), **logueadas una sola vez al final** (`finalize()` al cerrar el stream) para
  no doblar el rrd (911 MB → 1.9 GB si era por frame; ahora ~918 MB).

### 3.6 Herramientas
- `Study_seg/seam_normals_viz.py` — mide y visualiza el giro de normal por par de
  dominancia; genera `.rrd` con costura + parche W + normales. Requiere run
  `split_mode: partial` (dueños intactos) con normales.
- Skill `contest-data` actualizada con esta herramienta.

---

## 4. Config nueva (bloque `contest:` bajo `semantic`)
- `max_seam_angle` (default **15.0**) — umbral de giro de normal en dominancia.
- `min_persist_split` (default 0.1) — ahora aplica también a dominancia (antes solo
  partial).

---

## 5. Pendiente / validar (priorizado)
1. **CORRER Y VALIDAR en el visor** un run con `split_mode: all` + ambos guards: que
   las sillas, cojines y objetos-sobre-mesa dejan de robar, y que 89→103 sobrevive.
   (Esta sesión NO se corrió el run final del guard integrado.)
2. **Pocos casos buenos** (1: el 89). El hueco 10↔22° es amplio, pero conviene más
   fragmentos reales para fijar el umbral con confianza.
3. **Solo Replica (depth GT).** Con sensor real el parche k=15 será relevante (ruido);
   validar ahí.
4. **Métricas reales (mIoU/AP)**: todos los runs fueron `stages:[run]`. Nunca medimos
   calidad de mapa, solo veredictos. Pendiente correr `segment`+`eval`.
5. **Sin tests** del discriminador/aggregator/seam.

---

## 6. Runs de referencia (office3)
| run | qué |
|---|---|
| `..._contest-split-all-pia_ebed6` | dominancia aplicada (sin guards nuevos) |
| `..._contest-dom-persist-pia_0d9ca` | + guard persistencia (sillas fuera) |
| `..._contest-normals-pia_4b928` | normales en mapa + viz (split aplicado) |
| **`..._contest-normals-partial-pia_b5f77`** | **normales + dueños intactos → análisis de la señal** |
