# Refresh geométrico por BA local: diseño (2026-07-12)

**Rama:** `study/contest-realtime-online`.
**Estado:** diseño consolidado tras sesión de estudio del backbone ORB-SLAM. **Aún NO se ha
tocado código.** Documento de continuidad.

Lectura previa: `docs/contest_online_design_2026-07-09.md` (contest online, misma rama).

---

## 0. El problema observado
La nube densa de OVO se construye proyectando depth **en el momento en que nace cada keyframe**,
usando la pose que ORB da en ese instante (tracking en vivo, `get_last_trajectory_point`). Esa pose
es cruda: ORB la sigue refinando después con **BA local**. Pero OVO **solo relee poses en big
change** (loop closure / GBA), vía el trigger `get_last_big_change_idx()`.

→ Entre big changes, los refinamientos de BA local se **ignoran**. Cada KF queda clavado en su pose
de nacimiento. La nube luce ruidosa (doble pared, grosor), y la **reproyección por frame** (que
alimenta la evidencia del contest: `record_sighting/claim/grab`) proyecta puntos mal colocados →
evidencia sucia **antes de que caiga ningún loop closure**.

## 1. Quién lee la geometría (y cuándo importa refrescar)
- **Paso semántico** (reproyección + evidencia contest) → cada `segment_every` (10 por defecto).
- Re-fusión de big change → ya cubierto por el path actual.

Entre pasos semánticos nadie mira la nube. Refrescar ahí es tirar cómputo.

## 2. La señal: `mnMapChange` (no `mnBigChangeIdx`)
ORB tiene dos contadores:

| Contador | Sube en BA local | Sube en LC/GBA | Expuesto en binding |
|---|---|---|---|
| `mnBigChangeIdx` (usado hoy) | **No** | Sí | Sí (`get_last_big_change_idx`) |
| **`mnMapChange`** (el que queremos) | **Sí** | Sí | **No** |

Evidencia: `Optimizer::LocalBundleAdjustment` → `IncreaseChangeIndex()` [Optimizer.cc:1489] →
`mnMapChange++` [Map.cc:341]. `mnBigChangeIdx` solo lo tocan LoopClosing/merge
[LoopClosing.cc:1192, 2488].

**`mnMapChange` vive solo en `Map`, no forwarded a System/Atlas → el binding no lo alcanza.**
Exponerlo = 3 líneas, mismo patrón que `get_last_big_change_idx`:
1. `System::GetMapChangeIndex()` → `mpAtlas->GetCurrentMap()->GetMapChangeIndex()` (copia
   `System::GetLastBigChangeIdx` [System.cc:1121]).
2. `ORBSlamPython::getMapChangeIndex()` wrapper (copia [ORBSlamPython.cpp:360]).
3. `.def("get_map_change_index", ...)` [cerca de ORBSlamPython.cpp:56]. Recompilar binding.

La señal solo dice "algo cambió desde la última vez" (contador monótono). No dice qué KF ni cuánto.

Descartado: flags por-KF (`mnBALocalForKF` = "metido en el set de optimización", no "pose cambió");
`mnNumberOfOpt` está **muerto** (nunca se incrementa). La señal buena es a nivel Map.

## 3. Qué KF se movieron: diff de poses en Python
El contador no lo dice. Se averigua comparando:
```
nuevas = get_keyframe_points()          # poses actuales (incluyen BA local; lectura viva, sin gating)
movidos = [kf for kf in nuevas si delta(estimated_c2ws[kf.id], kf.pose) > eps]
```
Comparar matrices 4x4 es trivial aunque haya 200 KF. El coste real (mover puntos) solo se paga por
los movidos. BA local optimiza una **ventana local** (~10-20 KF recientes); el resto queda clavado.

## 4. Culling: ORB SÍ poda entre big changes (¡confirmado!)
**Corrección importante de una suposición inicial errónea.** El thread de Local Mapping poda en
**cada** iteración, sin guard de loop closure:
- `MapPointCulling()` [LocalMapping.cc:92] — borra map points recientes malos.
- `KeyFrameCulling()` [LocalMapping.cc:191] — KF redundante (90% de sus puntos vistos por ≥3 KF)
  → `SetBadFlag()` [LocalMapping.cc:1046] → `Map::EraseKeyFrame()` [KeyFrame.cc:677].

→ Un KF que OVO conoce (por `mnId`) **puede desaparecer** de `get_keyframe_points()` en cualquier
poll, no solo en big change.

Matiz a favor: la nube densa de OVO **no son** los map points de ORB (OVO la construye desde depth,
indexada por `pcd_idxs` propios). El `MapPointCulling` de ORB no toca las rebanadas de OVO. Solo el
**culling de KFs** afecta: pierdes la pose-ancla de ese KF.

## 5. La clave: podar es BARATO; lo caro es la RE-FUSIÓN
Coste separado en tres niveles:

| Operación | Coste | ¿Path ligero? |
|---|---|---|
| Mover puntos (matmul rígido, GPU) | barato | Sí |
| Podar + concat (rebuild geométrico) | barato — O(puntos), GPU | **Sí** |
| Quitar instancias muertas (`_remove_missing_instances` [ovo.py:607]) | barato | Sí |
| **Fusión overlapping + contest classify (seam/color kNN o3d)** | **CARO** | **No** ← lo único que se salta |

`update_map` hoy pega DOS cosas: rebuild geométrico (barato) + `map_updated=True` que dispara la
re-fusión semántica (caro). El path ligero hace el rebuild geométrico completo (mover + podar +
reconciliar instancias muertas) pero **NO** dispara la re-fusión. `map_updated` es el punto de corte.

→ **Sí queremos podar** (mantiene la nube limpia y consistente, evita fantasmas). Lo que NO queremos
es la re-fusión cara. Poda ≠ re-fusión; estaban pegadas por error.

## 6. Momento idóneo: enganchado a `segment_every`, con guarda de contador
```
antes del paso semántico (frame % segment_every == 0):
    if get_map_change_index() cambió:          # ¿BA local movió algo? guarda: gratis si escena quieta
        nuevas = get_keyframe_points()
        rebuild geométrico: mover supervivientes + dropear podados   # barato, GPU
        _remove_missing_instances(...)                               # reconciliación mínima, barata
        estimated_c2ws = nuevas
        # NO map_updated, NO _fuse_overlapping_instances, NO contest.report
    detect_and_track_objects(...)              # ahora reproyecta sobre nube fresca
```
Nube fresca **en el instante exacto** en que se reproyecta. Máx. 1 refresh cada 10 frames. Doble
guarda (contador) → cero coste en escena quieta.

## 7. Thread-safety / mutex: arrancar SIN candado
`getKeyframePoints` protege distinto cada mitad:
- **Qué KF existen** → `system->GetKeyFrames()` [ORBSlamPython.cpp:377] copia el vector bajo el mutex
  del mapa → el **conjunto** es un snapshot consistente.
- **Pose de cada KF** → `pKF->GetRotation()`/`GetCameraCenter()` [ORBSlamPython.cpp:391] leídas
  después, KF por KF. Cada una bloquea el `mMutexPose` de ESE KF → **atómica por KF**. NO hay lock
  que abarque todas juntas.

Riesgos:

| Riesgo | ¿Puede pasar? | Impacto |
|---|---|---|
| Media matriz escrita (basura) | **No** — cada pose es atómica | — |
| Conjunto de KF roto | **No** — GetKeyFrames da snapshot | — |
| KF#5 en estado BA-N y KF#6 en BA-N+1 (poses desfasadas entre sí) | **Sí** — ventana pequeña | pose **válida**, solo de instante ligeramente distinto; **auto-corrige** en el siguiente `mnMapChange`; error transitorio, no acumulativo. BA local va a ráfagas cortas → ventana minúscula |

**Decisión: empezar sin candado nuevo.** No hay riesgo de corrupción, solo micro-desfase transitorio
y auto-sanable. Cero trabajo C++ por esto. Medir en Rerun si aparece algún artefacto.

**Hardening opcional (solo si el experimento muestra artefactos visibles):** getter C++ que lea TODAS
las poses bajo `Map::mMutexMapUpdate` de un tirón → snapshot 100% coherente. ~20 líneas. No es
requisito, no bloquea el plan.

## 8. Reparto final de los dos paths

| Path | Trigger | Mueve supervivientes | Limpia podados | Re-fusión semántica | Coste |
|---|---|---|---|---|---|
| **Nuevo** (BA local) | `mnMapChange` | Sí (rebuild geom) | Sí | **No** | ~ventana local |
| Existente (LC/GBA) | `mnBigChangeIdx` | Sí (rebuild) | Sí | Sí | como hoy |

Lo intacto en el path ligero: descriptores CLIP (no dependen de geometría), evidencia contest
acumulada (keyed por point id, no por coordenada), membresía punto→instancia de supervivientes.

## 9. Trabajo total
- **C++**: 3 líneas (`get_map_change_index`) + recompilar binding. Mutex opcional aparte.
- **Python**: separar el path de refresh geométrico del de re-fusión (`map_updated` = punto de corte);
  añadir la guarda `mnMapChange` antes del paso semántico en `ovomapping.py`; reutilizar la lógica de
  rebuild de `orbslam2.py:update_map` sin el flag.

## 10. Cabos abiertos (los decide el experimento)
1. **`eps`** del diff de poses: bajo → mueves de más; alto → dejas ruido. Tunear.
2. **¿Cuántos KF se podan por ventana entre big changes?** (magnitud del trabajo del path ligero).
3. **Validar que el ruido baja de verdad**: correr con/sin, mirar en Rerun (validación visual).
4. **¿Los KF podados eran redundantes de verdad?** (si sí, dropearlos no quita información real).
