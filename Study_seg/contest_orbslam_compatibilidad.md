# Compatibilidad del Contest Mechanism con ORB-SLAM

**Fecha:** 2026-06-16
**Pregunta:** ¿El mecanismo de contest (merge/split de instancias) funciona técnicamente con ORB-SLAM, igual que en simulated/vanilla? ¿Tenemos todas las señales?

---

## TL;DR

- El **mecanismo es backend-agnóstico**. Las 4 señales que consume el contest NO las da ORB-SLAM: las fabrica la capa `VanillaMapper`, común a los 3 backends. Conceptualmente, todas existen.
- **MERGE funciona en ORB sin tocar nada** (solo usa containment = puntos + ids permanentes, que ORB reconstruye bien).
- **SPLIT NO funciona en ORB tal cual**: el primer loop closure con poda corrompe `pcd_obs` (persistence) y crashea/desincroniza `pcd_normals` (seam).
- **Fix obligatorio si se quiere SPLIT**: pequeño y localizado, en `orbslam2.update_map()`.
- Aparte (no bloqueante): bajo ORB la fusión + `contest.report()` solo se dispara en eventos de loop-closure/GBA, no en cadencia regular.

---

## 1. Quién genera las señales

`WrapperORBSLAM2` **hereda de `VanillaMapper`** (`ovo/slam/orbslam2.py:17`). El point cloud denso y todas las señales por punto se fabrican en la capa Vanilla (desproyección de depth). ORB-SLAM solo aporta:

- poses de cámara estimadas (no GT)
- flag `is_last_frame_kf`
- eventos de loop-closure / GBA (`get_last_big_change_idx`) que disparan `update_map()`

### Señales que consume el contest

| Señal | Origen | Uso en contest |
|---|---|---|
| `point_ids` (pcd_ids) | arange monótono, permanente | clave del store; aguanta reordenamiento |
| `points_ins_ids` (pcd_obj_ids) | OVO semántico | dueño actual del punto |
| `point_obs` (pcd_obs) | scatter_add en re-match | **persistence** (guard de SPLIT) |
| `point_normals` (pcd_normals) | depth_to_normals al crear | **seam angle** (guard de SPLIT-dominancia) |
| CLIP sim | objetos OVO | desempate franja parcial |

El contest indexa el historial **por id permanente de punto**, no por fila. Por eso sobrevive a reordenamientos/podas del SLAM (`prune_to_live`, `on_remove`, `on_merge`). Ese diseño es justo lo que lo hace viable bajo ORB.

---

## 2. La diferencia estructural: dos formas de mutar el mapa

El problema NO está en cómo se generan las señales, sino en qué pasa cuando el SLAM corrige el mapa (loop closure).

Los datos viven en **arrays paralelos**: la fila `i` une `pcd[i]`, `pcd_ids[i]`, `pcd_obs[i]`, `pcd_normals[i]`. La única atadura entre una columna y su punto es estar en la misma fila (salvo los ids, que además son permanentes).

### Simulated — lo hace BIEN (transforma in-place)

`ovo/slam/simulated/slam.py` corrige el mapa moviendo los puntos **in-place, por slice de keyframe**:

```python
self.pcd[start:end]         = (T @ pcd_slice_hom.T).T[:, :3]
self.pcd_normals[start:end] = (T[:3, :3] @ self.pcd_normals[start:end].T).T  # slam.py:295
```

El array nunca cambia de tamaño ni de orden → `obs` y `normales` siguen alineados automáticamente. `obs` es un contador (mover el punto no le afecta); las normales se rotan explícitamente. **Por eso el contest funciona en simulated** — no por ausencia de loop closure, sino porque la corrección es in-place y respeta todas las columnas.

### ORB — lo hace MAL (reconstruye por concatenación)

`orbslam2.update_map()` (`ovo/slam/orbslam2.py:74`) tras un loop closure **reconstruye el array** concatenando los slices de los KFs vivos:

```python
new_pcd, new_pcd_ids, new_pcd_obj_ids, new_pcd_colors = [], [], [], []
# ... recorre KFs vivos, recoloca puntos con la pose corregida ...
self.pcd        = torch.cat(new_pcd)
self.pcd_ids    = torch.cat(new_pcd_ids)
self.pcd_obj_ids= torch.cat(new_pcd_obj_ids)
self.pcd_colors = torch.cat(new_pcd_colors)
```

Reconstruye 4 columnas y **olvida `pcd_obs` y `pcd_normals`**. Quedan con el largo y orden VIEJOS mientras `pcd` pasa al nuevo → **desincronización**.

---

## 3. Orden de KFs: estable (verificado)

`getKeyframePoints()` (binding C++) hace `std::sort(vpKFs, KeyFrame::lId)` → ordena por id de creación (monótono). El `pcd` original también está en orden de creación. Por tanto los supervivientes **conservan su orden relativo**; la poda solo abre huecos, no baraja filas.

Consecuencias afinadas:

- **Loop closure sin podar nada**: mismo largo y orden → `obs` correcto; normales alineadas pero **sin rotar** (dirección del frame viejo). Sin crash, seam degradado.
- **Loop closure con poda** (caso normal): N_viejo ≠ N_nuevo →
  - **`obs`**: se indexa el array viejo con posiciones del nuevo → **corrupto en silencio** (sin crash, filas equivocadas → persistence basura).
  - **`normales`**: `chunk_mask`(N_nuevo) aplicado sobre `pcd_normals`(N_viejo) → tamaños no casan → **`RuntimeError`** cuando aparezca un candidato de SPLIT-dominancia.

---

## 4. Cadencia de fusión bajo ORB (hallazgo de diseño)

`map_updated` se pone a `True` **solo dentro de `orbslam2.update_map()`**, que solo corre cuando cambia `big_change_id`. `InformNewBigChange` se llama **únicamente desde `LoopClosing.cc`** (corrección de bucle + GBA global).

Por tanto: bajo ORB, **toda la fusión semántica + `contest.report()` se dispara SOLO en eventos de loop-closure/GBA**, no en cadencia regular.

Matiz: `contest.record()` (acumular votos, camino caliente) SÍ corre regular (va en el paso de detección/tracking, gobernado por `segment_every`). Lo que se aplaza al loop closure es la **evaluación** (`report`) y la fusión de instancias.

→ Si una escena cierra pocos bucles, el contest casi no actúa. No es un bug, es comportamiento del wrapper, pero condiciona fuertemente el resultado. Decisión de diseño a tomar aparte.

---

## 5. Cuadro comparativo

| | simulated | ORB-SLAM |
|---|---|---|
| construye/corrige mapa | in-place por slice | rebuild por concatenación |
| `obs` sincronizado tras LC | ✓ | ✗ (corrupto si hay poda) |
| `normales` tras LC | rotadas ✓ | no recopiadas → crash / stale |
| cadencia fusión + contest | regular | solo en loop-closure/GBA |
| MERGE (containment) | ✓ | ✓ |
| SPLIT (persistence + seam) | ✓ | ✗ |

---

## 6. Qué hay que modificar

### Obligatorio para SPLIT (cambio chico, 1 sitio)

En `orbslam2.update_map()`, añadir las 2 columnas al rebuild, espejo de lo que ya hace con pcd/ids/colores:

1. **`pcd_obs`**: copiar el slice del KF (es un contador, no se transforma).
2. **`pcd_normals`**: copiar el slice **y rotarlas** por `transform[:3, :3]` (mismo patrón que `simulated/slam.py:295`).

```python
new_pcd_obs.append(self.pcd_obs[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]])
new_pcd_normals.append(
    torch.einsum('mn,bn->bm', transform[:3, :3],
                 self.pcd_normals[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]])
)
# ... y al final:
self.pcd_obs     = torch.cat(new_pcd_obs, dim=0)
self.pcd_normals = torch.cat(new_pcd_normals, dim=0)
```

### No bloqueante (decisión de diseño)

La cadencia event-driven (fusión solo en loop-closure). Decidir si el contest debería evaluarse con más frecuencia bajo ORB.

---

## 7. Confianza y pendientes

**Confianza alta (leído en código):** herencia Vanilla, ausencia de obs/normales en el rebuild, indexación por id permanente, orden de KFs por lId, cadencia atada a `InformNewBigChange` desde LoopClosing.

**No verificado empíricamente:**
- El crash/corrupción no se observó en ejecución (es inferencia del código).
- Cuántos loop closures dispara Replica con ORB → define si el SPLIT-crash es frecuente o raro y si la fusión event-driven es suficiente. Requiere correr.
