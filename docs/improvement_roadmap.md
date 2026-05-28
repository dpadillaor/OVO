# OVO Improvement Roadmap

Análisis crítico del pipeline actual + plan de experimentos y mejoras.
Basado en auditoría del código (mayo 2026).

---

## 1. Métricas computacionales a instrumentar

El logger actual ya registra `t_sam`, `t_obj`, `t_clip`, `t_up`, `t_fusion`, `t_crit_*`.
Lo que **falta** para tener visibilidad real:

| Métrica | Por qué importa | Dónde añadir |
|---|---|---|
| `n_pairs_evaluated` | Cuántos pares entran al chain por ciclo | `strategy.py: pop_decisions` |
| `n_pairs_short_circuited_at_*` | Qué criterio para más pares (ratio de filtrado) | `strategy.py: per-criterion count` |
| `t_precompute_fusion` | Tiempo de construir `obj_pcds` dict antes del O(n²) | `ovo.py:557-563` |
| `t_overlap_kd` | Tiempo real de KD-tree en overlap (vs overhead Python) | `instance_utils.py:70` |
| `n_instances_alive` | Tamaño del mapa en cada ciclo de fusión | `ovo.py:_fuse_overlapping_instances` |
| `t_descriptor_update` | Tiempo de L1 aggregation por instancia | `instance3d.py:162` |
| `t_loop_closure_refusion` | Coste de re-fusionar todo el mapa en LC | `ovo.py:522-541` |
| `gpu_cpu_transfer_bytes` | Bytes transferidos por frame (pcd, masks, depth) | `ovomapping.py:275-283` |
| `keyframes_queue_depth` | Cuántos KFs esperan procesamiento CLIP | `ovo.py:420` |

### Visualización sugerida
Añadir estas métricas al sidecar `experiment_meta.json` como promedios de run,
y al `fusion_decisions.csv` por ciclo para análisis temporal.

---

## 2. Cuello de botella principal: fusión O(n²)

### El problema real

Con 200 instancias → 19,900 pares por ciclo de fusión.
Si `map_every=5` y hay 2000 frames → **400 ciclos** → **8M de evaluaciones de pares**.
El criterio `overlap` (KD-tree, Open3D) cuesta ~1-5ms por par cuando se llega a él.
Si solo el 10% llega a overlap → 800k × 2ms = **26 minutos** solo en overlap.

Actualmente el cooccurrence graph veta, pero NO pruna — sigue evaluando todos los pares.

### Métricas que confirmarían el problema
- `n_pairs_evaluated` vs `n_instances²/2`
- Distribución de tiempo en `t_crit_p_dist` vs total `t_fusion`
- Qué % de pares llegan a overlap

---

## 3. Plan de experimentos: rendimiento

### 3.1 Cuantificar baseline computacional
**Experimento**: Correr office0 con logging granular (métricas de la tabla de arriba).
**Objetivo**: Saber exactamente dónde va el tiempo antes de optimizar.
**Output esperado**: Breakdown de tiempo por etapa, distribución de pares filtrados por criterio.

### 3.2 Impacto del tamaño del mapa en fusión
**Experimento**: Variar `track_th` (umbral de puntos para tracking) en [50, 100, 200, 500].
`track_th` alto → menos instancias vivas → O(n²) más manejable, pero pierde objetos pequeños.
**Métricas**: `n_instances_alive`, `t_fusion`, mIoU.
**Hipótesis**: track_th=200 da mejor tradeoff tiempo/calidad que el default 100.

### 3.3 Impacto de `segment_every`
**Experimento**: Variar `segment_every` en [5, 10, 20, 40].
SAM es caro (~50-150ms). Segmentar menos frecuente ahorra tiempo pero puede perder instancias.
**Métricas**: `t_sam` total, `n_instances_alive`, mIoU.
**Hipótesis**: segment_every=20 no degrada mIoU en escenas estáticas pero ahorra 50% tiempo SAM.

### 3.4 Impacto de `map_every`
**Experimento**: Variar `map_every` en [3, 5, 10, 20].
Fusionar más frecuente = más ciclos O(n²) pero mapa más limpio.
**Métricas**: `t_fusion` total, n_merges, mIoU.
**Hipótesis**: map_every=10 suficiente para escenas con movimiento lento de cámara.

### 3.5 Impacto del chain de criterios en tiempo
**Experimento**: Comparar chains:
- `[centroid, overlap]` — mínimo viable
- `[centroid, cos_sim, overlap]` — sin cooc
- `[cooccurrence, centroid, cos_sim, overlap]` — default
- `[centroid, aabb, cos_sim, overlap]` — con AABB en lugar de centroid solo

**Métricas**: `t_crit_*`, mIoU, merge correctness.
**Hipótesis**: `aabb` reemplaza bien a `centroid` en objetos elongados sin coste extra significativo.

---

## 4. Plan de experimentos: calidad semántica

### 4.1 Thresholds de fusión
**Experimento**: Grid search sobre `th_centroid` × `th_cossim`:
```
th_centroid: [0.5, 1.0, 1.5, 2.0, 3.0]
th_cossim:   [0.75, 0.78, 0.81, 0.85, 0.90]
```
**Métricas**: mIoU, Fusion_Accept_Rate, n_merges.
**Crítica**: El default `th_centroid=1.5` es arbitrario. Con AABB el umbral debería ser diferente.

### 4.2 Umbral de coocurrencia
**Experimento**: Variar `cooccurrence_veto_threshold` en [1, 3, 5, 10, 20].
¿Cuánto coocurrencia es suficiente evidencia de que son objetos distintos?
**Hipótesis**: threshold=3 es suficiente para la mayoría de escenas Replica.
**Caso problemático**: Objetos en la misma habitación siempre coocurrirán — el grafo puede bloquear fusiones legítimas.

### 4.3 AABB vs centroid
**Experimento**: Comparar `[centroid, cos_sim, overlap]` vs `[aabb, cos_sim, overlap]` en escenas con objetos elongados (office3, office4 tienen mesas largas).
**Hipótesis**: AABB mejora en objetos elongados, neutral en objetos compactos.

### 4.4 k_top_views para CLIP
**Experimento**: Variar `k_top_views` en [3, 5, 10, 20].
El descriptor CLIP se construye como agregado de las mejores vistas.
Más vistas = descriptor más robusto pero más cómputo.
**Hipótesis**: k=5 da 95% del beneficio de k=10 con la mitad del coste.

### 4.5 Impacto del ruido en calidad de fusión
**Ya tenemos datos**: smoke-test (J2-T0.02) vs paper-repro (vanilla sin ruido).
**Siguiente**: Barrer trans_noise en [0.01, 0.02, 0.05, 0.1, 0.2] con GT+loops.
**Objetivo**: Curva degradación mIoU vs ruido, identificar punto de ruptura del sistema.

---

## 5. Mejoras de rendimiento (ordenadas por impacto estimado)

### 🔴 Alta prioridad

#### 5.1 Pruning espacial previo al O(n²) — **máximo impacto**
**Problema**: Se evalúan pares sin ninguna garantía de proximidad espacial.
**Solución**: Antes del loop O(n²), construir un kd-tree o grid espacial sobre centroides.
Solo evaluar pares con centroid distance < th_centroid*2.
**Complejidad**: O(n log n) para construir + O(n × k) donde k = vecinos en radio.
**Impacto estimado**: Reducir 80-90% de pares evaluados en mapas grandes.

```python
# Pseudocódigo en _fuse_overlapping_instances:
centroids = np.stack([data[id][1] for id in ids])
from scipy.spatial import cKDTree
tree = cKDTree(centroids)
candidate_pairs = tree.query_pairs(r=th_centroid * 2)  # O(n log n)
# Solo iterar candidate_pairs en lugar de n²
```

#### 5.2 Point overlap en GPU — **alto impacto en tiempo de criterio**
**Problema**: compute_pcd_overlap crea objetos Open3D en CPU por par.
**Solución**: Implementar overlap con KD-tree en PyTorch/FAISS directamente en GPU.
Procesar batch de pares simultáneamente.
**Impacto**: De O(n_pairs × 5ms) a O(batch / GPU_throughput).

#### 5.3 Cosine similarity batch en GPU
**Problema**: `F.cosine_similarity()` llamada 1 vez por par en el chain.
**Solución**: Precalcular matriz de cosine similarity para todos los pares candidatos de una vez.
`features_matrix @ features_matrix.T` → O(n²×d) pero en GPU = 1 kernel.
**Impacto**: Elimina loop en Python para cos_sim. Importante si muchos pares llegan a este criterio.

### 🟡 Media prioridad

#### 5.4 Keyframe descriptor aggregation O(n²) → O(n)
**Problema**: L1 aggregation en `instance3d.py:162` es O(n_kfs²) por instancia.
**Solución**: Mantener descriptor agregado incrementalmente. En cada nuevo KF:
`descriptor = (descriptor * n + new_kf_descriptor) / (n + 1)` — O(1) por update.
Alternativa: Usar mediana online (Welford o similar).
**Crítica**: El L1 minimization "medoid" es más robusto que media — pero puede no merecer el coste.

#### 5.5 Co-occurrence graph: sets en lugar de lists
**Problema**: `cooccurrence_graph.py:13-14` usa `list.append()` + `if kf_id not in list` → O(n).
**Solución**: Cambiar `defaultdict(list)` a `defaultdict(set)`.
```python
# Antes:
if kf_id not in self._graph[i][j]:
    self._graph[i][j].append(kf_id)
# Después:
self._graph[i][j].add(kf_id)  # O(1), deduplication gratis
```
**Impacto**: Pequeño aislado, pero se llama por cada par en cada frame segmentado.

#### 5.6 Pinned memory para transfers GPU→CPU
**Problema**: `pcd.cpu().numpy()` en ovomapping.py:275 hace transfer síncrona.
**Solución**: Pre-alocar buffers en pinned memory y usar transfers async.
`torch.cuda.Stream()` para solapar transfers con cómputo CPU.

#### 5.7 Loop closure: re-fusionar solo instancias afectadas
**Problema**: Loop closure re-funde TODO el mapa (ovo.py:522-541).
**Solución**: Trackear qué instancias están en el área afectada por la corrección de trayectoria.
Solo re-funder instancias cuyo centroid se ha movido > epsilon tras la corrección.
**Complejidad**: Requiere pasar información del LC (qué poses cambiaron cuánto) al módulo semántico.

### 🟢 Baja prioridad / investigación

#### 5.8 Precompute SAM masks offline
Ya existe opción (`precomputed: true`). Explotarlo más agresivamente para benchmarks.
Guardar masks a disco una vez, reusar en múltiples runs con distintos parámetros de fusión.
**Impacto**: Elimina 30-150ms/frame de SAM para experimentos de fusión.

#### 5.9 Descriptor quantization
Almacenar CLIP features como float16 en lugar de float32.
SigLIP-384 tiene 768 dims × 4 bytes = 3KB por instancia por KF.
Con 500 instancias × 50 KFs = 75MB solo en features. Float16 → 37MB.
**Impacto en calidad**: Mínimo (cosine similarity es robusta a quantization).

---

## 6. Mejoras de calidad (ideas más especulativas)

### 6.1 Descriptor dinámico vs. estático
**Problema actual**: El descriptor CLIP se fija en las mejores K vistas y no evoluciona mucho.
**Idea**: Usar un descriptor que decay con el tiempo → vistas más recientes pesan más.
Útil si el mapa se está construyendo en tiempo real y la iluminación cambia.

### 6.2 Fusion confidence score
**Problema actual**: La fusión es binaria (fusionar / no fusionar).
**Idea**: Asignar un score de confianza a cada merge basado en cuántos criterios pasaron y con qué margen.
Instancias con baja confianza → candidatas para re-evaluar en loop closures.
**Output**: `Fusion_Confidence` column en fusion_decisions.csv.

### 6.3 Geometric shape descriptor para complement cosine
**Problema actual**: Dos objetos del mismo tipo semántico (dos sillas) pueden tener cos_sim alto
pero ser objetos distintos. El overlap ayuda, pero solo si están en el mismo punto del espacio.
**Idea**: Añadir un descriptor de forma (PCA de la nube de puntos → 3 eigenvalues).
Criterio: si los shapes son muy similares Y los centroides están cerca → misma instancia.

### 6.4 Hierarchical fusion
**Problema actual**: Fusión flat — todos los objetos se comparan entre sí.
**Idea**: Fusión en dos etapas:
1. Fusión local: solo pares vistos en el mismo keyframe (ya casi implementado con cooccurrence).
2. Fusión global: solo en loop closures, con criterios más relajados.
Reduce el scope del O(n²) a sub-grafos locales.

### 6.5 Online evaluation proxy
**Problema actual**: Para saber si los cambios funcionan hay que correr eval completo.
**Idea**: Durante el run, computar una métrica proxy rápida (instance consistency score):
¿Cuántas veces el mismo punto 3D es asignado a instancias distintas en frames consecutivos?
Alta inconsistencia → la fusión no está funcionando bien.
No requiere GT.

### 6.6 Adaptive fusion frequency
**Problema actual**: `map_every` es fijo.
**Idea**: Fusionar más frecuente cuando hay loop closure inminente (SLAM detecta loop candidate),
y menos frecuente en movimiento lineal estable.
Requiere señal del módulo SLAM.

---

## 7. Deuda técnica crítica

| Issue | Severidad | Ubicación |
|---|---|---|
| `keyframes["frame_id"]` es lista con "Deleted" strings | Alta | ovo.py:488 |
| Fusion O(n²) sin pruning espacial | Alta | ovo.py:567-579 |
| Open3D para operaciones triviales de distancia | Media | instance_utils.py:64-70 |
| L1 aggregation O(n²) por instancia | Media | instance3d.py:162 |
| No hay límite de instancias en memoria | Media | ovo.py:72 |
| Co-occurrence usa lists en lugar de sets | Baja | cooccurrence_graph.py:9-14 |
| `time.time()` × n_pairs × n_criteria | Baja | strategy.py:27-29 |

---

## 8. Orden de ataque recomendado

1. **Instrumentar métricas** (§1) — sin datos no hay decisiones informadas
2. **Pruning espacial pre-O(n²)** (§5.1) — mayor ROI, no cambia lógica
3. **Grid search thresholds** (§4.1) — bajo coste, puede mejorar mIoU sin cambiar código
4. **Precompute SAM** (§5.8) — desacopla experimentos de fusión del coste SAM
5. **Sets en co-occurrence** (§5.5) — trivial, sin riesgo
6. **Batch cos_sim** (§5.3) — requiere refactor del chain pero impacto claro
7. **Point overlap en GPU** (§5.2) — mayor esfuerzo, mayor ganancia
