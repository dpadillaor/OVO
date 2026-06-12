# OVO — Code-Level Optimizations

Mejoras concretas e implementables por módulo. Ordenadas por ROI dentro de cada sección.
Complementa `improvement_roadmap.md` (estrategia/experimentos); este doc es implementación.

---

## Fusion

### F1 — Reemplazar O3D KD-tree por matmul en `compute_pcd_overlap`

**Archivo**: `ovo/utils/instance_utils.py:43–70`  
**Speedup función**: ~5–15×  
**Speedup fusión total**: ~3–6×

#### Problema

Cada par que llega al criterio `overlap` hace:
1. `tensor.cpu().numpy()` × 2 — copia innecesaria si ya en CPU
2. `o3d.geometry.PointCloud()` × 2 — alloc Python
3. `o3d.utility.Vector3dVector(...)` × 2 — copia a formato interno O3D
4. `compute_point_cloud_distance` — construye KD-tree en big (~200–500μs overhead Python+C++)

El KD-tree se reconstruye en cada llamada; no se reutiliza entre pares.

#### Solución

Distancia euclídea mínima por brute-force con matmul. Para N,M < 500 puntos (tamaño típico de instancia), el overhead O3D supera el coste del brute-force O(N×M).

```python
def compute_pcd_overlap(points1: torch.Tensor, points2: torch.Tensor, th_points: float) -> float:
    if points1.shape[0] <= points2.shape[0]:
        small, big = points1, points2
    else:
        small, big = points2, points1

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a @ b.T)
    norm_s = (small ** 2).sum(dim=1, keepdim=True)   # (N, 1)
    norm_b = (big   ** 2).sum(dim=1, keepdim=True)   # (M, 1)
    dot    = small @ big.T                             # (N, M)
    dist2  = torch.clamp(norm_s + norm_b.T - 2 * dot, min=0.0)  # (N, M)

    # Comparar dist^2 < th^2 evita N*M sqrt
    min_dist2 = dist2.min(dim=1).values               # (N,)
    return (min_dist2 < th_points ** 2).float().mean().item()
```

**Notas**:
- `torch.clamp(..., min=0)` necesario: floating-point cancellation puede dar valores negativos pequeños cuando puntos son muy cercanos.
- Si N,M > ~2000 puntos, KD-tree puede ganar. Medir antes de descartar O3D para instancias grandes.
- `compute_pcd_old_overlap` recibe la misma mejora; no hace swap small/big (es asimétrica).

---

### F2 — Cache AABB en precompute de `obj_pcds`

**Archivo**: `ovo/entities/ovo.py:561–564` + `ovo/utils/instance_utils.py:27–40`  
**Speedup función**: ~10–20× para `AabbDistanceCriterion`  
**Speedup fusión total**: ~1.5–2.5×

#### Problema

`compute_aabb_distance` hace 4 scans O(N) por par (`min/max` × 2 nubes):

```python
# Llamado en AabbDistanceCriterion.check() — por cada par
min1, _ = points1.min(dim=0)   # scan O(N)
max1, _ = points1.max(dim=0)   # scan O(N)
min2, _ = points2.min(dim=0)   # scan O(N)
max2, _ = points2.max(dim=0)   # scan O(N)
```

Si hay 3000 pares y cada instancia tiene ~200 puntos: 3000 × 4 × 200 = 2.4M ops vectorizados + overhead kernel PyTorch × 12000 llamadas (~12ms solo en launches).

El AABB de una instancia no cambia entre pares — solo cambia si se añaden puntos nuevos.

#### Solución

Calcular AABB una vez en el bloque de precompute que ya existe:

```python
# ovo.py:561–564 — ya precomputa obj_pcds
for instance in objects_list:
    obj_pcd = points_3d[points_ins_ids == instance.id]
    centroid = obj_pcd.mean(dim=0)
    aabb_min, _ = obj_pcd.min(dim=0)
    aabb_max, _ = obj_pcd.max(dim=0)
    obj_pcds[instance.id] = [obj_pcd, centroid, (aabb_min, aabb_max)]
```

Modificar `AabbDistanceCriterion.check()` para recibir/usar las cajas cacheadas, o pasar `(aabb_min, aabb_max)` como parte de los datos del par en lugar de `points`.

**Alternativa más limpia**: `compute_aabb_distance` acepta `(min1, max1, min2, max2)` directamente en lugar de reconstruirlas de los points.

---

### F3 — Dot product en lugar de `F.cosine_similarity` (features normalizadas)

**Archivo**: `ovo/entities/fusion/criteria.py:93`  
**Speedup función**: ~1.5–2×  
**Speedup fusión total**: ~1.1×

#### Problema

```python
cos_sim = torch.nn.functional.cosine_similarity(f1, f2, dim=0).item()
```

`F.cosine_similarity` calcula: `dot(f1,f2) / (||f1|| × ||f2||)` — dos normas adicionales O(D) cada vez.

**Los features están normalizados L2 en origen** (confirmado):
- `clip_generator.py:123,130,134,180` — `F.normalize(..., p=2, dim=-1)`
- `sam3_generator.py:137,147` — `F.normalize(..., p=2, dim=-1)`
- `pe_generator.py:148,177` — `F.normalize(..., p=2, dim=-1)`
- `clips_merging.py:55` — `F.normalize(clips, dim=-1)`

Para features L2-normalizadas: `cosine_similarity(f1, f2) = f1 · f2`.

#### Solución

```python
# criteria.py:93
cos_sim = (f1 @ f2).item()
```

Añadir comentario si se quiere documentar el invariante:

```python
cos_sim = (f1 @ f2).item()  # features are L2-normalized at source
```

**Riesgo**: Si algún path almacena features sin normalizar (e.g., features cargadas de disco antes de que existiera normalización), esto produce resultados incorrectos silenciosamente. Verificar con `assert abs(f1.norm().item() - 1.0) < 1e-4` en debug antes de mergear.

---

### F4 — Batch cosine similarity (todos los candidatos de una vez)

**Archivo**: `ovo/entities/ovo.py:570–578` + `ovo/entities/fusion/strategy.py`  
**Speedup función**: ~30–50× para paso cos_sim  
**Speedup fusión total**: ~1.2–1.5×

#### Problema

El loop O(N²) en `ovo.py` llama a `strategy.same_instance()` par a par. `CosSimilarityCriterion` lanza un kernel Python separado por cada par que llega a ese criterio.

Para N=80 instancias, ~500 pares llegan a cos_sim: 500 llamadas × overhead Python ≈ 5ms solo en el paso cos_sim.

#### Solución

Precomputar la matriz completa antes del loop:

```python
# En _fuse_overlapping_instances, antes del O(n^2):
feature_attr = "clip_feature"  # o el que esté activo
ids = [ins.id for ins in objects_list]
features = torch.stack([getattr(ins, feature_attr)[0] for ins in objects_list])  # (N, D)
# features ya normalizadas → cos_sim_matrix[i,j] = dot product
cos_sim_matrix = features @ features.T  # (N, N) — 1 kernel
```

Luego en el loop, lookup en lugar de compute:

```python
cos_sim = cos_sim_matrix[i, j].item()  # O(1)
```

**Tradeoff**: Precomputa N² valores aunque la mayoría de pares se rechacen antes de llegar a cos_sim. Si cooccurrence/AABB rechazan el 90%, se computan ~9× más cos_sims de los necesarios. Rentable cuando N > ~50 y muchos pares superan los filtros previos.

**Prerequisito**: Refactorizar cómo `CosSimilarityCriterion` accede al valor — recibir como argumento o via `ctx`, no recalcular.

---

### Resumen de impacto

| ID | Mejora | Speedup función | Speedup fusión | Dificultad | Prerequisito |
|----|--------|----------------|----------------|------------|--------------|
| F1 | Matmul en pcd_overlap | 5–15× | 3–6× | Baja | — |
| F2 | Cache AABB | 10–20× | 1.5–2.5× | Baja | — |
| F3 | Dot product cos_sim | 1.5–2× | 1.1× | Trivial | Verificar normalización |
| F4 | Batch cos_sim | 30–50× | 1.2–1.5× | Media | F3 |

**Orden recomendado**: F1 → F2 → F3 → F4. F1+F2+F3 son cambios locales sin riesgo de regresión.  
Baseline total combinado (F1+F2+F3): ~6× sobre fusión completa (estimado, depende de N instancias y ratio de filtrado por criterio).

---

### F5 — `time.time()` por criterio por par en `strategy.py`

**Archivo**: `ovo/entities/fusion/strategy.py:30–32`  
**Speedup**: elimina ~7ms overhead puro por ciclo de fusión

#### Problema

```python
for criterion in self.criteria:
    t0 = time.time()
    verdict, decision = criterion.check(...)
    self._criterion_times[...] += (time.time() - t0)
```

3000 pares × 4 criterios × 2 syscalls = **24,000 `time.time()` por ciclo**. Cada syscall ~0.3μs → ~7ms de overhead solo en timing. Mayor que el coste real de algunos criterios.

#### Solución

Deshabilitar timing en producción con flag, o usar `time.perf_counter_ns()` (más barato), o solo acumular si `self._timing_enabled`:

```python
if self._timing_enabled:
    t0 = time.perf_counter_ns()
    verdict, decision = criterion.check(...)
    self._criterion_times[criterion.name] += time.perf_counter_ns() - t0
else:
    verdict, decision = criterion.check(...)
```

---

### F6 — Precompute `obj_pcds` con sort en lugar de K scans del mapa

**Archivo**: `ovo/entities/ovo.py:562–564`  
**Speedup función**: ~6–30× (depende de si points en GPU o CPU)  
**Ahorro absoluto**: ~0.3s (GPU) / ~5–14s (CPU) por run 2000 frames

#### Problema

```python
for instance in objects_list:
    obj_pcd = points_3d[points_ins_ids == instance.id]  # scan O(N_points) por instancia
```

K instancias × scan O(N_points) → O(K × N_points). Para K=80, N=100k: 8M comparaciones con overhead de kernel launch × K.

#### Solución

Ordenar `points_ins_ids` una vez y slicear por instancia:

```python
sort_idx = points_ins_ids.argsort()
sorted_ids = points_ins_ids[sort_idx]
sorted_pts = points_3d[sort_idx]
unique_ids, counts = sorted_ids.unique_consecutive(return_counts=True)
offsets = torch.cat([torch.tensor([0]), counts.cumsum(0)])
id_to_slice = {uid.item(): (offsets[i].item(), offsets[i+1].item())
               for i, uid in enumerate(unique_ids)}
for instance in objects_list:
    s, e = id_to_slice[instance.id]
    obj_pcd = sorted_pts[s:e]
```

Coste: O(N log N) argsort una vez + O(K) slices — sin scans repetidos.

---

### F7 — Batch `fuse_instances` — diferir remap del tensor

**Archivo**: `ovo/entities/ovo.py:579` + `ovo/utils/instance_utils.py:131`  
**Speedup tensor ops**: ~5–10×

#### Problema

```python
# instance_utils.py:131
points_ins_ids[points_ins_ids == instance2.id] = instance1.id
```

Llamado dentro del loop O(N²). Para F merges: F scans completos del tensor `points_ins_ids`. El remap in-place NO es necesario para las decisiones de fusión del loop — `obj_pcds` no se actualiza mid-loop de todas formas.

#### Solución

Acumular merges y aplicar en un único pass al final:

```python
# En _fuse_overlapping_instances, después del loop:
for src_id, dst_id in merge_map.items():
    points_ins_ids[points_ins_ids == src_id] = dst_id
# O con remap vectorizado si los ids son conocidos:
remap = torch.arange(points_ins_ids.max() + 1)
for src_id, dst_id in merge_map.items():
    remap[src_id] = dst_id
points_ins_ids = remap[points_ins_ids]  # 1 indexing op, cero scans
```

---

### F8 — `_remove_missing_instances`: `in` sobre tensor en lugar de set

**Archivo**: `ovo/entities/ovo.py:502`  
**Speedup**: trivial, ~1μs → ~0ms por objeto

```python
# Actual — O(K) por objeto
map_ins_ids = points_ins_ids.unique()
if ins_id in map_ins_ids:  # búsqueda lineal en tensor

# Fix — O(1) por objeto
map_ins_ids_set = set(points_ins_ids.unique().tolist())
if ins_id in map_ins_ids_set:
```

---

## Instance3D

### I1 — Medoid O(N²×D) — reemplazar por mediana O(N×D)

**Archivo**: `ovo/entities/instance3d.py:160–165` (y copia en `update_pe:194–199`, `update_sam3:~214–219`)  
**Speedup**: ~N× menos ops, N²× menos memoria intermedia  
**Impacto memoria**: Para N=50 kfs, D=512: actual=5MB tensor intermedio por instancia → nuevo=0

#### Problema

```python
clips = clips[:, None]                                        # (N, 1, D)
l1_distances = torch.abs(clips - clips.permute(1,0,2)).sum((1,2))  # (N, N, D) intermedio
kf = l1_distances.argmin()
self.clip_feature = clips[kf]
```

Crea tensor **(N × N × D)** para encontrar el medoid. Para N=50, D=512: 50×50×512×4 bytes = **5.2MB por objeto por update**. Además el patrón está copiado 3 veces (`update_clip`, `update_pe`, `update_sam3`).

#### Distinción medoid vs mediana

El código busca el **medoid** (punto del conjunto que minimiza suma de distancias L1 a los demás) — no la mediana componentwise. El comentario del docstring dice "equivalente a la mediana" pero es incorrecto: la mediana minimiza L1 pero puede no ser un punto del conjunto.

- **Medoid** (actual): O(N²×D) — punto real del conjunto
- **Mediana componentwise** (`clips.median(dim=0).values`): O(N×D) — minimizador L1 real, pero no necesariamente un KF observado

Si la aplicación acepta un descriptor que no es un KF real (solo un vector sintético), la mediana es correcta y mucho más barata:

```python
clips = torch.vstack(clips)                  # (N, D)
self.clip_feature = clips.median(dim=0).values.unsqueeze(0)  # (1, D)
self.to_update = False
```

Si se necesita un KF real (para `clip_feature_kf`), medoid es necesario — pero se puede calcular más eficientemente con `torch.cdist`:

```python
clips = torch.vstack(clips)                   # (N, D)
dists = torch.cdist(clips, clips).sum(dim=1)  # (N,) — usa matmul trick internamente
kf = dists.argmin()
self.clip_feature = clips[kf].unsqueeze(0)
```

#### DRY — refactorizar en método común

Los tres `update_clip/pe/sam3` son idénticos. Extraer:

```python
def _compute_medoid(self, embeds: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.vstack(embeds)
    dists = torch.cdist(stacked, stacked).sum(dim=1)
    return stacked[dists.argmin()].unsqueeze(0)
```

---

### I2 — `kfs_ids` como lista con membership check O(N)

**Archivo**: `ovo/entities/instance3d.py:73`

```python
if kf_id not in self.kfs_ids:        # O(N) scan de lista
    self.kfs_ids.append(kf_id)
```

Llamado por cada keyframe de cada instancia. Con objetos que acumulan muchos KFs, la búsqueda crece. Fix: `self.kfs_ids` como `set`, convertir a lista solo donde se necesite orden (e.g., serialización).

---

## CooccurrenceGraph

### C1 — Listas en lugar de sets

**Archivo**: `ovo/utils/cooccurrence_graph.py:10–16`  
**Speedup**: O(N) → O(1) por `increment`, llamado O(N²) veces por keyframe

#### Problema

```python
self.graph = defaultdict(lambda: defaultdict(list))

def increment(self, i, j, kf_id):
    if kf_id not in self.graph[i][j]:   # O(K_kfs) scan
        self.graph[i][j].append(kf_id)
    if kf_id not in self.graph[j][i]:   # O(K_kfs) scan
        self.graph[j][i].append(kf_id)
```

Con N=30 instancias por frame: 435 llamadas/frame × 2 scans × K_kfs acumulados → coste crece con el tiempo.

Además `merge()` hace conversión innecesaria:
```python
combined = set(self.graph[target][neighbor] + kfs)  # list+list → set → luego...
self.graph[target][neighbor] = list(combined)        # ...vuelve a list
```

#### Solución

```python
self.graph = defaultdict(lambda: defaultdict(set))

def increment(self, i, j, kf_id):
    self.graph[i][j].add(kf_id)   # O(1), dedup gratis
    self.graph[j][i].add(kf_id)

def merge(self, target, source):
    for neighbor, kfs in list(self.graph[source].items()):
        if neighbor == target: continue
        self.graph[target][neighbor] |= kfs   # set union O(K)
        self.graph[neighbor][target] |= kfs
        self.graph[neighbor].pop(source, None)
    del self.graph[source]
```

**Riesgo**: `to_dict()`/`from_dict()` usan `list(kfs)` — actualizar para convertir set→list en serialización.

---

## GeometryUtils

### G1 — `.repeat()` materializa tensor O(M×N×3)

**Archivo**: `ovo/utils/geometry_utils.py:19`

```python
dist = torch.norm(
    cluster_0.unsqueeze(1).repeat(1, n, 1).cpu() - cluster_1.cpu(),
    dim=-1
)
```

`.repeat(1, n, 1)` crea tensor **(M×N×3)** completo en memoria. Para M=N=5000: **300MB**. Además fuerza `.cpu()` explícito.

```python
# Fix directo
dist = torch.cdist(cluster_0, cluster_1)  # matmul trick, sin repeat, GPU-compatible
```

### G2 — `torch.tensor(list(range(n)))` en lugar de `torch.arange`

**Archivo**: `ovo/utils/geometry_utils.py:61`

```python
idx = torch.tensor(list(range(n_points)), device=device)  # crea lista Python primero
# Fix
idx = torch.arange(n_points, device=device)  # directo, sin alloc Python
```

### G3 — `torch.cat` añade columna de unos por frame

**Archivo**: `ovo/utils/geometry_utils.py:248`

```python
plane_product = torch.cat([points, ones], axis=1) @ frustum_planes.T
```

Crea tensor `(N, 4)` nuevo cada llamada concatenando `ones`. Para N=50k puntos: ~800kB de alloc + copy por llamada.

Fix con broadcasting o view homogéneo persistente. Alternativa: reformular producto plano para no necesitar coordenada homogénea (dot + offset separados).

---

## VanillaMapper

### V1 — `torch.vstack` incremental = O(N²) total de copias

**Archivo**: `ovo/slam/vanilla_mapper.py:81–84`  
**Impacto**: Alto — domina en runs largos  
**Dificultad**: Media

#### Problema

```python
self.pcd         = torch.vstack((self.pcd,         points[:,:3]))
self.pcd_ids     = torch.vstack((self.pcd_ids,     ...))
self.pcd_obj_ids = torch.vstack((self.pcd_obj_ids, ...))
self.pcd_colors  = torch.vstack((self.pcd_colors,  ...))
```

Cada `vstack` aloca un tensor nuevo y copia todo el PCD existente + nuevos puntos. Con 400 map steps añadiendo ~1000 pts/step (PCD final ~400k pts):

```
Total bytes copiados ≈ Σ(k × size_por_pt) = O(N²)
Al final del run: ~25MB de copia por map step × 4 tensores
```

#### Solución

Acumular en listas Python, flush en `get_map()`:

```python
# __init__: inicializar buffers
self._new_pcd, self._new_ids, self._new_obj, self._new_col = [], [], [], []

# map(): append en lugar de vstack
self._new_pcd.append(points[:,:3])
self._new_ids.append(...)
self._new_obj.append(...)
self._new_col.append(...)

# get_map(): flush al leer
def get_map(self):
    if self._new_pcd:
        self.pcd         = torch.cat([self.pcd]         + self._new_pcd)
        self.pcd_ids     = torch.cat([self.pcd_ids]     + self._new_ids)
        self.pcd_obj_ids = torch.cat([self.pcd_obj_ids] + self._new_obj)
        self.pcd_colors  = torch.cat([self.pcd_colors]  + self._new_col)
        self._new_pcd, self._new_ids, self._new_obj, self._new_col = [], [], [], []
    return self.pcd, self.pcd_ids, self.pcd_obj_ids.squeeze()
```

`torch.cat` sobre lista hace una sola alloc del tamaño final — O(N) en vez de O(N²).

---

### V2 — Meshgrid `(H, W)` recomputado cada frame

**Archivo**: `ovo/slam/vanilla_mapper.py:53`

```python
# Actual — cada map step
y, x = torch.meshgrid(torch.arange(h, device=self.device),
                       torch.arange(w, device=self.device), indexing="ij")
```

`h`, `w` son constantes (intrínsecas fijas). Fix: cachear en `__init__` y reusar.

```python
# __init__:
h, w = config["cam"]["H"], config["cam"]["W"]
self._y_grid, self._x_grid = torch.meshgrid(
    torch.arange(h, device=self.device),
    torch.arange(w, device=self.device), indexing="ij"
)
```

---

### V3 — `torch.linalg.inv(c2w)` recomputado cada map step

**Archivo**: `ovo/slam/vanilla_mapper.py:60`

```python
_, matches = geometry_utils.match_3d_points_to_2d_pixels(
    depth, torch.linalg.inv(c2w), self.pcd[frustum_mask], ...
)
```

`c2w` viene de `estimated_c2ws[frame_id]`. El inverso `w2c` podría almacenarse al hacer `track_camera` o pasarse ya calculado desde `ovomapping.py`, que tiene acceso a ambos.

---

### V4 — Columna de unos para homogéneas creada cada frame

**Archivo**: `ovo/slam/vanilla_mapper.py:78`

```python
points = torch.hstack((x_3d.reshape(-1,1), y_3d.reshape(-1,1), z_3d.reshape(-1,1),
                        torch.ones((x_3d.shape[0],1), device=self.device)))
points = torch.einsum("ij,mj->mi", c2w, points)
```

`torch.ones(N,1)` + `hstack` = alloc O(N) por frame. Alternativa: aplicar rotación y traslación por separado sin homogéneas:

```python
pts = torch.stack([x_3d, y_3d, z_3d], dim=1)           # (N,3)
pts = pts @ c2w[:3,:3].T + c2w[:3,3]                    # R·p + t, sin alloc extra
```

---

### V5 — `squeeze/unsqueeze` round-trip en `pcd_obj_ids`

**Archivo**: `ovo/slam/vanilla_mapper.py:100,154`

`pcd_obj_ids` guardado como `(N,1)`, `get_map()` devuelve `.squeeze()` → `(N,)`, `update_pcd_obj_ids()` hace `.unsqueeze(-1)` → `(N,1)`. Sin razón. Guardar como `(N,)` directamente y eliminar ambas conversiones.

---

### V6 — `max_frame_points` definido pero nunca aplicado

**Archivo**: `ovo/slam/vanilla_mapper.py:13`

```python
self.max_frame_points = config["mapping"].get("max_frame_points", 1e5)  # nunca usado
```

El PCD crece sin límite — sin subsampling, sin poda. Si se aplicara este límite (e.g., subsampling aleatorio o por distancia mínima entre puntos), controlaría la explosión de memoria y el coste O(N) de todas las operaciones posteriores.

---

### V7 — Frustum matching O(N_frustum) para enmascarar profundidad

**Archivo**: `ovo/slam/vanilla_mapper.py:58–61`

```python
frustum_mask = geometry_utils.compute_frustum_point_ids(self.pcd, ...)
_, matches = geometry_utils.match_3d_points_to_2d_pixels(
    depth, torch.linalg.inv(c2w), self.pcd[frustum_mask], ...
)
mask[matches[:,1], matches[:,0]] = False
```

Propósito: evitar reprojectar profundidad donde ya hay puntos 3D. Pero requiere transformar N_frustum puntos a espacio cámara, proyectarlos y matchearlos. Para N_frustum=50k: ~5ms por map step.

Alternativa: mantener buffer 2D `(H, W)` de píxeles ya ocupados, actualizar al añadir puntos. Lookup O(1) en lugar de O(N_frustum):

```python
# Al añadir puntos nuevos, marcar sus píxeles origen como ocupados
self._occupied[new_y_pixels, new_x_pixels] = True
# En map():
mask &= ~self._occupied  # directo, sin proyección
```

Requiere mantener correspondencia pixel→punto pero elimina el frustum matching por completo.

---

### Resumen VanillaMapper

| ID | Problema | Impacto | Dificultad |
|----|----------|---------|------------|
| V1 | vstack incremental O(N²) | Alto — domina en runs largos | Media |
| V7 | frustum matching O(N_frustum)/step | Medio-Alto en mapas grandes | Media |
| V4 | alloc columna ones cada frame | Bajo | Trivial |
| V3 | inv(c2w) recomputado | Bajo | Trivial |
| V2 | meshgrid recomputado | Bajo | Trivial |
| V5 | squeeze/unsqueeze round-trip | Cosmético | Trivial |
| V6 | max_frame_points sin aplicar | Control memoria no implementado | Baja |
