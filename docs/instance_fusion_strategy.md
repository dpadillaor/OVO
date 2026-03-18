# Estrategia de Fusión de Instancias Semánticas

Documento de investigación y análisis sobre mejoras a la lógica de fusión de instancias 3D en OVO.
Cubre el estado actual del sistema, análisis de casos, señales disponibles, revisión de literatura y propuestas concretas.

---

## Índice

1. [Estado actual del sistema](#1-estado-actual-del-sistema)
2. [Bugs identificados](#2-bugs-identificados)
3. [Taxonomía de casos](#3-taxonomía-de-casos)
4. [Señales disponibles y propuestas](#4-señales-disponibles-y-propuestas)
5. [Mejoras a los descriptores semánticos](#5-mejoras-a-los-descriptores-semánticos)
6. [Consistencia global: enfoque de grafos](#6-consistencia-global-enfoque-de-grafos)
7. [Revisión de literatura (2022–2025)](#7-revisión-de-literatura-20222025)
8. [Síntesis de recomendaciones priorizadas](#8-síntesis-de-recomendaciones-priorizadas)
9. [Papers relevantes](#9-papers-relevantes)

> **Nota**: La sección §4.9 describe el uso del grafo de covisibilidad del SLAM como señal adicional. Es una mejora futura que requiere que el backend SLAM lo exponga (ORB-SLAM3 lo tiene, otros backends pueden no tenerlo).

---

## 1. Estado actual del sistema

### Dónde ocurre la fusión

La fusión se ejecuta en `ovo/entities/ovo.py::update_map()`, que es llamado por el orquestador principal cuando el SLAM detecta un loop closure o realiza una actualización global del mapa. El flujo es:

```
update_map()
  └─ complete_semantic_info()         # vacía la queue de descriptores pendientes
  └─ _remove_missing_instances()      # elimina instancias sin puntos en el mapa actual
  └─ _fuse_overlapping_instances()    # ← AQUÍ ocurre la fusión
  └─ _update_descriptors_after_fusion()
  └─ update_objects_clip() / fusion_encoder.update_objects()
```

### Lógica de decisión actual (`SemanticGeometricFusion`)

```python
# Archivo: ovo/entities/fusion.py

def same_instance(inst1, inst2, pts_centroid1, pts_centroid2) -> bool:
    # Paso 1: distancia de centroides (filtro rápido)
    if centroid_distance(c1, c2) > th_centroid:   # default: 1.5m
        return False

    # Paso 2: similitud coseno del descriptor semántico
    cos_sim = cosine_similarity(feat1, feat2)
    if cos_sim < th_cossim:                        # default: 0.81
        return False

    # Paso 3: overlap de nube de puntos
    p_dist = compute_pcd_overlap(pts1, pts2)       # th_points default: 0.1m

    return p_dist > 0.5 or (cos_sim > 0.9 and p_dist > 0.2)
```

Los criterios son secuenciales (AND) con una relajación final: alta similitud semántica puede compensar overlap moderado.

### Modos de fusión disponibles

| `fusion_method` | Feature usada | Clase |
|---|---|---|
| `clip` | `clip_feature` (siempre disponible) | `SemanticGeometricFusion` |
| `dino` | `dino_feature` | `SemanticGeometricFusion` |
| `pe` | `pe_feature` | `SemanticGeometricFusion` |
| `sam3` | `sam3_feature` | `SemanticGeometricFusion` |
| `geometric` | ninguna | `GeometricOnlyFusion` |

### Agregación de descriptores

Los descriptores de cada instancia se agregan por **L1-median** a lo largo de los keyframes observados: se selecciona el keyframe cuyo descriptor minimiza la distancia L1 respecto al resto. Esto equivale a elegir la observación más "mediana", robusta a outliers de vistas atípicas. Se hace en `Instance3D.update_clip()`, `update_pe()`, `update_sam3()`.

Opcionalmente se usa un heap de los `n_top_kf` keyframes con mayor área de máscara (`top_kf`), priorizando las observaciones donde el objeto es más prominente.

---

## 2. Bugs identificados

### Bug 1: Overlap asimétrico

```python
# ovo/utils/instance_utils.py::compute_pcd_overlap
dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
return (dists < th_points).astype(float).mean()
```

Solo mide qué fracción de los puntos de `pcd1` están cerca de `pcd2`. No mide lo contrario. Consecuencias:

- Si `pcd1` es pequeño y está contenido dentro de `pcd2`, `overlap_12 ≈ 1.0` pero `overlap_21 ≈ 0.1`.
- Dependiendo del orden de iteración en el bucle O(n²), el mismo par de instancias puede dar resultados opuestos.
- Un objeto pequeño (cojín) puede ser incorrectamente fusionado con uno grande (sofá) aunque sean instancias distintas.

**Fix**: calcular en ambas direcciones y usar la versión simétrica apropiada al caso (ver §4.3).

### Bug 2: Inconsistencia transitiva del greedy

El loop actual:
```python
for i, inst1 in enumerate(objects_list):
    for inst2 in objects_list[i+1:]:
        if same_instance(inst1, inst2):
            fuse(inst1, inst2)  # inst1 muta en A+B
```

Si A+B se fusionan, el `inst1` que se compara con C ya es la instancia A+B fusionada. El orden de iteración afecta el resultado. Dos instancias que NO deberían fusionarse pueden acabar fusionadas transitivamente: A≈B (se fusionan), B≈C (C es compatible con B pero no con A), A≇C.

**Fix**: enfoque de componentes conexas (ver §6).

### Bug 3: La condición relajada es demasiado agresiva

```python
return p_dist > 0.5 or (cos_sim > 0.9 and p_dist > 0.2)
```

La segunda condición permite fusionar con solo un 20% de overlap si la similitud semántica es alta. Esto puede fusionar incorrectamente dos objetos de la misma categoría colocados uno al lado del otro (dos papeleras adyacentes), ya que su similitud coseno puede superar 0.9 y pueden tener un 20% de overlap si están muy juntas.

---

## 3. Taxonomía de casos

Comprender qué casos debe discriminar el sistema es fundamental para diseñar señales útiles.

| ID | Caso | Geo. | Semántica | Co-ocurrencia | Esperado |
|----|------|------|-----------|---------------|----------|
| **A** | Loop closure — mismo objeto, dos recorridos distintos | 0 overlap (tiempos distintos), AABB parecido, vol. similar | similitud alta | nunca co-ocurrieron juntos | **FUSIONAR** |
| **B** | Observación parcial → completa — mismo objeto con más puntos después | overlap asimétrico: partial⊂completo, vol. distinto | similitud alta | pueden co-ocurrir o no | **FUSIONAR** |
| **C** | Over-segmentation — planta en ramas, silla en partes | sin overlap, AABBs adyacentes o solapados, coherentes juntos | similitud muy alta (≈1.0) | a veces co-ocurren en mismo frame como distintas | **FUSIONAR** |
| **D** | Mismo tipo, instancias distintas — dos papeleras juntas | sin overlap, centroides próximos, vol. similar | similitud muy alta (≈1.0) | probablemente co-ocurren en mismo frame | **NO FUSIONAR** |
| **E** | Objeto pequeño encima/dentro de objeto grande — cojín en sofá | overlap asimétrico alto para el pequeño, vol. muy distinto | similitud baja-media | frecuente co-ocurrencia | **NO FUSIONAR** |
| **F** | Objetos en contacto pero distintos — TV sobre mueble TV | adyacentes o tocan, AABBs distintos | semanticamente distintos | co-ocurren | **NO FUSIONAR** |
| **G** | Mismo objeto, mismo recorrido, más observaciones | overlap alto (mismos puntos) | similitud alta | co-ocurren (mismo objeto en frames distintos) | **FUSIONAR** |
| **H** | Dos objetos distintos que accidentalmente tienen overlap de puntos por imprecisión del SLAM | overlap bajo-medio | semánticamente distintos | co-ocurren | **NO FUSIONAR** |

### Los pares más peligrosos

**Caso A vs D**: misma firma semántica (alta similitud), centroides próximos, bajo overlap. La diferencia clave es la **co-ocurrencia**: en el caso D las dos papeleras siempre han aparecido juntas en el mismo keyframe. En el caso A, el mismo objeto fue visto en recorridos distintos y nunca co-ocurrió.

**Caso C vs D**: over-segmentation de un objeto vs dos objetos del mismo tipo. La diferencia es que en C los fragmentos forman un AABB coherente y compacto, mientras que en D los dos objetos tienen AABBs separados aunque la distancia de centroides pueda ser similar.

---

## 4. Señales disponibles y propuestas

### 4.1 Co-ocurrencia en keyframes (señal nueva, alta prioridad)

**La señal más poderosa que no se usa, y que está disponible sin coste adicional.**

```python
shared_kfs = set(inst1.kfs_ids) & set(inst2.kfs_ids)
```

`Instance3D.kfs_ids` ya almacena todos los keyframes donde fue observada cada instancia. Si dos instancias co-ocurrieron frecuentemente en los mismos keyframes, SAM2 las distinguió explícitamente en esos frames → evidencia fuerte de que son objetos distintos.

| Patrón | Interpretación |
|--------|---------------|
| `len(shared_kfs)` alto, similitud semántica alta | Mismo tipo, instancias distintas (caso D) → **NO fusionar** |
| `len(shared_kfs)` = 0, similitud semántica alta | Probablemente loop closure (caso A) → **candidato a fusionar** |
| `len(shared_kfs)` bajo pero distinto de 0 | Ambiguo → ponderar con otras señales |

**Excepción**: el over-segmentation (caso C) puede tener co-ocurrencia alta (SAM2 vio las ramas juntas y las segmentó por separado). Pero ahí la señal de adyacencia del AABB o la similitud extrema (≈1.0) puede distinguirlo.

Implementación práctica: usar como **penalización** en un score combinado, no como veto absoluto, con un peso alto.

### 4.2 Hard veto por ratio de volumen/puntos

```python
n1, n2 = len(pts1), len(pts2)
point_ratio = max(n1, n2) / min(n1, n2)
if point_ratio > 5.0:  # diferencia de 5x en puntos
    return False
```

Si una instancia tiene 5 veces más puntos (o volumen de AABB) que la otra, difícilmente son el mismo objeto. La única excepción sería una observación muy parcial, pero en ese caso los descriptores semánticos serán igualmente poco fiables.

Esto es un hard veto barato: O(1), sin acceso a Open3D, y con muy pocos falsos positivos.

### 4.3 Overlap simétrico con detección de contenido

Tres métricas complementarias a calcular con ambas direcciones:

```python
overlap_12 = frac(pts1 near pts2)   # solo esto existe actualmente
overlap_21 = frac(pts2 near pts1)   # NUEVO

sym_overlap   = min(overlap_12, overlap_21)  # ambas se solapan mutuamente
max_overlap   = max(overlap_12, overlap_21)  # al menos una contiene a la otra
containment   = max_overlap - sym_overlap    # grado de asimetría (A dentro de B)
```

Uso por caso:
- **Caso A/G** (loop closure, mismo objeto): `sym_overlap` alto después de la corrección geométrica.
- **Caso B** (observación parcial→completa): `containment` alto, `sym_overlap` bajo. La instancia pequeña está contenida en la grande.
- **Caso C** (over-segmentation): todos los overlaps bajos, pero AABBs adyacentes.
- **Caso D** (papeleras): todos los overlaps bajos o cero → no fusionar.

### 4.4 IoU 3D del AABB como filtro espacial

En lugar (o además) de la distancia de centroides:

```
IoU3D = vol(intersección de AABBs) / vol(unión de AABBs)
```

- **Más informativo que el centroide** para objetos grandes o elongados: una estantería y un libro sobre ella pueden tener centroides a 0.3m pero IoU3D bajo porque la estantería es enorme.
- **Más rápido** que el overlap de nube de puntos.
- Puede usarse como **pre-filtro**: si IoU3D = 0 y distancia centroide > umbral_grande → descartar sin calcular nada más.

Nota: la literatura (ConceptGraphs, HOV-SG) confirma que el overlap de nube de puntos es preferido sobre el AABB IoU para la decisión final porque el AABB es sensible a objetos no convexos. Pero el AABB IoU como pre-filtro es estándar.

### 4.5 Variante voxelizada del overlap (más robusta a ruido)

En lugar de comparar puntos crudos, downsamplear ambas nubes a una rejilla de vóxeles y hacer set intersection/union:

```python
voxels1 = set(voxelize(pts1, voxel_size=0.05))
voxels2 = set(voxelize(pts2, voxel_size=0.05))
iou_vox = len(voxels1 & voxels2) / len(voxels1 | voxels2)
```

Más robusto para nubes de densidad variable (un objeto visto de cerca tiene muchos puntos, el mismo objeto de lejos tiene pocos). Con la métrica de puntos crudos, la densidad afecta el resultado; con la voxelizada no.

### 4.6 Compatibilidad de dimensiones del AABB

Las dimensiones ordenadas del bounding box `(h, w, d)` son invariantes al punto de vista (siempre que se tenga suficiente cobertura). Una instancia muy elongada verticalmente (poste) y una compacta (silla) tienen AABBs incompatibles incluso si su similitud CLIP es parecida.

```python
dims1 = sorted(aabb1.extent)  # [min_dim, mid_dim, max_dim]
dims2 = sorted(aabb2.extent)
elongation_ratio = (dims1[2]/dims1[0]) / (dims2[2]/dims2[0])
# si elongation_ratio >> 1: formas muy distintas → no fusionar
```

Útil como señal negativa (veto por incompatibilidad de forma), no como señal positiva.

### 4.7 Threshold adaptativo por confianza de descriptor

Un descriptor estimado de 2 observaciones no debería tener el mismo peso que uno de 100. Con pocas observaciones, el feature puede capturar una vista atípica.

```python
conf1 = min(len(inst1.kfs_ids), N_max) / N_max
conf2 = min(len(inst2.kfs_ids), N_max) / N_max
min_conf = min(conf1, conf2)

# Threshold más bajo cuando una instancia está poco observada
adaptive_th = th_cossim_max - (1 - min_conf) * slack
# ej: th_cossim_max=0.85, slack=0.1 → con conf=0: th=0.75; con conf=1: th=0.85
```

Esto permite fusionar más agresivamente instancias recién creadas (donde el feature es incierto) y ser más estricto con instancias bien consolidadas.

### 4.9 Grafo de covisibilidad del SLAM *(señal futura — requiere soporte del backend)*

> Esta señal no está disponible en el sistema actual. Requiere que el backend SLAM exponga su grafo de covisibilidad entre keyframes. ORB-SLAM3 lo tiene; otros backends (GT, etc.) pueden no tenerlo.

#### Qué es el grafo de covisibilidad

Los sistemas SLAM basados en keyframes (como ORB-SLAM3) mantienen internamente un **grafo de covisibilidad**: un grafo donde los nodos son keyframes y las aristas conectan pares de keyframes que observaron un número mínimo de puntos 3D del mapa en común. El peso de la arista es el número de puntos compartidos — cuanto mayor, más "vecinos espaciales" son esos dos keyframes.

Este grafo es diferente de la co-ocurrencia de instancias que ya tenemos (`kfs_ids`):

| Señal | Qué mide | Granularidad |
|-------|----------|-------------|
| `kfs_ids` (actual) | "¿Apareció esta instancia en este keyframe?" | Por instancia |
| Covisibilidad SLAM | "¿Cuántos puntos del mapa comparten estos dos keyframes?" | Entre pares de keyframes |

La covisibilidad es una propiedad del mapa global, independiente de las instancias semánticas.

#### Cómo ayudaría a la fusión

**Caso 1 — Pre-filtrado de candidatos O(N²) → O(N·K)**

Actualmente se comparan todas las instancias entre sí. Con el grafo de covisibilidad, dos instancias solo son candidatas a fusión si sus keyframes de observación están conectados en el grafo (o son los mismos):

```python
# Solo comparar inst1 e inst2 si algún keyframe de inst1
# es covisible con algún keyframe de inst2
kfs1 = set(inst1.kfs_ids)
kfs2 = set(inst2.kfs_ids)
covisible = any(
    covisibility_graph.are_covisible(kf1, kf2, min_weight=30)
    for kf1 in kfs1 for kf2 in kfs2
)
if not covisible:
    continue  # nunca se vieron desde regiones espacialmente próximas
```

Esto puede reducir drásticamente el número de comparaciones en escenas grandes, pasando de O(N²) a O(N·K) donde K es el número medio de vecinos covisibles de cada instancia.

**Caso 2 — Señal de loop closure a nivel de instancia**

ORB-SLAM3 detecta loop closures y los registra como aristas especiales en el grafo de poses. Si dos keyframes están conectados por una arista de loop closure, los objetos observados en esos keyframes son candidatos naturales a ser el mismo objeto:

```python
loop_closure_pairs = slam.get_loop_closure_edges()  # [(kf_A, kf_B), ...]
for kf_A, kf_B in loop_closure_pairs:
    instances_A = get_instances_in_keyframe(kf_A)
    instances_B = get_instances_in_keyframe(kf_B)
    # Los pares (inst_a, inst_b) son candidatos a fusión con threshold relajado
    for inst_a in instances_A:
        for inst_b in instances_B:
            if semantic_similarity(inst_a, inst_b) > th_loop_closure:
                fuse(inst_a, inst_b)
```

Esto permite usar un threshold geométrico muy relajado o incluso eliminarlo para los pares identificados por loop closure, ya que el SLAM ya certifica que esas regiones son el mismo lugar.

**Caso 3 — Señal negativa reforzada (complementa la co-ocurrencia de instancias)**

La co-ocurrencia actual (`kfs_ids`) detecta si dos instancias aparecieron en el **mismo** keyframe. La covisibilidad añade un nivel más: si los keyframes de inst1 y los keyframes de inst2 tienen **covisibilidad fuerte** (muchos puntos compartidos, observaciones espacialmente solapadas) pero las instancias nunca se fusionaron en el tracking 2D → SAM las vio como objetos distintos desde prácticamente la misma posición → señal muy fuerte de que son objetos distintos.

```
covisibilidad_media(kfs1, kfs2) > umbral_alto
    Y len(kfs1 ∩ kfs2) = 0  (nunca en el mismo frame)
    → Observadas desde perspectivas muy parecidas pero siempre separadas
    → Evidencia fuerte de instancias distintas
```

Mientras que la co-ocurrencia simple dice "las vi juntas en el mismo frame", la covisibilidad dice "las vi desde prácticamente el mismo sitio y seguían siendo distintas".

**Caso 4 — Covisibilidad cero como señal positiva de loop closure**

Si inst1 tiene keyframes en `{kf_10, kf_11, kf_12}` e inst2 en `{kf_85, kf_86}`, y esos keyframes no tienen **ninguna** covisibilidad entre sí (recorridos completamente separados en el tiempo), pero sus posiciones 3D después de la corrección son similares → muy probablemente es el mismo objeto visto en dos recorridos distintos.

```
covisibilidad(kfs1, kfs2) ≈ 0       # recorridos distintos
Y distancia_centroide(inst1, inst2) < th_post_corrección
Y similitud_semántica(inst1, inst2) > th_semantico
→ Candidato fuerte a loop closure → fusionar
```

Este patrón invierte la intuición habitual: la ausencia de covisibilidad entre los keyframes de dos instancias similares es precisamente la evidencia de que son el mismo objeto visto en momentos distintos.

#### Diferencia conceptual respecto a la co-ocurrencia actual

```
kfs_ids actual:
  "inst1 apareció en kf_A, inst2 en kf_A → estaban en el mismo frame"
  → Señal binaria, local, por instancia

Covisibilidad SLAM:
  "¿Qué fracción del campo de visión de kf_A solapa con kf_B?"
  → Señal continua, global, relacional entre keyframes
  → Permite razonar sobre gradaciones de "misma región" vs "región distante"
```

#### Consideraciones de implementación

Para exponer esta información desde ORB-SLAM3 habría que:
1. Añadir un método en el wrapper `ovo/slam/` que devuelva el grafo de covisibilidad o consultas sobre él.
2. Pasar ese grafo (o una versión resumida) al método `_fuse_overlapping_instances` en `ovo.py`.
3. Para backends sin covisibilidad (GT, etc.), degradar graciosamente al comportamiento actual.

La interfaz podría ser tan simple como:
```python
class SLAMBackend:
    def get_covisible_keyframes(self, kf_id: int, min_weight: int = 30) -> List[int]:
        """Keyframes covisibles con kf_id con al menos min_weight puntos compartidos."""
        ...
    def get_loop_closure_pairs(self) -> List[Tuple[int, int]]:
        """Pares de keyframes conectados por loop closure edges."""
        ...
```

---

### 4.8 Threshold adaptativo post loop-closure

Después de la corrección global de poses (`correct_map_globally()`), la alineación mejora pero no es perfecta. Se puede relajar temporalmente el threshold geométrico solo durante el pase de fusión post-corrección:

```python
# En update_map(), detectar si se está en contexto post-corrección
if post_loop_correction:
    original_th = self.fusion_strategy.th_centroid
    self.fusion_strategy.th_centroid *= 2.0  # relajar temporalmente
    result = self._fuse_overlapping_instances(...)
    self.fusion_strategy.th_centroid = original_th
```

---

## 5. Mejoras a los descriptores semánticos

### 5.1 Máscara binaria vs recorte de bounding box

Actualmente los descriptores se extraen sobre el recorte de bounding box de la imagen, que incluye fondo. Usar la **máscara binaria de SAM** para eliminar el fondo antes de extraer el descriptor reduce la contaminación semántica del entorno.

- **Para SAM3**: SAM3 ya genera la máscara, aplicarla es casi gratuito.
- **Para PE/DINO**: son modelos basados en patches. Enmascarar el fondo antes de pasarlo al encoder es estándar en la literatura y bien soportado.
- **Para CLIP**: se puede hacer padding con negro/gris neutro en la zona enmascarada. La comunidad documenta mejoras consistentes con esta técnica.

### 5.2 Features intermedias vs output final

Para encoders como SAM3 (basado en ViT), los tokens intermedios del image encoder (antes del mask decoder) codifican textura y estructura del objeto sin el sesgo de la tarea de segmentación. Extraer estas representaciones intermedias puede proporcionar descriptores más ricos y más discriminativos para comparación inter-instancia.

Esto es análogo a cómo DINO extrae features del ViT pre-entrenado sin finetunear para segmentación — los tokens son más generales y más útiles para reconocimiento de partes y apariencia local.

### 5.3 DINO + CLIP como señales complementarias, no alternativas

Actualmente el sistema tiene modos excluyentes: o usas CLIP, o usas DINO, o usas PE. La literatura (especialmente Gaussian Grouping y ConceptGraphs) documenta que la combinación es mejor:

- **CLIP**: bueno para categoría semántica global, viewpoint-dependiente.
- **DINO**: bueno para similitud de apariencia local (partes, textura), más invariante a viewpoint.

Para el over-segmentation: el respaldo y el asiento de una silla pueden tener CLIP similar pero DINO muy diferente (materiales distintos) o muy similar (mismo tejido). Usarlos juntos como weighted sum del score da más información.

```python
score = w_clip * cosine_sim(clip1, clip2) + w_dino * cosine_sim(dino1, dino2)
```

### 5.4 Varianza del descriptor como señal de confianza

Si los embeddings de una instancia a lo largo de sus keyframes tienen alta varianza, ese objeto es muy sensible al punto de vista → su descriptor representativo (L1-median) es menos fiable. Se puede usar la varianza intra-instancia para ponderar dinámicamente el threshold de similitud:

```python
var1 = compute_feature_variance(all_clips_of_inst1)
var2 = compute_feature_variance(all_clips_of_inst2)
# Mayor varianza → threshold más bajo (menos exigente con la similitud)
```

### 5.5 Comparación multi-vista en lugar de descriptor único

En vez de comparar solo el descriptor "representativo" (L1-median), comparar las nubes completas de descriptores por pares y quedarse con la similitud máxima:

```python
max_sim = max(cosine_sim(f1_i, f2_j) for f1_i in feats1 for f2_j in feats2)
```

Esto captura mejor el caso loop-closure donde las instancias solo comparten algunas vistas compatibles (las que tienen ángulos de cámara parecidos).

### 5.6 Re-computar el descriptor tras fusión

Actualmente cuando se fusionan A+B, el descriptor resultante es el que tenía el sobreviviente (A). Una mejora sería re-computar el L1-median con el pool combinado de keyframes:

```python
# En _update_descriptors_after_fusion:
# Actualmente: transfiere descriptores de B a A
# Mejorado: llama a update_clip() con kfs_ids de A ∪ B
instance1.to_update = True  # forzar re-cómputo con pool combinado
```

La infraestructura ya existe (`update_clip`, `update_pe`, `update_sam3` con `force_update=True`).

---

## 6. Consistencia global: enfoque de grafos

### El problema con el greedy

El bucle actual fusiona instancias de forma incremental y mutable. Una vez que A+B se fusionan, el resultado afecta las comparaciones posteriores con C, D, etc. Esto puede llevar a fusiones en cadena incorrectas.

**Ejemplo de fallo**:
- A y B son compatibles → se fusionan en A+B
- A+B y C parecen compatibles (porque B era compatible con C, aunque A no lo fuera)
- Resultado: A, B, C se fusionan aunque A y C sean objetos distintos

### Componentes conexas (OVIR-3D approach)

La solución correcta es calcular **todas** las compatibilidades primero, construir el grafo, y luego fusionar solo los grupos internamente consistentes.

```python
# Paso 1: calcular todas las aristas del grafo de compatibilidad
edges = {}  # {(id1, id2): True/False}
for i, inst1 in enumerate(objects_list):
    for inst2 in objects_list[i+1:]:
        edges[(inst1.id, inst2.id)] = fusion_strategy.same_instance(
            inst1, inst2, pcds[inst1.id], pcds[inst2.id]
        )

# Paso 2: encontrar componentes conexas
components = find_connected_components(objects_list, edges)

# Paso 3: fusionar solo si TODOS los pares dentro del componente son compatibles
for component in components:
    if all_pairs_compatible(component, edges):
        merge_all(component)
```

La condición "todos los pares compatibles" previene la fusión transitiva incorrecta: para fusionar {A, B, C} se requiere que A≈B **Y** A≈C **Y** B≈C sean todos verdaderos.

Variante menos estricta: usar un **umbral de arista mínima** dentro del componente (el 80% de los pares deben ser compatibles, o la arista más débil debe superar un umbral).

### Complejidad

- Construcción del grafo: O(N²) — igual que el greedy actual
- Componentes conexas: O(N + E) con union-find — prácticamente gratuito
- La diferencia es la semántica del resultado, no la eficiencia

---

## 7. Revisión de literatura (2022–2025)

### ConceptGraphs — Gu et al., ICRA 2024 (arxiv 2309.16650)

**El sistema más similar a OVO en filosofía y arquitectura.** Mantiene un grafo de objetos 3D con embeddings (CLIP + DINO), nube de puntos y bounding box.

**Criterio de fusión**: punto cloud overlap (IoU) > threshold **Y** CLIP cosine similarity > threshold. Criterio conjunto (AND), ambos deben pasar.

**Over-segmentation**: paso de NMS 2D antes de levantar a 3D — si dos máscaras SAM tienen IoU 2D > 0.6, suprimir la más pequeña antes de crear Instance3D. Segundo pase de fusión en 3D cuando el overlap de nubes supera el 50%.

**Limitaciones documentadas**: no gestiona bien loop closure (sin mecanismo de threshold adaptativo post-corrección). Over-segmentation es un problema abierto para objetos complejos.

### OVIR-3D — Lu et al., ICCV 2023 (arxiv 2309.00809)

**Referencia directa para la solución de transitividad.** Aborda explícitamente el problema del greedy y propone componentes conexas sobre el grafo de compatibilidad como reemplazo drop-in.

**Señales adicionales**: ratio de volumen como gate previo; categoría CLIP como filtro de pre-selección (solo comparar instancias con top-k categorías compatibles).

### HOV-SG — Werby et al., RSS 2024 (arxiv 2309.14308)

Grafo de escena jerárquico (piso → habitación → objeto). Asociación por re-proyección: overlap 3D primero, luego similitud semántica, luego consistencia de escala. Tres gates en secuencia.

**Over-segmentation**: paso de graph-cut post-hoc: dos nodos con IoU proyectado > 0.5 y CLIP similarity > 0.85 se fusionan.

**Señal de tamaño**: documenta bien el uso del ratio de volumen del AABB como gate previo duro.

### Gaussian Grouping — Ye et al., ECCV 2024 (arxiv 2312.00732)

3D Gaussian Splatting con campos de identidad por Gaussiana. Segmentación por tracking 2D consistente (DINO identity embeddings) propagado a través de frames de entrenamiento. Los Gaussianos votan el ID que mejor reconstruye las máscaras 2D.

**Mejor tratamiento del over-segmentation**: dos máscaras SAM con embeddings DINO muy similares que siempre co-aparecen → se fusionan en el tracking 2D antes de levantarse a 3D. Los objetos 3D heredan identidades ya consolidadas.

**Lección clave**: resolver la co-identidad en 2D antes de levantar a 3D es más limpio que fusionar en 3D a posteriori.

### OpenMask3D — Takmaz et al., NeurIPS 2023 (arxiv 2309.10171)

Sistema offline: segmentación 3D con Mask3D (transformer sobre nube completa), luego embeddings CLIP multi-vista enmascarados. **No tiene problema de fusión online** porque trabaja sobre la nube completa y final. Referencia para el caso offline o post-procesado.

**Lección**: para escenarios offline (post loop-closure, evaluación final), correr un segmentador 3D aprendido sobre la nube corregida puede ser mejor que la fusión online greedy.

### ConceptFusion — Jatavallabhula et al., RSS 2023 (arxiv 2211.11684)

Sistema de features densas por surfel, sin instancias. Fusión multi-modelo (CLIP + DINO + LSeg + SAM) por acumulación en el mapa denso. Las instancias se definen en tiempo de query, no en tiempo de construcción.

**Lección**: almacenar features por punto y diferir la agrupación puede evitar el problema de premature commitment que tiene el sistema online de OVO.

### OpenScene — Peng et al., CVPR 2023 (arxiv 2211.15654)

Destilación de features 2D open-vocabulary (LSeg/OpenSeg) en nube 3D por back-projection. Sin segmentación de instancias online. La agrupación se hace en post-procesado con DBSCAN o segmentadores 3D separados.

### Resumen comparativo

| Sistema | Criterio geométrico | Criterio semántico | Over-seg | Loop closure | Algoritmo |
|---------|--------------------|--------------------|----------|--------------|-----------|
| **OVO actual** | Centroide + overlap asimétrico | CLIP cosine | parcial (2D) | post-corrección global | greedy O(N²) |
| ConceptGraphs | Centroide + overlap simétrico | CLIP + DINO | NMS 2D pre-lift | no explícito | greedy |
| OVIR-3D | Centroide + overlap + vol. ratio | CLIP categoría | indirecto | no explícito | **componentes conexas** |
| HOV-SG | IoU proyectado + centroide + vol. | CLIP | graph-cut | no explícito | greedy + post-hoc |
| Gaussian Grouping | rendering vote | DINO 2D tracking | **2D DINO track** | no aplica (NeRF) | tracking 2D |
| OpenMask3D | Mask3D 3D transformer | CLIP multi-view | aprendido | offline | offline batch |

---

## 8. Síntesis de recomendaciones priorizadas

### Tier 1: Correcciones urgentes (bajo coste, alto impacto)

**1. Fix del overlap asimétrico**
- Archivo: `ovo/utils/instance_utils.py::compute_pcd_overlap`
- Calcular en ambas direcciones, exponer `sym_overlap`, `max_overlap`, `containment`
- Impacto: corrige comportamiento incorrecto en casos B y E

**2. Hard veto por ratio de puntos/volumen**
- Archivo: `ovo/entities/fusion.py::SemanticGeometricFusion.same_instance`
- `if max(n_pts1, n_pts2) / min(n_pts1, n_pts2) > 5.0: return False`
- Impacto: previene fusiones absurdas entre objetos de escalas muy distintas

**3. Pre-3D 2D NMS (inspirado en ConceptGraphs)**
- Archivo: `ovo/entities/ovo.py::_fuse_masks_with_same_ins_id` o antes de `_track_objects`
- Si dos máscaras SAM tienen IoU 2D > 0.6 en el mismo frame, suprimir la más pequeña antes de crear Instance3D
- Impacto: reduce la población de instancias redundantes en origen, especialmente para over-segmentation

### Tier 2: Mejoras de lógica (coste medio, alto impacto)

**4. Co-ocurrencia como señal de penalización**
- Archivo: `ovo/entities/fusion.py::SemanticGeometricFusion.same_instance`
- Requiere pasar `kfs_ids` de ambas instancias al método `same_instance` (actualmente no se pasa)
- Penalizar si `len(set(kfs1) & set(kfs2)) > N_threshold`
- Impacto: distingue casos A (loop closure) de D (papeleras), la señal más discriminativa para ese par

**5. Componentes conexas en lugar de greedy** (OVIR-3D)
- Archivo: `ovo/entities/ovo.py::_fuse_overlapping_instances`
- Construir grafo de compatibilidad completo, luego fusionar por componentes con consistencia interna
- Impacto: elimina inconsistencias transitivas

**6. Threshold adaptativo post loop-closure**
- Archivo: `ovo/entities/ovo.py::update_map` o en `ovomapping.py` al llamar al update
- Relajar `th_centroid` temporalmente (×2) durante el pase de fusión post-corrección global
- Impacto: mejora recall en el escenario de loop closure sin afectar la fusión online normal

### Tier 3: Mejoras semánticas (coste medio-alto, impacto variable)

**7. DINO + CLIP como señales combinadas**
- Usar `w_clip * sim_clip + w_dino * sim_dino` en lugar de una única similitud
- Especialmente útil para over-segmentation (DINO es mejor para apariencia local)

**8. Threshold adaptativo por confianza del descriptor**
- `adaptive_th = base_th - (1 - min_confidence) * slack`
- Instancias poco observadas → threshold más permisivo

**9. Máscara binaria para extracción de features**
- Enmascarar el fondo antes de pasar la imagen al encoder
- Impacto mayor para PE y DINO, moderado para CLIP

### Tier 4: Mejoras avanzadas (coste alto, potencial alto)

**10. Varianza del descriptor como señal de confianza**
- Calcular varianza de los embeddings a lo largo de los keyframes de cada instancia
- Pesar el threshold de similitud inversamente a la varianza

**11. Comparación multi-vista**
- Comparar nubes completas de descriptores (en lugar del L1-median) y tomar el máximo de similitud
- Mejor para loop closure con ángulos de vista muy distintos

**12. Re-cómputo del descriptor post-fusión**
- Llamar a `update_clip/pe/sam3` con `force_update=True` tras cada fusión, con el pool combinado de keyframes
- Mejora la calidad del descriptor de la instancia superviviente

**13. Tratamiento específico del over-segmentation**
- Modo especial cuando `cos_sim > 0.95 AND containment_signal > 0.8 AND n_shared_kfs > 0`
- Fusionar partes coherentes aunque la co-ocurrencia normalmente sería penalizante

**14. Grafo de covisibilidad del SLAM** *(requiere soporte del backend)*
- Exponer la covisibilidad de keyframes desde el wrapper de ORB-SLAM3 (ya lo tiene internamente)
- Usar para: (a) pre-filtrar candidatos O(N²)→O(N·K), (b) identificar pares de loop closure a nivel de instancia, (c) reforzar la señal negativa (covisibilidad alta + instancias siempre separadas → objetos distintos), (d) usar covisibilidad cero entre keyframes de dos instancias como señal positiva de loop closure
- Ver §4.9 para el análisis completo

---

## 9. Papers relevantes

| Paper | Venue | arxiv | Relevancia |
|-------|-------|-------|------------|
| ConceptGraphs | ICRA 2024 | 2309.16650 | Sistema más similar a OVO; mismo criterio greedy con NMS 2D |
| OVIR-3D | ICCV 2023 | 2309.00809 | Solución de transitividad con componentes conexas; directamente aplicable |
| HOV-SG | RSS 2024 | 2309.14308 | Gate de ratio de tamaño; grafo de escena jerárquico |
| Gaussian Grouping | ECCV 2024 | 2312.00732 | Mejor tratamiento over-segmentation vía DINO 2D tracking |
| OpenMask3D | NeurIPS 2023 | 2309.10171 | Referencia offline; benchmark sin problema de fusión online |
| ConceptFusion | RSS 2023 | 2211.11684 | Features densas multi-modelo; alternativa a instancias online |
| OpenScene | CVPR 2023 | 2211.15654 | Destilación 2D→3D; agrupación post-hoc |

---

*Documento generado como resultado de análisis del sistema OVO y revisión de literatura. Fecha: 2026-03-18.*
