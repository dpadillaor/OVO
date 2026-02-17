# Visualización con Rerun - OVO

**Última actualización:** 2026-02-17
**Versión Rerun:** 0.23.1

## Actualizaciones Recientes

### 2026-02-17: Modo Fusion Mejorado
- ✅ Instancias individuales toggleables (before/after)
- ✅ Flechas 3D (`Arrows3D`) con dirección clara
- ✅ Bounding boxes con API correcta (`centers`/`half_sizes`)
- ✅ Sincronización de cámaras con `auto_layout=True`
- ✅ Panel de estadísticas con Markdown
- ✅ ~780 entidades en entity tree (organizado por carpetas)

## Descripción General

La visualización con Rerun es un sistema de streaming en tiempo real que permite visualizar el proceso de mapeo semántico SLAM durante la ejecución. Usa la librería [Rerun](https://rerun.io/) para mostrar nubes de puntos 3D, poses de cámara, y segmentaciones de instancias de forma interactiva.

## Archivos Involucrados

### 1. `ovo/entities/rerun_visualizer.py`
**Función principal:** `stream_rerun()`

Este archivo contiene la lógica de streaming a Rerun. Se ejecuta en un proceso independiente y recibe datos a través de una cola multiproceso.

**Ubicación:** `ovo/entities/rerun_visualizer.py:6`

**Responsabilidades:**
- Inicializar la sesión de Rerun
- Configurar el modelo de cámara Pinhole con los intrínsecos
- Recibir datos del proceso principal vía `mpqueue`
- Visualizar nubes de puntos con colores RGB
- Visualizar IDs de instancias de objetos
- Visualizar poses de cámara (transformaciones c2w)

### 2. `ovo/entities/ovomapping.py`
**Integración principal**

Este archivo orquesta el sistema completo de mapping y decide cuándo usar el visualizador de Rerun.

**Puntos clave:**
- **Línea 15:** Importa `stream_rerun`
- **Línea 51:** Lee el tipo de visualizador desde la configuración (`vis.type`)
- **Líneas 140-147:** Selecciona entre visualizador Rerun u Open3D según configuración
- **Línea 147:** Lanza el proceso de visualización en paralelo
- **Líneas 203-211:** Envía datos al visualizador cada vez que se segmenta un frame

## Configuración

### Archivo de Configuración
**Ubicación:** `data/working/configs/ovo.yaml`

```yaml
vis:
  stream: true           # Activar/desactivar streaming
  show_stream: true      # Mostrar la visualización (spawn viewer)
  type: "rerun"          # Tipo de visualizador: "rerun" o "open3d"
  rerun_mode: "fusion"   # Modo de visualización: "stream" o "fusion"
```

### Parámetros de Configuración

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `stream` | bool | Si es `true`, activa el streaming de visualización |
| `show_stream` | bool | Si es `true`, spawns el viewer de Rerun |
| `type` | str | Tipo de visualizador: `"rerun"` o `"open3d"` (por defecto: `"open3d"`) |
| `rerun_mode` | str | Modo de Rerun: `"stream"` (continuo) o `"fusion"` (eventos de fusión) (por defecto: `"stream"`) |

## Flujo de Datos

```
┌─────────────────────────────────────────────────────────────┐
│                   Proceso Principal (OVOSemMap)              │
│                                                              │
│  1. Track camera                                            │
│  2. Map points (cada map_every frames)                      │
│  3. Segment objects (cada segment_every frames)             │
│     │                                                        │
│     └─> Si frame_id % segment_every == 0:                   │
│         ┌────────────────────────────────────────┐          │
│         │ Obtener datos del SLAM backbone:        │          │
│         │  - pcd: puntos 3D (N, 3) float16        │          │
│         │  - pcd_obj_ids: IDs (N, L) int16        │          │
│         │  - colors: RGB (N, 3) uint8             │          │
│         │  - c2w: pose (4, 4) float16             │          │
│         └────────────────────────────────────────┘          │
│                         │                                    │
│                         ▼                                    │
│              mpqueue.put([pcd, obj_ids, colors, c2w])       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          │ Multiprocessing Queue
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Proceso Paralelo (RerunVisualizer)             │
│                                                              │
│  Loop infinito:                                             │
│    1. Verificar si hay datos en mpqueue                     │
│    2. Si hay datos:                                         │
│       - Unpack: points, obj_ids, colors, c2w                │
│       - Convertir tipos (float32, uint8)                    │
│       - rr.set_time_sequence("step", step)                  │
│       - rr.log("world/camera", Transform3D)                 │
│       - rr.log("world/points", Points3D + colors + IDs)     │
│       - Incrementar step                                    │
│    3. Si no hay datos: sleep(0.01)                          │
│    4. Si data == None: break (fin)                          │
└─────────────────────────────────────────────────────────────┘
```

## Datos Visualizados

### 1. Cámara Virtual
- **Entidad:** `"world/camera"`
- **Tipo:** `rr.Pinhole`
- **Datos:**
  - Resolución: `[width, height]`
  - Matriz intrínseca: `K` (3x3)
  - Se loggea como **static** (una sola vez)

### 2. Pose de Cámara
- **Entidad:** `"world/camera"`
- **Tipo:** `rr.Transform3D`
- **Datos:**
  - Traslación: `c2w[:3, 3]`
  - Rotación: `c2w[:3, :3]` (matriz 3x3)
  - Se actualiza cada frame

### 3. Nube de Puntos (dos capas toggleables)

#### Capa RGB
- **Entidad:** `"world/points/rgb"`
- **Tipo:** `rr.Points3D`
- **Datos:**
  - Posiciones: `points` (N, 3) float32
  - Colores RGB originales: `colors` (N, 3) uint8

#### Capa Instancias
- **Entidad:** `"world/points/instances"`
- **Tipo:** `rr.Points3D` + `rr.AnnotationContext`
- **Datos:**
  - Posiciones: `points` (N, 3) float32
  - Colores por instancia: coloreados con colormap tab20b+tab20c (mismos colores que Open3D)
  - IDs de clase/instancia: `class_ids` (N,) uint16
  - Cada instancia tiene label `obj_<id>` y color asignado vía `AnnotationContext`

**Toggle de visibilidad:** Usa el icono 👁 en el panel izquierdo de Rerun para mostrar/ocultar cada capa.

**Filtro de techo:** Se aplica automáticamente (elimina puntos con z > max - 0.2), igual que Open3D.

---

## Modo Fusion (Análisis de Loop Closure)

**Actualizado:** 2026-02-17

### Descripción

El modo `rerun_mode: "fusion"` visualiza eventos de fusión de instancias que ocurren durante loop closure. Muestra comparaciones lado a lado del mapa **antes** y **después** de cada fusión, con capas de análisis detalladas.

### Configuración

```yaml
vis:
  stream: true
  show_stream: true
  type: "rerun"
  rerun_mode: "fusion"  # Activa modo fusion
```

### Layout de la Ventana

```
┌──────────────────────────────────────────────────────────┐
│  [Before Fusion View]    │    [After Fusion View]        │
│                          │                               │
│  Instancias toggleables  │   Instancias toggleables      │
│  Capas diff compartidas  │   Capas diff compartidas      │
│                          │                               │
│  (Cámaras sincronizadas con auto_layout=True)            │
├──────────────────────────────────────────────────────────┤
│              [Fusion Stats Panel - Markdown]             │
│                                                          │
│  # Frame X Fusion Event                                  │
│  ## Summary: X→Y instances (Z fused)                     │
│  ## Fusion Details: mapeo específico                     │
└──────────────────────────────────────────────────────────┘
```

### Estructura del Entity Tree

```
📁 before/
│  └─ 📁 instances/
│     ├─ 📊 ins_0      [👁]  ← Toggle individual
│     ├─ 📊 ins_1      [👁]
│     ├─ 📊 ins_5      [👁]
│     └─ ... (~586 instancias)
│
📁 after/
│  └─ 📁 instances/
│     ├─ 📊 ins_0      [👁]
│     ├─ 📊 ins_5      [👁]
│     └─ ... (~194 instancias)
│
📁 diff/
│  ├─ 📊 changed_points          [👁]  ← Puntos amarillos
│  ├─ 📁 disappeared/
│  │  ├─ 📊 ins_12              [👁]  ← Puntos rojos
│  │  ├─ 📊 ins_18              [👁]
│  │  └─ ... (386 instancias fusionadas)
│  ├─ 📈 merge_arrows            [👁]  ← Flechas 3D magenta
│  └─ 📁 boxes/
│     ├─ 📦 ins_12              [👁]  ← Cajas rojas
│     ├─ 📦 ins_18              [👁]
│     └─ ... (386 boxes)
│
📁 stats/
   └─ 📄 fusion_info              ← Panel de texto (Markdown)
```

### Capas de Visualización

#### **1. Before/After Instances (Toggleables Individualmente)**

**Implementación:** `rerun_visualizer.py:311-346`

```python
for ins_id in unique_before:
    rr.log(f"before/instances/ins_{ins_id}",
           rr.Points3D(pts[mask], colors=color))

for ins_id in unique_after:
    rr.log(f"after/instances/ins_{ins_id}",
           rr.Points3D(pts[mask], colors=color))
```

**Características:**
- Cada instancia es una entidad separada
- Toggle individual en entity tree
- Colores consistentes (tab20b+tab20c colormap)
- Permite aislar instancias específicas para análisis

**Casos de uso:**
- Ver solo instancia 5: Activar `before/instances/ins_5` y `after/instances/ins_5`
- Comparar múltiples: Activar `ins_5`, `ins_12`, `ins_18`
- Ocultar todas: Desactivar carpeta `before/instances/`

#### **2. diff/changed_points (Amarillo)**

**Implementación:** `rerun_visualizer.py:368-377`

Puntos cuyo ID de instancia cambió debido a fusión.

```python
changed_mask = b_ids != a_ids
rr.log("diff/changed_points", rr.Points3D(pts[changed_mask], colors=[255, 255, 0]))
```

**Qué muestra:**
- Todos los puntos afectados por fusión
- Tamaño mayor (radii=0.012 vs 0.008) para destacar
- Visible en ambas vistas

**Uso:** Evaluar magnitud espacial de las fusiones

#### **3. diff/disappeared/ins_X (Rojo)**

**Implementación:** `rerun_visualizer.py:380-392`

Instancias que fueron fusionadas (dejaron de existir).

```python
for dis_id in disappeared_ids:
    mask = b_ids == dis_id
    rr.log(f"diff/disappeared/ins_{dis_id}",
           rr.Points3D(pts[mask], colors=[255, 0, 0]))
```

**Características:**
- Una capa por instancia fusionada
- Toggle individual (ej: solo ver ins_12)
- Color rojo = "desapareció"

**Uso:** Inspeccionar geometría de instancias fusionadas

#### **4. diff/merge_arrows (Magenta)**

**Implementación:** `rerun_visualizer.py:395-407`

Flechas 3D mostrando dirección de fusión.

```python
arrow_origins = []     # Centroides de instancias fusionadas
arrow_vectors = []     # Vectores hacia supervivientes

rr.log("diff/merge_arrows",
       rr.Arrows3D(origins=origins, vectors=vectors, colors=[255, 0, 255]))
```

**Qué muestra:**
- Origen: Centroide de instancia fusionada
- Punta: Centroide de instancia superviviente (después de fusión)
- Color magenta con punta visible

**Interpretación:**
```
Origen ━━━━━━━━━━→ Punta
(ins 12)           (ins 5, superviviente)
```

**Uso:** Validar que fusiones tienen sentido espacialmente

#### **5. diff/boxes/ins_X (Rojo)**

**Implementación:** `rerun_visualizer.py:410-431`

Bounding boxes de instancias fusionadas.

```python
center = (bbox_min + bbox_max) / 2
half_sizes = (bbox_max - bbox_min) / 2

rr.log(f"diff/boxes/ins_{dis_id}",
       rr.Boxes3D(centers=[center], half_sizes=[half_sizes],
                  labels=[f"Fused: {dis_id}"], colors=[[255, 0, 0]]))
```

**Qué muestra:**
- AABB (Axis-Aligned Bounding Box) de cada instancia fusionada
- Label con ID: `"Fused: 12"`
- Wireframe rojo

**Uso:**
- Contar fusiones visualmente (cada caja = 1 fusión)
- Ver extensión espacial de instancias
- Identificar rápidamente dónde ocurrieron fusiones

#### **6. stats/fusion_info (Markdown)**

**Implementación:** `rerun_visualizer.py:434-465`

Panel de texto con estadísticas detalladas.

```markdown
# Frame 143 Fusion Event

## Summary
- **Before:** 586 instances
- **After:** 194 instances
- **Fused:** 392 instances

## Fusion Details
- Instance **5** ← merged from [5, 12, 18]
- Instance **8** ← merged from [8, 22]

## Disappeared Instances
12, 18, 22, ...
```

**Información incluida:**
- Resumen numérico de fusiones
- Mapeo específico (quién se fusionó con quién)
- Lista de IDs desaparecidos

**Uso:** Referencia textual para validar visualización

### Sincronización de Cámaras

**Implementación:** `rerun_visualizer.py:181-207`

```python
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Before Fusion", origin="/", ...),
            rrb.Spatial3DView(name="After Fusion", origin="/", ...),
        ),
        ...
    ),
    auto_layout=True,  # ← Clave para sincronizar cámaras
)
```

**`auto_layout=True`** asegura que:
- ✅ Rotar en una vista → rota en ambas
- ✅ Zoom en una → zoom en ambas
- ✅ Pan en una → pan en ambas

**Versión Rerun:** 0.23.1 (instalada vía pip)

### Flujos de Trabajo Recomendados

#### **Flujo 1: Vista General**
1. Activar `before/instances/` y `after/instances/`
2. Activar `diff/merge_arrows`
3. Ver distribución general de fusiones

#### **Flujo 2: Inspección de Fusión Específica**
1. Desactivar todo
2. Activar `before/instances/ins_12` (instancia específica)
3. Activar `after/instances/ins_5` (superviviente)
4. Activar `diff/merge_arrows` (ver flecha 12→5)
5. Verificar en stats panel: "Instance 5 ← merged from [12, ...]"

#### **Flujo 3: Contar Fusiones Rápidamente**
1. Desactivar todo excepto `diff/boxes/`
2. Contar cajas rojas visualmente
3. Comparar con stats: "Fused: X instances"

#### **Flujo 4: Analizar Región Específica**
1. Zoom a región de interés
2. Activar solo instancias en esa región
3. Ver qué fusiones ocurrieron allí
4. Validar con `diff/changed_points`

### Consideraciones de Rendimiento

**Logging de instancias individuales:**
- ~780 llamadas a `rr.log()` por evento de fusión (586 before + 194 after)
- Overhead estimado: <1 segundo por evento
- Aceptable dado que fusiones son poco frecuentes

**Ventajas:**
- Control granular sobre visibilidad
- Análisis detallado de fusiones individuales
- Mejor que 1 sola entidad con todos los puntos

**Entity tree grande:**
- ~780 entradas en total
- Organizadas por carpetas (before/after/diff)
- Colapsable para reducir ruido

### API de Rerun Utilizada

**Archetypes:**
- `rr.Points3D` - Nubes de puntos
- `rr.Arrows3D` - Flechas 3D con dirección
- `rr.Boxes3D` - Bounding boxes (AABB)
- `rr.TextDocument` - Panel de stats (Markdown)

**Blueprint:**
- `rrb.Spatial3DView` - Vistas 3D
- `rrb.TextDocumentView` - Panel de texto
- `rrb.Horizontal` / `rrb.Vertical` - Layouts
- `auto_layout=True` - Sincronización de cámaras

---

## Detalles de Implementación

### Tipos de Datos
El sistema usa tipos comprimidos para la comunicación entre procesos:

**En el proceso principal (antes de enviar):**
```python
pcd.cpu().numpy().astype(np.float16)           # Puntos 3D
pcd_obj_ids.cpu().numpy().astype(np.int16)     # IDs de objetos
colors (uint8)                                  # Colores RGB
c2w.cpu().numpy().astype(np.float16)           # Pose de cámara
```

**En el visualizador (después de recibir):**
```python
points.astype(np.float32)    # Convertir para Rerun
colors.astype(np.uint8)      # Asegurar tipo correcto
c2w.astype(np.float32)       # Convertir para Rerun

# Filtro de techo (mismo que Open3D)
ceiling_z = points[:, -1].max() - 0.2
mask = points[:, -1] < ceiling_z

# Colores por instancia (mismos que Open3D)
instance_colors = _get_instance_colors(instance_ids, cmap)  # tab20b+tab20c
```

### Multiprocesamiento
- **Gestor:** `mp.Manager()`
- **Cola:** `mp.Queue()` para transferir datos de visualización
- **Proceso:** `mp.Process` con nombre `"RerunVisualizer"`
- **Comunicación adicional:** `mp.Pipe()` para queries semánticas (funcionalidad de query interactiva)

### Sistema de Queries (Opcional)
El visualizador soporta queries semánticas interactivas:
- `query_flag`: valor compartido para coordinar queries (0=idle, 1=requested, 2=completed)
- `vis_pipe`: pipe para enviar/recibir queries y resultados
- Permite al usuario hacer búsquedas semánticas durante la visualización

### Blueprint Automático
Al iniciar, se envía un Blueprint con dos vistas 3D lado a lado:
- **Vista "RGB":** Muestra `world/camera` + `world/points/rgb`
- **Vista "Instances":** Muestra `world/camera` + `world/points/instances`

Esto permite ver ambas representaciones simultáneamente. También se puede usar
una sola vista y togglear las capas con el icono 👁 en el panel lateral.

### AnnotationContext
Se registra un `rr.AnnotationContext` estático en `world/points/instances` con:
- Un `rr.AnnotationInfo` por cada instancia única detectada
- Label: `obj_<id>` (ej: `obj_0`, `obj_5`, `obj_12`)
- Color: tomado del colormap tab20b+tab20c (mismos colores que Open3D)

---

## Impacto en Rendimiento

### Arquitectura de la Visualización

#### 1. Proceso Separado
**Ubicación:** `ovomapping.py:147-148`

```python
p = mp.Process(target=stream_rerun, ...)
p.start()
```

**✅ Ventaja:**
- Proceso completamente independiente con su propio espacio de memoria
- No comparte CPU/GPU con OVO
- Paralelismo verdadero (no threading de Python)

#### 2. Comunicación via Cola
**Ubicación:** `ovomapping.py:137, 211`

```python
mpqueue = mp.Queue()  # Sin tamaño máximo especificado
mpqueue.put([pcd.cpu().numpy().astype(np.float16), ...])
```

**⚠️ Problema Potencial:**
- `mp.Queue()` sin tamaño máximo puede crecer indefinidamente
- `put()` sin timeout puede **bloquearse** si la cola alcanza límites del sistema
- Si el visualizador es más lento que OVO, la cola crece → posible bloqueo

#### 3. Consumo de Datos
**Ubicación:** `rerun_visualizer.py:32-81`

```python
if not mpqueue.empty():
    data = mpqueue.get()
    # Procesar y visualizar
else:
    time.sleep(0.01)  # 10ms de espera
```

**✅ Bueno:** No bloquea si no hay datos
**⚠️ Problema:** Si el visualizador es lento, no consume rápido → cola crece

### Impactos Específicos en Rendimiento

#### ✅ NO Afecta (Teóricamente)

1. **CPU/GPU de procesamiento:** Proceso separado, no compite por recursos de cómputo
2. **Lógica de OVO:** El SLAM y segmentación continúan sin interferencia

#### ⚠️ SÍ Afecta (En la Práctica)

##### 1. Transferencia GPU → CPU
**Ubicación:** `ovomapping.py:204-209`

```python
pcd.cpu().numpy().astype(np.float16)          # Transferencia GPU→CPU
pcd_obj_ids.cpu().numpy().astype(np.int16)    # Transferencia GPU→CPU
c2w.cpu().numpy().astype(np.float16)          # Transferencia GPU→CPU
```

- **Overhead:** Cada transferencia bloquea el stream CUDA
- **Magnitud:** Crece con el tamaño del mapa (1M puntos ≈ 12MB por frame)
- **Frecuencia:** Ocurre cada `segment_every` frames (default: 10)

##### 2. Copia de TODO el Mapa
```python
pcd.cpu().numpy()  # Copia TODO el mapa acumulado cada vez
```

**Importante:**
- No copia solo puntos nuevos, sino **TODO el mapa cada update**
- 500k puntos → ~6MB de copia por update
- 1M+ puntos → ~20-30MB por update

##### 3. Posible Bloqueo por Cola Llena
Si el visualizador procesa lentamente (nubes grandes):
```python
mpqueue.put(...)  # Puede bloquearse aquí si la cola está llena
```

Esto **DETIENE todo OVO** hasta que haya espacio en la cola.

### Estimaciones de Overhead

#### Escenario Típico
- Mapa de 200k puntos
- `segment_every: 10`
- Transferencia GPU→CPU: ~2-5ms
- Copia de memoria: ~1-2ms
- **Overhead por frame de segmentación: ~3-7ms**
- **Overhead promedio: ~0.3-0.7ms por frame** (amortizado)

#### Escenario Grande
- Mapa de 1M puntos
- `segment_every: 10`
- Transferencia GPU→CPU: ~10-20ms
- Copia de memoria: ~5-10ms
- **Overhead por frame de segmentación: ~15-30ms**
- **Overhead promedio: ~1.5-3ms por frame** (amortizado)

### ¿Por Qué No Es Peor?

1. **Compresión de datos:** Usa `float16` en lugar de `float32` (50% menos datos)
2. **Frecuencia baja:** Solo cada `segment_every` frames (no todos los frames)
3. **Proceso separado:** El rendering no bloquea (a menos que la cola se llene)
4. **Operaciones asíncronas:** La cola permite buffering temporal

### Cuándo SÍ Es Problemático

| Escenario | Impacto | Solución Potencial |
|-----------|---------|-------------------|
| Mapas muy grandes (>1M puntos) | Copias costosas (>20ms) | Implementar decimación de puntos |
| RAM limitada | Cola llena → bloqueos | Limitar tamaño de cola: `mp.Queue(maxsize=5)` |
| Visualizador lento | Cola crece → bloqueo | Añadir timeout en `put()` |
| `segment_every=1` | Overhead constante | Aumentar `segment_every` o deshabilitar streaming |

### Resumen de Impacto

| Aspecto | Impacto | Magnitud |
|---------|---------|----------|
| CPU/GPU de procesamiento | ✅ Nulo | Proceso separado |
| Transferencias GPU→CPU | ⚠️ Bajo-Medio | ~3-30ms cada `segment_every` frames |
| Copia de memoria | ⚠️ Bajo | Todo el mapa, pero poco frecuente |
| Bloqueo de cola | ⚠️ Potencial | Si visualizador es muy lento |
| **Impacto Total Estimado** | ⚠️ **~2-5% overhead** | En escenarios típicos |

**Conclusión:**
- El impacto es **leve** en la mayoría de casos
- El overhead es **proporcional al tamaño del mapa**
- Para experimentos sin visualización, desactivar con `stream: false` elimina completamente el overhead
- El diseño con proceso separado evita que el rendering afecte directamente al procesamiento

---

## Optimizaciones Posibles

### 1. Limitar Tamaño de Cola
**Problema actual:** `mp.Queue()` sin límite puede crecer indefinidamente

**Solución:**
```python
mpqueue = mp.Queue(maxsize=5)  # Limitar a 5 elementos
```

**Ventajas:**
- Evita consumo excesivo de RAM
- Fuerza al sistema a esperar si el visualizador está retrasado

**Desventajas:**
- `put()` puede bloquearse si la cola está llena (necesita timeout)

**Implementación:**
- Archivo: `ovomapping.py:137`
- Añadir parámetro `maxsize` configurable

---

### 2. Timeout en put() para Evitar Bloqueos
**Problema actual:** `mpqueue.put()` puede bloquearse indefinidamente

**Solución:**
```python
try:
    mpqueue.put(data, timeout=0.01)  # 10ms timeout
except queue.Full:
    # Descartar frame de visualización, continuar procesamiento
    pass
```

**Ventajas:**
- Nunca bloquea el pipeline principal
- Visualización se "salta" frames si está retrasada
- El procesamiento SLAM/segmentación nunca se detiene

**Implementación:**
- Archivo: `ovomapping.py:211`
- Añadir try-except con timeout configurable

---

### 3. Decimación de Puntos para Visualización
**Problema actual:** Se envían TODOS los puntos del mapa (puede ser 1M+)

**Solución:**
```python
# Opciones de decimación:
# a) Random sampling
indices = np.random.choice(len(pcd), size=max_vis_points, replace=False)
pcd_vis = pcd[indices]

# b) Voxel downsampling (más uniforme espacialmente)
pcd_vis, indices = voxel_downsample(pcd, voxel_size=0.05)

# c) Decimación fija (cada N puntos)
pcd_vis = pcd[::decimation_factor]
```

**Ventajas:**
- Reduce drásticamente transferencias GPU→CPU
- Reduce tamaño de datos en cola
- Visualización más fluida en Rerun
- Mapas de 1M puntos → 100k puntos (~10x reducción)

**Desventajas:**
- Se pierde detalle visual (aceptable para visualización en tiempo real)
- IDs de objetos deben mantenerse consistentes

**Implementación:**
- Archivo: `ovomapping.py:204-211`
- Añadir parámetro de configuración `vis.max_points` o `vis.decimation_factor`

**Ejemplo configuración:**
```yaml
vis:
  stream: true
  show_stream: true
  type: "rerun"
  max_points: 100000  # Máximo de puntos a visualizar
```

---

### 4. Enviar Solo Puntos Nuevos (Incremental Updates)
**Problema actual:** Se envía TODO el mapa cada vez, incluso puntos ya visualizados

**Solución:**
```python
# En ovomapping.py:
if stream:
    # Obtener solo puntos añadidos desde último envío
    new_pcd = pcd[last_sent_idx:]
    new_obj_ids = pcd_obj_ids[last_sent_idx:]
    new_colors = colors[last_sent_idx:]

    mpqueue.put(['incremental', new_pcd, new_obj_ids, new_colors, c2w])
    last_sent_idx = len(pcd)

# En rerun_visualizer.py:
# Acumular puntos en el visualizador
if data[0] == 'incremental':
    all_points = np.vstack([all_points, data[1]])
    all_colors = np.vstack([all_colors, data[2]])
    # ...
```

**Ventajas:**
- Transferencias GPU→CPU mucho menores (solo nuevos puntos)
- Cola más pequeña
- Mejor para mapas grandes

**Desventajas:**
- No funciona bien con updates de loop closure (necesita reenviar todo)
- Mayor complejidad en el código
- Necesita manejo especial para actualizaciones globales

**Implementación:**
- Archivos: `ovomapping.py:204-211`, `rerun_visualizer.py:38-77`
- Añadir modo incremental + modo "full update" para loop closures

---

### 5. Memoria Compartida en Lugar de Cola
**Problema actual:** `mp.Queue()` copia todos los datos al enviarlos

**Solución:**
```python
# Usar shared memory de Python 3.8+
from multiprocessing import shared_memory

# Crear shared memory para el mapa
shm = shared_memory.SharedMemory(create=True, size=pcd.nbytes)
shared_pcd = np.ndarray(pcd.shape, dtype=pcd.dtype, buffer=shm.buf)

# Escribir datos
np.copyto(shared_pcd, pcd.cpu().numpy())

# Solo enviar metadatos por la cola
mpqueue.put({'shm_name': shm.name, 'shape': pcd.shape, 'dtype': pcd.dtype})
```

**Ventajas:**
- Elimina copia de datos grande en la cola
- Mucho más eficiente para mapas grandes
- Reduce uso de RAM (una sola copia en memoria compartida)

**Desventajas:**
- Mayor complejidad de implementación
- Necesita sincronización cuidadosa (locks)
- Problemas si el visualizador lee mientras se escribe

**Implementación:**
- Archivos: `ovomapping.py`, `rerun_visualizer.py`
- Requiere refactorización significativa
- Añadir locks para sincronización

---

### 6. Compresión Adicional de Datos
**Problema actual:** Solo usa `float16`, pero aún ocupa espacio significativo

**Solución:**
```python
import zlib
import pickle

# Comprimir antes de enviar
data_bytes = pickle.dumps([pcd, obj_ids, colors, c2w])
compressed = zlib.compress(data_bytes, level=1)  # level 1 = rápido
mpqueue.put(compressed)

# Descomprimir en visualizador
decompressed = zlib.decompress(compressed)
pcd, obj_ids, colors, c2w = pickle.loads(decompressed)
```

**Ventajas:**
- Reduce tamaño en cola (ratio ~2-4x dependiendo de datos)
- Menor uso de RAM para la cola

**Desventajas:**
- CPU adicional para comprimir/descomprimir
- Puede ser más lento que el beneficio obtenido
- Trade-off CPU vs memoria

**Evaluación:**
- Probablemente **NO vale la pena** para este caso
- El cuello de botella es GPU→CPU, no el tamaño de la cola

---

### 7. Frecuencia Adaptativa de Visualización
**Problema actual:** Frecuencia fija cada `segment_every` frames

**Solución:**
```python
# Reducir frecuencia si la cola está creciendo
if mpqueue.qsize() > 3:
    vis_skip_counter += 1
    if vis_skip_counter % 2 != 0:  # Saltar 50% de updates
        continue

# O basado en tiempo
if time.time() - last_vis_update < min_vis_interval:
    continue
```

**Ventajas:**
- Se adapta automáticamente a la capacidad del visualizador
- Evita saturar la cola
- Mantiene procesamiento principal sin ralentizar

**Implementación:**
- Archivo: `ovomapping.py:203-211`
- Añadir lógica de throttling adaptativo

---

### 8. Transferencias GPU→CPU Asíncronas
**Problema actual:** `.cpu()` bloquea hasta que la transferencia completa

**Solución:**
```python
# Usar CUDA streams para transferencias asíncronas
with torch.cuda.stream(transfer_stream):
    pcd_cpu = pcd.cpu()  # No bloquea el stream principal
    # ... otras transferencias

# Continuar procesamiento mientras se transfiere
# ...

# Sincronizar solo cuando sea necesario
transfer_stream.synchronize()
mpqueue.put([pcd_cpu.numpy(), ...])
```

**Ventajas:**
- Overlap entre procesamiento y transferencia
- Reduce impacto en el pipeline principal

**Desventajas:**
- Mayor complejidad
- Necesita gestión cuidadosa de streams CUDA
- Beneficio limitado (la transferencia aún ocurre)

**Implementación:**
- Archivo: `ovomapping.py:204-211`
- Crear CUDA stream dedicado para visualización

---

### Resumen de Optimizaciones

| Optimización | Dificultad | Impacto | Prioridad | Archivos Afectados |
|--------------|------------|---------|-----------|-------------------|
| **1. Limitar tamaño cola** | Baja | Bajo-Medio | Alta | `ovomapping.py:137` |
| **2. Timeout en put()** | Baja | Alto | **CRÍTICA** | `ovomapping.py:211` |
| **3. Decimación puntos** | Media | Alto | **CRÍTICA** | `ovomapping.py:204-211` |
| **4. Updates incrementales** | Alta | Medio-Alto | Media | `ovomapping.py`, `rerun_visualizer.py` |
| **5. Memoria compartida** | Muy Alta | Alto | Baja | Ambos archivos (refactorización) |
| **6. Compresión** | Baja | Bajo | Muy Baja | Ambos archivos |
| **7. Frecuencia adaptativa** | Media | Medio | Media | `ovomapping.py:203-211` |
| **8. Transferencias async** | Alta | Bajo-Medio | Baja | `ovomapping.py:204-211` |

### Recomendaciones de Implementación

**Para mejora inmediata (quick wins):**
1. ✅ **Timeout en put()** (Optimización #2) - Evita bloqueos críticos
2. ✅ **Decimación de puntos** (Optimización #3) - Mayor impacto con poco esfuerzo
3. ✅ **Limitar tamaño cola** (Optimización #1) - Previene problemas de memoria

**Para mejora a medio plazo:**
4. **Frecuencia adaptativa** (Optimización #7) - Mejora robustez
5. **Updates incrementales** (Optimización #4) - Para mapas muy grandes

**No recomendadas (trade-off desfavorable):**
- Compresión (Optimización #6) - CPU overhead > beneficio
- Memoria compartida (Optimización #5) - Complejidad >> beneficio
- Transferencias async (Optimización #8) - Difícil implementación, beneficio limitado

---

## Secciones para Futuras Modificaciones

### 🔧 Modificaciones Planeadas

#### 1. Mejoras en la Visualización de Puntos
**Estado:** Pendiente
**Descripción:** TBD
**Archivos afectados:**
- `ovo/entities/rerun_visualizer.py`

**Notas:**


---

#### 2. Optimizaciones de Performance
**Estado:** Pendiente
**Descripción:** TBD
**Archivos afectados:**
- `ovo/entities/rerun_visualizer.py`
- `ovo/entities/ovomapping.py`

**Notas:**


---

#### 3. Nuevas Entidades a Visualizar
**Estado:** Pendiente
**Descripción:** TBD
**Archivos afectados:**
- `ovo/entities/rerun_visualizer.py`
- `ovo/entities/ovomapping.py`

**Posibles adiciones:**
- [ ] Visualizar máscaras de segmentación 2D
- [ ] Visualizar keyframes del SLAM
- [ ] Visualizar embeddings CLIP/PE
- [ ] Visualizar trayectoria de la cámara
- [ ] Visualizar bounding boxes 3D de objetos

**Notas:**


---

#### 4. Configuración Avanzada
**Estado:** Pendiente
**Descripción:** TBD
**Archivos afectados:**
- `data/working/configs/ovo.yaml`
- `ovo/entities/ovomapping.py`

**Parámetros potenciales:**
- [ ] Decimación de puntos para visualización
- [ ] Frecuencia de actualización independiente
- [ ] Filtros de visualización por ID de instancia
- [ ] Configuración de colores y estilos

**Notas:**


---

#### 5. Sistema de Queries Mejorado
**Estado:** Pendiente
**Descripción:** TBD
**Archivos afectados:**
- `ovo/entities/rerun_visualizer.py`
- `ovo/entities/ovomapping.py`

**Notas:**


---

## Referencias

### Documentación de Rerun
- [Rerun Docs](https://www.rerun.io/docs)
- [Python API Reference](https://ref.rerun.io/docs/python/)
- [Points3D](https://ref.rerun.io/docs/python/stable/common/archetypes/#rerun.archetypes.Points3D)
- [Arrows3D](https://ref.rerun.io/docs/python/stable/common/archetypes/#rerun.archetypes.Arrows3D)
- [Boxes3D](https://ref.rerun.io/docs/python/stable/common/archetypes/#rerun.archetypes.Boxes3D)
- [TextDocument](https://ref.rerun.io/docs/python/stable/common/archetypes/#rerun.archetypes.TextDocument)
- [Transform3D](https://ref.rerun.io/docs/python/stable/common/archetypes/#rerun.archetypes.Transform3D)
- [Pinhole](https://ref.rerun.io/docs/python/stable/common/archetypes/#rerun.archetypes.Pinhole)
- [Blueprint](https://ref.rerun.io/docs/python/stable/common/blueprint_api/)

### Archivos del Proyecto
- Visualizador: `ovo/entities/rerun_visualizer.py`
  - Función `stream_rerun()` - Modo stream (línea ~31)
  - Función `stream_rerun_fusion()` - Modo fusion (línea ~163)
- Integración: `ovo/entities/ovomapping.py:140-147, 203-211`
- Configuración: `data/working/configs/ovo.yaml`
- Documentación: `docs/rerun_visualization.md` (este archivo)

### Versiones
- **Rerun SDK:** 0.23.1 (pip)
- **Python:** 3.x
- **Última actualización documentación:** 2026-02-17

---

**Última actualización:** 2026-02-17
**Autor:** Claude Code
**Rama:** `feature/rerun_vis`
