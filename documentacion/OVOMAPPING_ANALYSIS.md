# Análisis Detallado: ovomapping.py

## Propósito General
`ovomapping.py` es el **orquestador principal** del sistema OVO. Coordina la ejecución temporal de SLAM y procesamiento semántico, gestionando el flujo de datos entre componentes y controlando el timing de operaciones.

---

## Estructura del Archivo

### **Imports y Dependencias**
```python
from .logger import Logger
from .ovo import OVO
from .datasets import get_dataset
from ..slam.vanilla_mapper import VanillaMapper
from ..utils import io_utils
```

### **Factory Function: `get_slam_backbone()`**
**Ubicación**: Líneas ~15-25  
**Propósito**: Seleccionar e instanciar el backend SLAM según configuración

**Backends Soportados**:
- `gaussian_slam` → `WrapperGaussianSLAM`
- `orbslam2` → `WrapperORBSLAM2` 
- `vanilla` (default) → `VanillaMapper`

**Interfaz Común**: Todos los backends deben implementar:
- `track_camera(frame_data)`
- `map(frame_data, c2w)`
- `get_c2w(frame_id)`
- `get_map()`
- `get_kfs()`

### **Clase Principal: `OVOSemMap`**

#### **Constructor (`__init__`)**
**Responsabilidades**:
1. **Setup de paths**: `_setup_output_path()`
2. **Configuración**: Guardar config, cargar parámetros timing
3. **Componentes principales**:
   - `Logger` - para métricas y logs
   - `Dataset` - carga de imágenes/depth/poses
   - `OVO` - módulo semántico (CLIP+SAM)
   - `SLAM Backend` - mapping 3D

**Parámetros de Timing**:
- `map_every` - frecuencia de mapping (ej: cada 10 frames)
- `segment_every` - frecuencia de segmentación (ej: cada 10 frames)
- `track_every` - frecuencia de tracking (ej: cada frame)

#### **Método Principal: `run()`**
**Bucle temporal por frames**:

```python
for frame_id in range(first_frame, len(dataset)):
    # 1. TRACKING - Estimar pose de cámara
    if frame_id % track_every == 0:
        slam_backbone.track_camera(frame_data)
        estimated_c2w = slam_backbone.get_c2w(frame_id)
    
    # 2. MAPPING - Actualizar mapa 3D
    if frame_id % map_every == 0:
        slam_backbone.map(frame_data, estimated_c2w)
        if slam_backbone.map_updated:
            # Loop closure semántico
            map_data = slam_backbone.get_map()
            updated_points_ins_ids = ovo.update_map(map_data, kfs)
    
    # 3. SEMANTIC - Detectar y trackear objetos
    if frame_id % segment_every == 0:
        map_data = slam_backbone.get_map()
        updated_points_ins_ids = ovo.detect_and_track_objects(
            scene_data, map_data, estimated_c2w)
        
        ovo.compute_semantic_info()
```

#### **Gestión de Estado**
- **`save_representation()`**: Guarda mapa + parámetros OVO en checkpoint
- **`restore_representation()`**: Carga estado previo para continuar ejecución

#### **Visualización en Tiempo Real**
- **Proceso paralelo**: Open3D visualizer si `stream: true`
- **Query interactivo**: Permite consultas semánticas en tiempo real

---

## Flujo de Datos Principal

### **1. Inicialización**
```
Config → Dataset → Camera Intrinsics → OVO → SLAM Backend
```

### **2. Bucle Por Frame**
```
Frame Data → SLAM Tracking → Pose Estimation
     ↓
Map Update → Semantic Processing → Instance Tracking
     ↓
Update 3D Map ← Loop Closure Handling
```

### **3. Sincronización Temporal**
- **Track**: Cada frame (pose estimation)
- **Map**: Cada `map_every` frames (3D reconstruction)  
- **Segment**: Cada `segment_every` frames (semantic processing)

---

## Partes Relevantes para Proyecto Baseline

### **🎯 Modificación Principal: Factory Function**

**Ubicación**: Función `get_slam_backbone()` (líneas ~15-25)

**Cambio Requerido**: Añadir soporte para `slam_module: "baseline"`

**Implementación Planificada**:
```python
def get_slam_backbone(config, dataset, cam_intrinsics):
    backbone = config["slam"].get("slam_module","vanilla")
    
    # NUEVA LÍNEA A AÑADIR
    if backbone == "baseline":
        from ..slam.baseline_slam import BaselineSLAM
        baseline_path = config["slam"]["baseline_path"]
        return BaselineSLAM(config, baseline_path)
    # ... resto del código existente
```

### **🎯 Compatibilidad de Interfaz**

**Requerimiento**: `BaselineSLAM` debe implementar la misma interfaz que otros backends

**Métodos Críticos a Implementar**:
- `track_camera()` - Solo incrementar frame counter
- `map()` - Simular map updates sin cálculo
- `get_c2w()` - Devolver pose pre-calculada
- `get_map()` - Devolver puntos 3D pre-calculados
- `get_kfs()` - Devolver lista de keyframes

### **🎯 Timing y Sincronización**

**Consideración**: El baseline debe respetar el timing original
- Mantener `map_every` y `segment_every` para simular comportamiento temporal
- Sincronizar updates con frames específicos

### **🎯 Configuración**

**Archivo Nuevo**: `data/working/configs/slam/baseline.yaml`
```yaml
slam:
  slam_module: "baseline"
  baseline_path: "data/baselines/{dataset}_{scene}_baseline.pth"
  map_every: 10  # Mantener timing original
```

---

## Estrategia de Implementación

### **Paso 1**: Crear `BaselineSLAM` class
- Hereda de `VanillaMapper` para reutilizar código común
- Carga baseline data en constructor
- Implementa métodos requeridos sin cálculos

### **Paso 2**: Modificar Factory Function
- Una línea de código en `get_slam_backbone()`
- Import condicional para evitar dependencias

### **Paso 3**: Configuración
- Nuevo archivo YAML para modo baseline
- Modificación mínima en `run_eval.py` para usar config baseline

### **Ventajas del Enfoque**
- ✅ **Mínima invasividad**: Solo una línea en factory function
- ✅ **Compatibilidad total**: Misma interfaz que backends existentes  
- ✅ **Flexibilidad**: Fácil switch entre baseline y SLAM real
- ✅ **Reutilización**: Aprovecha infraestructura existente

---

## Puntos de Atención

### **Gestión de Memory**
- El baseline debe gestionar memoria similarmente al SLAM original
- Considerar limpieza de datos no utilizados

### **Error Handling**
- Validar existencia de baseline file
- Manejo de frames faltantes en baseline data

### **Debugging**
- Mantener logs compatibles para comparación
- Preservar métricas de timing para análisis