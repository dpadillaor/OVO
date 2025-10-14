# Estrategia de Implementación: Baseline SLAM para Experimentación Semántica

## Objetivo

Crear un entorno controlado y reproducible para experimentar con diferentes métodos de fusión de instancias 3D (CLIP vs DINO) sin la variabilidad del SLAM, eliminando loop closures y manteniendo poses y puntos 3D deterministas.

## Problemática Actual

- **Variabilidad del SLAM**: Diferentes ejecuciones pueden producir mapas ligeramente diferentes
- **Loop Closures**: Alteran retroactivamente el mapa y complican la comparación
- **Tiempo de experimentación**: Cada prueba requiere recalcular SLAM completo
- **Falta de control**: Difícil aislar el impacto de métodos de fusión semántica

## Solución Propuesta

### Fase 1: Extracción de Baseline SLAM
Ejecutar SLAM una vez sin componente semántico y guardar el estado final del mapa.

### Fase 2: Modo Experimental
Cargar datos pre-calculados y experimentar solo con la parte semántica.

---

## Implementación por Fases

### **FASE 1: Extracción de Baseline**

#### Archivos a Crear/Modificar:

**`scripts/extract_slam_baseline.py`** (NUEVO)
- **Propósito**: Script principal para generar baseline
- **Funcionalidad**:
  - Cargar dataset y configuración SLAM
  - Ejecutar SLAM sin módulo semántico (`disable_semantic: true`)
  - Extraer estado final del mapa (puntos 3D, poses, keyframes)
  - Guardar en formato compacto con metadatos
- **Output**: `data/baselines/{dataset}_{scene}_baseline.pth`

**`data/working/configs/slam/extraction.yaml`** (NUEVO)
- **Propósito**: Configuración específica para extracción
- **Contenido**:
  - Desactivar semántica completamente
  - Desactivar loop closure (si es posible)
  - Configuración optimizada para determinismo

#### Estructura de Datos del Baseline:
```python
baseline_data = {
    "metadata": {
        "dataset": "Replica",
        "scene": "office0", 
        "slam_backend": "orbslam2",
        "timestamp": datetime.now(),
        "config_hash": hash(config)
    },
    "slam_state": {
        "final_poses": dict(),        # frame_id -> c2w matrix
        "final_map": {
            "points_3d": torch.Tensor,   # Npoints x 3
            "points_ids": torch.Tensor,  # Npoints identificadores únicos
            "colors": torch.Tensor       # Npoints x 3 (opcional)
        },
        "keyframes": list(),          # Lista de frame_ids de keyframes
        "intrinsics": torch.Tensor    # Parámetros de cámara
    }
}
```

### **FASE 2: Backend de Replay**

#### Archivos a Crear/Modificar:

**`ovo/slam/baseline_slam.py`** (NUEVO)
- **Propósito**: Backend SLAM que carga datos pre-calculados
- **Herencia**: Heredar de `VanillaMapper` para mantener interfaz
- **Funcionalidad**:
  - Cargar baseline en `__init__`
  - Implementar métodos requeridos (`track_camera`, `map`, `get_c2w`, etc.)
  - Simular comportamiento temporal del SLAM
  - Mantener compatibilidad con pipeline OVO existente

**`ovo/entities/ovomapping.py`** (MODIFICAR)
- **Función**: `get_slam_backbone()`
- **Cambio**: Añadir caso para `slam_module: "baseline"`
- **Instanciación**: `return BaselineSLAM(config, baseline_path)`

### **FASE 3: Integración con Pipeline**

#### Archivos a Modificar:

**`data/working/configs/slam/baseline.yaml`** (NUEVO)
- **Propósito**: Configuración para modo baseline
- **Contenido**:
```yaml
slam:
  slam_module: "baseline"
  baseline_path: "data/baselines/{dataset}_{scene}_baseline.pth"
  map_every: 10  # Para simular comportamiento temporal
```

**`run_eval.py`** (MODIFICAR)
- **Nueva opción**: `--baseline-mode` o flag en configuración
- **Lógica**: Cargar configuración baseline cuando esté activo
- **Compatibilidad**: Mantener flujo normal para el resto del pipeline

### **FASE 4: Validación y Utilidades**

#### Archivos a Crear:

**`scripts/validate_baseline.py`** (NUEVO)
- **Propósito**: Verificar reproducibilidad del baseline
- **Tests**:
  - Comparar métricas finales baseline vs SLAM normal
  - Verificar determinismo entre ejecuciones
  - Validar integridad de datos guardados

**`scripts/compare_fusion_methods.py`** (NUEVO)
- **Propósito**: Script para comparación A/B de métodos de fusión
- **Funcionalidad**:
  - Ejecutar misma escena con diferentes métodos
  - Generar métricas comparativas
  - Visualizar diferencias en fusión

---

## Flujo de Trabajo

### 1. Generación de Baseline (Una vez por escena)
```bash
# Extraer baseline para escena específica
python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --slam orbslam2

# Validar baseline generado
python scripts/validate_baseline.py --baseline data/baselines/Replica_office0_baseline.pth
```

### 2. Experimentación (Repetible)
```bash
# Experimento con CLIP
python run_eval.py --dataset_name Replica --scene office0 --baseline-mode --fusion-method clip

# Experimento con DINO  
python run_eval.py --dataset_name Replica --scene office0 --baseline-mode --fusion-method dino

# Comparación automática
python scripts/compare_fusion_methods.py --scene office0 --methods clip,dino
```

---

## Ventajas del Enfoque

- ✅ **Reproducibilidad**: Mismos puntos 3D y poses en cada experimento
- ✅ **Velocidad**: ~10x más rápido para experimentación semántica
- ✅ **Control**: Aislamiento completo de variabilidad SLAM
- ✅ **Comparabilidad**: Métricas objetivas entre métodos de fusión
- ✅ **Escalabilidad**: Fácil probar múltiples métodos y parámetros

## Consideraciones

- **Limitación**: Solo válido para escenas sin loop closure críticos
- **Almacenamiento**: ~10-50MB por baseline (vs ~GB para secuencias completas)
- **Compatibilidad**: Mantiene interfaz existente, no rompe funcionalidad actual
- **Extensibilidad**: Facilita añadir nuevos métodos de fusión en el futuro

---

## Próximos Pasos

1. **Implementar Fase 1**: Script de extracción y estructura de datos
2. **Implementar Fase 2**: Backend baseline SLAM
3. **Integrar Fase 3**: Modificaciones en pipeline existente
4. **Validar Fase 4**: Scripts de validación y comparación
5. **Documentar**: Guías de uso y ejemplos prácticos