# Cambios en la Estructura del Baseline SLAM

## Resumen
Se ha modificado `scripts/extract_slam_baseline.py` para generar archivos `.pth` con una estructura simplificada que captura explícitamente los eventos de loop closure con sus estados before/after.

---

## Estructura ANTERIOR

```python
baseline = {
    'metadata': {
        'scene_name': str,
        'dataset': str,
        'slam_backend': str,
        'loop_closure_enabled': bool,
        'config': dict,  # Configuración completa
        'extraction_date': str,
        'total_frames': int,
        'num_keyframes': int,
        'final_map_points': int
    },
    'camera_poses': {
        frame_id: c2w_matrix (4x4 tensor)
    },
    'map_evolution': {
        kf_id: {
            'pcd': tensor,
            'pcd_colors': tensor,
            'pcd_ids': tensor
        }
    },
    'keyframes_info': {
        'ids': [kf_id, ...],
        'timestamps': [float, ...],
        'frame_to_kf': {frame_id: kf_id},
        'pcd_ranges': {kf_id: (start, end)}
    },
    'intrinsics': tensor
}
```

### Problemas de la estructura anterior:
- ❌ Guardaba nubes de puntos completas en cada keyframe (muy pesado)
- ❌ No capturaba explícitamente los eventos de loop closure
- ❌ No guardaba el estado before/after de los loop closures
- ❌ Incluía información redundante (`frame_to_kf` puede calcularse)

---

## Estructura NUEVA (basada en prueba.py)

```python
baseline = {
    'metadata': {
        'scene': str,                    # Renombrado de scene_name
        'dataset': str,
        'slam_backend': str,
        'timestamp': str,                # Renombrado de extraction_date
        'config_hash': str,              # Hash MD5 en lugar de config completa
        'intrinsics': tensor,            # Movido a metadata
        'loop_closure_enabled': bool,
        'total_frames': int,
        'num_keyframes': int,
        'num_loop_closures': int,        # Nuevo
        'final_map_points': int
    },
    'estimated_c2ws': {                  # Renombrado de camera_poses
        frame_id: c2w_matrix (4x4 tensor)
    },
    'kfs': [                             # Simplificado de keyframes_info
        frame_id_1, frame_id_2, ...
    ],
    'loop_closures': [                   # Nuevo
        {
            'frame_id': int,             # Frame donde se detecta el loop closure
            'before': {
                'estimated_c2ws': {frame_id: c2w},
                'kfs': [frame_id, ...]
            },
            'after': {
                'estimated_c2ws': {frame_id: c2w},
                'kfs': [frame_id, ...]
            }
        },
        ...
    ]
}
```

### Ventajas de la nueva estructura:
- ✅ Más ligera: no guarda nubes de puntos en cada keyframe
- ✅ Captura explícitamente eventos de loop closure
- ✅ Permite comparar estados before/after de loop closures
- ✅ Estructura más simple y clara
- ✅ `config_hash` en lugar de configuración completa (más ligero)
- ✅ Nombres más consistentes con el código (`estimated_c2ws` coincide con el atributo del SLAM)

---

## Cambios en el Código

### 1. Estructura de datos inicializada
```python
# ANTES
{
    'camera_poses': {},
    'map_evolution': {},
    'keyframes_info': {...}
}

# DESPUÉS
{
    'estimated_c2ws': {},
    'kfs': [],
    'loop_closures': []
}
```

### 2. Captura de estados durante procesamiento
- Se guarda un snapshot del estado **antes** del mapping
- Si ocurre un loop closure (cambio en `last_big_change_id`), se captura:
  - Estado "before" (del snapshot previo)
  - Estado "after" (estado actual después del loop closure)

### 3. Nueva función: `save_state_snapshot()`
Guarda el estado actual antes de cada operación de mapping (solo para ORB-SLAM):
- `estimated_c2ws`: poses de todas las cámaras
- `kfs`: lista de keyframes

### 4. Nueva función: `capture_loop_closure()`
Captura el evento de loop closure con:
- `frame_id`: frame donde se detectó
- `before`: estado previo al loop closure
- `after`: estado posterior al loop closure

### 5. Eliminada función: `complete_frame_to_kf()`
Ya no es necesaria porque no guardamos el mapeo frame→keyframe.

### 6. Validación actualizada
- Verifica nuevas claves: `estimated_c2ws`, `kfs`, `loop_closures`
- Valida estructura de cada evento de loop closure
- Verifica consistencia de `num_loop_closures` en metadata

---

## Compatibilidad hacia atrás

⚠️ **Los archivos `.pth` antiguos NO son compatibles con la nueva estructura.**

Si necesitas usar baselines antiguos, deberás:
1. Regenerarlos con el nuevo script, o
2. Crear un script de conversión (no recomendado, ya que no captura loop closures)

---

## Uso

### Generar un nuevo baseline
```bash
# Con loop closure (ORB-SLAM)
python scripts/extract_slam_baseline.py --dataset Replica --scene office0

# Sin loop closure
python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --no-loop-closure

# Con vanilla SLAM (sin loop closure)
python scripts/extract_slam_baseline.py --dataset Replica --scene office0 --slam vanilla
```

### Verificar estructura de un baseline
```bash
python scripts/verify_baseline_structure.py data/baselines/Replica_office0_orbslam2_with_lc.pth
```

---

## Notas técnicas

### ¿Por qué no guardar nubes de puntos en loop closures?
Según el comentario en `prueba.py`:
```python
# "points_3d": {       # Nube de puntos antes/después del loop closure
#     "pcd": ...,
#     "pcd_ids": ...,
#     "pcd_colors": ...
# } Decidimos no calcularlo porque no hará falta
```

Las nubes de puntos pueden ser muy grandes y ocupar mucho espacio. Para experimentos de evaluación, las poses de cámara y la lista de keyframes son suficientes para analizar el impacto de los loop closures.

### Detección de loop closures en ORB-SLAM
Los loop closures se detectan monitoreando `slam.last_big_change_id`:
- Este ID se actualiza cuando ORB-SLAM completa un loop closure y GBA (Global Bundle Adjustment)
- Cuando cambia, significa que las poses fueron optimizadas
- Capturamos el estado justo antes y después de este cambio

---

## Próximos pasos

1. Regenerar todos los baselines con la nueva estructura
2. Actualizar notebooks de visualización para la nueva estructura
3. Actualizar scripts de evaluación si usan los baselines
4. Documentar cómo usar la información de loop closures en análisis
