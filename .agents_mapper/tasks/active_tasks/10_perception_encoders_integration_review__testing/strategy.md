# Strategy: Perception Encoders Integration

## 1. Patrón de Selective Computation

### Problema
El código actual computa tanto CLIP como PE cuando PE está configurado, independientemente del `fusion_method`. Esto es ineficiente y conceptualmente incorrecto.

### Corrección Arquitectónica
- **CLIP debe calcularse SIEMPRE** - Es el backbone semántico core de OVO (tracking, queries, clasificación)
- **PE/DINO solo cuando se usan para fusión** - Son alternativas específicas para el proceso de fusion

### Arquitectura: "Always-CLIP + Optional Fusion Encoder"

```
┌─────────────────────────────────────────────────────────────────┐
│                    _compute_semantic_info()                     │
├─────────────────────────────────────────────────────────────────┤
│  1. CLIP (SIEMPRE)                                              │
│     └── extract_clip() → update_matched_objects_clip()          │
│                                                                 │
│  2. Fusion Encoder (CONDICIONAL)                                │
│     └── if fusion_method in ["pe", "dino"]:                     │
│           └── extract_{method}() → update_matched_objects_...() │
└─────────────────────────────────────────────────────────────────┘
```

### Implementación: FusionEncoderAdapter Pattern

**Nuevo archivo**: `ovo/entities/fusion_encoders.py`

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import torch

class FusionEncoderAdapter(ABC):
    """
    Adapter for optional fusion encoders (PE, DINO, etc.)

    Encapsulates all encoder-specific logic for:
    - Descriptor extraction
    - Storage management
    - Instance updates
    - Merge transfers
    """

    @abstractmethod
    def compute_and_update(self, image, binary_maps, matched_ins_ids, kf_id, keyframes, objects):
        """Extract embeddings and update instances."""
        pass

    @abstractmethod
    def update_objects(self, objects, keyframes):
        """Batch update all objects with fused descriptors."""
        pass

    @abstractmethod
    def transfer_on_merge(self, source_ids, target_id, keyframes):
        """Transfer descriptors during instance merge."""
        pass

    @abstractmethod
    def cleanup_keyframe(self, kf_id, keyframes):
        """Remove descriptors for deleted keyframe."""
        pass


class PEFusionAdapter(FusionEncoderAdapter):
    """Adapter for Perception Encoder fusion."""

    def __init__(self, pe_generator, storage_key="ins_pe_descriptors"):
        self.generator = pe_generator
        self.storage_key = storage_key

    def compute_and_update(self, image, binary_maps, matched_ins_ids, kf_id, keyframes, objects):
        """Extract PE embeddings and store in keyframes + update instances."""
        pe_embeds = self.generator.extract_pe(image, binary_maps).cpu()

        # Initialize storage for keyframe if needed
        if kf_id not in keyframes[self.storage_key]:
            keyframes[self.storage_key][kf_id] = {}

        # Store embeddings and update instances
        for idx, ins_id in enumerate(matched_ins_ids):
            keyframes[self.storage_key][kf_id][ins_id] = pe_embeds[idx:idx+1]
            if ins_id in objects:
                objects[ins_id].update_pe(keyframes[self.storage_key])

    def update_objects(self, objects, keyframes):
        """Batch update all objects with fused PE descriptors."""
        for obj in objects.values():
            if obj.to_update_pe:
                obj.update_pe(keyframes[self.storage_key])

    def transfer_on_merge(self, source_ids, target_id, keyframes):
        """Transfer PE descriptors from source instances to target during merge."""
        for kf_id in keyframes[self.storage_key]:
            for source_id in source_ids:
                if source_id in keyframes[self.storage_key][kf_id]:
                    keyframes[self.storage_key][kf_id][target_id] = \
                        keyframes[self.storage_key][kf_id].pop(source_id)

    def cleanup_keyframe(self, kf_id, keyframes):
        """Remove PE descriptors for deleted keyframe."""
        if kf_id in keyframes[self.storage_key]:
            del keyframes[self.storage_key][kf_id]


class DINOFusionAdapter(FusionEncoderAdapter):
    """Adapter for DINO fusion (placeholder for future integration)."""

    def __init__(self, dino_generator, storage_key="ins_dino_descriptors"):
        self.generator = dino_generator
        self.storage_key = storage_key

    # Implement methods following same pattern as PEFusionAdapter
    # when DINO integration is ready
    pass
```

### Modificaciones en `ovo.py`

**1. Añadir helper para obtener fusion encoder:**

```python
def _get_fusion_encoder(self) -> Optional[FusionEncoderAdapter]:
    """Get the fusion encoder adapter based on config, or None if CLIP-only."""
    fusion_method = self.fusion_method.lower()

    if fusion_method == "pe" and self.pe_generator is not None:
        return PEFusionAdapter(self.pe_generator)
    elif fusion_method == "dino" and self.dino_generator is not None:
        return DINOFusionAdapter(self.dino_generator)

    # CLIP fusion uses CLIP features directly, no extra encoder needed
    return None
```

**2. Refactorizar `_compute_semantic_info()`:**

```python
def _compute_semantic_info(self, image, binary_maps, matched_ins_ids, kf_id):
    """
    Compute semantic descriptors for matched instances.

    CLIP is ALWAYS computed (core semantic backbone for tracking, queries, etc.)
    Additional fusion encoders (PE, DINO) are computed only when used for fusion.
    """
    # ========================================
    # STEP 1: CLIP - Always compute (core semantic)
    # ========================================
    clip_embeds = self._extract_clip(image, binary_maps).cpu()
    self._update_matched_objects_clip(clip_embeds, matched_ins_ids, kf_id)

    # ========================================
    # STEP 2: Fusion Encoder - Conditional
    # ========================================
    if self.fusion_encoder is not None:
        self.fusion_encoder.compute_and_update(
            image, binary_maps, matched_ins_ids, kf_id,
            self.keyframes, self.objects
        )
```

---

## 2. Validación de Configuración

### Problema
Usuario puede configurar `fusion_method="pe"` sin inicializar PE generator, causando fallo en runtime.

### Solución: Validación Temprana en `__init__`

```python
def _validate_fusion_config(self):
    """Validate that fusion_method has required generator available."""
    method = self.fusion_method.lower()

    validation_map = {
        "pe": (self.pe_generator, "PE generator", "pe"),
        "dino": (self.dino_generator, "DINO generator", "dino"),
        "clip": (self.clip_generator, "CLIP generator", "clip"),
    }

    if method in validation_map:
        generator, name, config_key = validation_map[method]
        if generator is None:
            raise ValueError(
                f"fusion_method='{method}' requires {name} to be configured. "
                f"Add '{config_key}' key to config or change fusion_method."
            )
```

**Ubicación**: Llamar después de crear generators y fusion strategy (línea ~70).

---

## 3. Beneficios del Patrón

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Claridad** | Lógica PE/CLIP mezclada en ovo.py | Encapsulada en adapters |
| **Extensibilidad** | Añadir DINO requiere cambios en múltiples lugares | Solo crear DINOFusionAdapter |
| **Testabilidad** | Difícil testear PE aislado | Adapters son unidades independientes |
| **Eficiencia** | Computa ambos siempre | Solo lo necesario |
| **Consistencia** | CLIP podría omitirse accidentalmente | CLIP garantizado siempre |

---

## 4. Archivos a Modificar

### Nuevos
- `ovo/entities/fusion_encoders.py` - Adapter pattern

### Modificar
- `ovo/entities/ovo.py`:
  - Import de fusion_encoders
  - `__init__`: crear `self.fusion_encoder` via `_get_fusion_encoder()`
  - `__init__`: llamar `_validate_fusion_config()`
  - `_compute_semantic_info()`: refactorizar con patrón Always-CLIP
  - `update_map()`: usar `fusion_encoder.transfer_on_merge()` y `cleanup_keyframe()`
  - `update_objects_pe()`: delegar a `fusion_encoder.update_objects()` o eliminar

---

## 5. Consideraciones

### Thresholds PE vs CLIP
- Actualmente PE usa mismos thresholds que CLIP (`th_cossim=0.81`)
- Puede no ser óptimo para PE (distribución de embeddings diferente)
- **Fase futura**: Añadir thresholds específicos por método en config

### Legacy Code
- `instance_utils.same_instance()` hardcoded a CLIP - NO tocar (deprecated)
- `old_restore()` no soporta PE - Documentar como known limitation

### Performance
- El adapter se crea en cada llamada a `_get_fusion_encoder()`
- **Optimización opcional**: Cachear adapter en `__init__` como `self.fusion_encoder`
