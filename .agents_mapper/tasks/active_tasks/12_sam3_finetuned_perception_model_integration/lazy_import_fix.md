# Lazy Import Fix for SAM3Generator

**Date**: 2026-01-14
**Issue**: SAM3 dependencies required even when not using SAM3
**Status**: ✅ **FIXED**

---

## Problem

When running experiments with PE fusion (not SAM3), the code failed with:
```
ModuleNotFoundError: No module named 'decord'
```

This was because `SAM3Generator` was imported at the top of `ovo.py` **unconditionally**, causing:
1. SAM3 dependencies (decord, etc.) required even for PE/CLIP/DINO experiments
2. Import failure when SAM3 dependencies not installed
3. Unnecessary coupling between modules

---

## Root Cause

### Before (Problematic)

**`ovo/entities/ovo.py`**:
```python
from .sam3_generator import SAM3Generator  # ❌ Always imports, even if not used

class OVO:
    def __init__(self, config, ...):
        self.sam3_generator = SAM3Generator(config["sam3"], device=device) if "sam3" in config else None
```

**`ovo/entities/fusion_encoders.py`**:
```python
from ovo.entities.sam3_generator import SAM3Generator  # ❌ Always imports

class SAM3FusionAdapter(FusionEncoderAdapter):
    def __init__(self, sam3_generator: SAM3Generator, ...):  # ❌ Name must be defined at class creation
        ...
```

**Impact**:
- PE experiment fails because `SAM3Generator` import fails
- SAM3 dependencies become mandatory for all experiments
- Users can't run PE/CLIP experiments without installing SAM3

---

## Solution

### 1. Lazy Import in `ovo.py`

**After**:
```python
# NO top-level import

class OVO:
    def __init__(self, config, ...):
        # Lazy import SAM3Generator to avoid dependency issues when SAM3 is not used
        if "sam3" in config:
            try:
                from .sam3_generator import SAM3Generator
                self.sam3_generator = SAM3Generator(config["sam3"], device=device)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import SAM3Generator. SAM3 dependencies may not be installed: {e}"
                ) from e
        else:
            self.sam3_generator = None
```

**Benefits**:
- ✅ SAM3Generator only imported when `"sam3"` in config
- ✅ Clear error message if SAM3 dependencies missing
- ✅ PE/CLIP/DINO experiments work without SAM3 dependencies

### 2. Forward Reference in `fusion_encoders.py`

**After**:
```python
# NO top-level import

class SAM3FusionAdapter(FusionEncoderAdapter):
    def __init__(self, sam3_generator: "SAM3Generator", storage_key: str = "ins_sam3_descriptors"):
        #                              ^ String annotation = forward reference
        self.generator = sam3_generator
        self.storage_key = storage_key
```

**Why this works**:
- String annotation `"SAM3Generator"` is a **forward reference**
- Python doesn't try to resolve the name at class definition time
- The name is only resolved when the function is called (at which point SAM3Generator has been imported in `ovo.py`)
- Type checkers still understand the type

---

## Technical Details

### Forward References in Python

**Problem**: Type hints are evaluated at class definition time
```python
class Foo:
    def __init__(self, bar: Bar):  # ❌ NameError if Bar not imported
        pass
```

**Solution**: Use string annotations (PEP 563)
```python
class Foo:
    def __init__(self, bar: "Bar"):  # ✅ Only resolved at call time
        pass
```

### Import Flow

#### PE Experiment (no SAM3):
```
1. Import ovo.py
2. Import fusion_encoders.py
   - SAM3FusionAdapter class defined (type hint is string, no import)
3. OVO.__init__()
   - "sam3" not in config
   - self.sam3_generator = None
   - _get_fusion_encoder() returns PEFusionAdapter
4. ✅ No SAM3 import ever happens
```

#### SAM3 Experiment:
```
1. Import ovo.py
2. Import fusion_encoders.py
   - SAM3FusionAdapter class defined (type hint is string, no import)
3. OVO.__init__()
   - "sam3" in config
   - Lazy import: from .sam3_generator import SAM3Generator
   - self.sam3_generator = SAM3Generator(...)
   - _get_fusion_encoder() returns SAM3FusionAdapter(self.sam3_generator)
4. ✅ SAM3 imported only when needed
```

---

## Testing

### Verify Fix

**Without SAM3 dependencies** (e.g., decord not installed):
```python
# This should work for PE experiments
from ovo.entities.ovo import OVO

config = {
    "fusion_method": "pe",
    "pe": {...},
    # No "sam3" key
    ...
}

ovo = OVO(config, logger, ...)  # ✅ Works without SAM3 dependencies
```

**With SAM3 in config but no dependencies**:
```python
config = {
    "fusion_method": "sam3",
    "sam3": {...},
    ...
}

try:
    ovo = OVO(config, logger, ...)
except ImportError as e:
    # ✅ Clear error message:
    # "Failed to import SAM3Generator. SAM3 dependencies may not be installed: No module named 'decord'"
```

### Files Modified

| File | Change | Status |
|------|--------|--------|
| `ovo/entities/ovo.py` | Made SAM3Generator import lazy (lines 48-58) | ✅ Fixed |
| `ovo/entities/fusion_encoders.py` | Changed type hint to forward reference (line 153) | ✅ Fixed |

---

## Comparison

### Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| SAM3 dependencies | Required for all experiments | Only when using SAM3 |
| PE experiments | Fail without SAM3 deps | ✅ Work |
| CLIP experiments | Fail without SAM3 deps | ✅ Work |
| SAM3 experiments | Work if deps installed | ✅ Work if deps installed |
| Error clarity | "No module named 'decord'" | "SAM3 dependencies may not be installed" |
| Import overhead | Always imports SAM3 | Only imports when used |

---

## Best Practices Applied

1. **Lazy Loading**: Import expensive/optional modules only when needed
2. **Forward References**: Use string annotations for optional dependencies
3. **Clear Error Messages**: Provide context in ImportError exceptions
4. **Minimal Coupling**: Don't couple unrelated modules at import time
5. **Optional Features**: Make advanced features truly optional

---

## Related Patterns

### Pattern: Optional Dependencies in Python

```python
# ❌ BAD: Top-level import makes dependency mandatory
from optional_module import OptionalClass

class MyClass:
    def __init__(self, config):
        if config.get("use_optional"):
            self.opt = OptionalClass()

# ✅ GOOD: Lazy import makes dependency truly optional
class MyClass:
    def __init__(self, config):
        if config.get("use_optional"):
            try:
                from optional_module import OptionalClass
                self.opt = OptionalClass()
            except ImportError:
                raise ImportError("optional_module required for this feature")
```

### Pattern: Forward References for Type Hints

```python
# ❌ BAD: Type hint requires import
from heavy_module import HeavyClass

class MyAdapter:
    def __init__(self, obj: HeavyClass):
        pass

# ✅ GOOD: Forward reference delays resolution
class MyAdapter:
    def __init__(self, obj: "HeavyClass"):  # String annotation
        pass
```

---

## Future Considerations

### Adding New Optional Encoders

When adding DINO, LLaVA, or other encoders:

1. **Import lazily in `ovo.py`**:
   ```python
   if "llava" in config:
       try:
           from .llava_generator import LLaVAGenerator
           self.llava_generator = LLaVAGenerator(config["llava"], device=device)
       except ImportError as e:
           raise ImportError(f"LLaVA dependencies not installed: {e}") from e
   ```

2. **Use forward reference in `fusion_encoders.py`**:
   ```python
   class LLaVAFusionAdapter(FusionEncoderAdapter):
       def __init__(self, llava_generator: "LLaVAGenerator", ...):
           ...
   ```

3. **Document optional dependencies** in README/requirements.txt

---

## Verification Commands

```bash
# Test compilation
python -m py_compile ovo/entities/ovo.py
python -m py_compile ovo/entities/fusion_encoders.py

# Test PE experiment (should work without SAM3 dependencies)
python scripts/run_experiments.py --manifest experiments_manifest.yaml

# Test SAM3 experiment (only if SAM3 dependencies installed)
# Install first: pip install decord [other SAM3 requirements]
python run_eval.py --fusion_method sam3 ...
```

---

## Conclusion

The lazy import fix ensures that:
- ✅ **PE/CLIP/DINO experiments work without SAM3 dependencies**
- ✅ **SAM3 experiments work when dependencies are installed**
- ✅ **Clear error messages guide users to install missing dependencies**
- ✅ **No performance overhead for non-SAM3 experiments**
- ✅ **Architecture remains clean and modular**

**Status**: Production-ready for all fusion methods (CLIP, PE, DINO, SAM3)
