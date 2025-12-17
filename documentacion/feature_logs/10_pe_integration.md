# Feature Log: PE Integration

## Objective
Add PE descriptor extraction parallel to CLIP in OVO's deferred semantic phase.

## Current Pipeline
**Phase 1**: Tracking ([ovo.py#L104](../../ovo/entities/ovo.py#L104)) → Queue keyframes  
**Phase 2**: Semantic ([ovo.py#L271](../../ovo/entities/ovo.py#L271)) → Extract CLIP → Update instances

**CLIP modes** ([clip_generator.py#L107-L141](../../ovo/entities/clip_generator.py#L107-L141)):
- Vanilla: 1 descriptor (masked crop)
- Multi-crop (default): 3 descriptors (global + masked + bbox) → fused

**PE strategy**: Use masked crop only (`also_bbox=False` in [segmap2segimg](../../ovo/utils/segment_utils.py#L29))

## PEGenerator Status
**Created** [pe_generator.py](../../ovo/entities/pe_generator.py):
- `encode_image` ([L63-L79](../../ovo/entities/pe_generator.py#L63-L79)): Processes (B,3,H,W) in [0,1]
- `extract_pe` ([L81-L114](../../ovo/entities/pe_generator.py#L81-L114)): Extracts from masked regions
- Text methods: `get_txt_embedding` ([L116](../../ovo/entities/pe_generator.py#L116)), `get_embed_txt_similarity` ([L123](../../ovo/entities/pe_generator.py#L123))

**Issue**: `mask_res` ([L24](../../ovo/entities/pe_generator.py#L24)) hardcoded to 336, should auto-detect from `model.image_size` (224/336/384/448 depending on PE model variant)

## Required Changes

### 1. Fix [pe_generator.py#L24](../../ovo/entities/pe_generator.py#L24)
```python
self.mask_res = config.get("mask_res", self.model.image_size)  # Auto-detect
```

### 2. OVO class ([ovo.py](../../ovo/entities/ovo.py))
- [L21](../../ovo/entities/ovo.py#L21): Add `self.pe_generator = PEGenerator(config["pe"], device) if "pe" in config else None`
- [L43-47](../../ovo/entities/ovo.py#L43-L47): Add `"ins_pe_descriptors": dict()` to keyframes
- After [L309](../../ovo/entities/ovo.py#L309): Add `_extract_pe()` method (mirror [_extract_clip](../../ovo/entities/ovo.py#L309-L320))
- After [L322](../../ovo/entities/ovo.py#L322): Add `_update_matched_objects_pe()` method (mirror [_update_matched_objects_clip](../../ovo/entities/ovo.py#L322-L342))
- [L271-297](../../ovo/entities/ovo.py#L271-L297): Update `_compute_semantic_info()` to extract PE if available

### 3. Instance3D class ([instance3d.py](../../ovo/entities/instance3d.py))
- Add: `self.pe_feature = None`, `self.pe_features_dict = {}`
- Add: `update_pe()` method (mirror [update_clip](../../ovo/entities/instance3d.py#L48))
- Update: [capture_dict](../../ovo/entities/instance3d.py#L75), [restore_dict](../../ovo/entities/instance3d.py#L103) for PE features

### Config
```python
config["pe"] = {
    "model_card": "PE-Core-L14-336",  # or T16-384, G14-448, etc.
    "use_half": False
}
```

## TODO
- [ ] Fix `mask_res` auto-detection ([pe_generator.py#L24](../../ovo/entities/pe_generator.py#L24))
- [ ] Add PE to OVO: init, extract method, update method ([ovo.py](../../ovo/entities/ovo.py))
- [ ] Extend Instance3D: storage + methods ([instance3d.py](../../ovo/entities/instance3d.py))
- [ ] Test end-to-end
- [ ] Benchmark PE vs CLIP
