# Feature Log: PE Integration

## Objective
Add PE descriptor extraction parallel to CLIP in OVO's deferred semantic phase.

## Current Pipeline
**Phase 1**: Tracking ([ovo.py#L104](../../ovo/entities/ovo.py#L104)) → Queue keyframes  
**Phase 2**: Semantic ([ovo.py#L348](../../ovo/entities/ovo.py#L348)) → Extract CLIP + PE → Update instances

**CLIP modes** ([clip_generator.py#L107-L141](../../ovo/entities/clip_generator.py#L107-L141)):
- Vanilla: 1 descriptor (masked crop)
- Multi-crop (default): 3 descriptors (global + masked + bbox) → fused

**PE strategy**: Use masked crop only (`also_bbox=False` in [segmap2segimg](../../ovo/utils/segment_utils.py#L29))

## PEGenerator Status ✅
**Created** [pe_generator.py](../../ovo/entities/pe_generator.py):
- `encode_image` ([L63-L79](../../ovo/entities/pe_generator.py#L63-L79)): Processes (B,3,H,W) in [0,1]
- `extract_pe` ([L81-L114](../../ovo/entities/pe_generator.py#L81-L114)): Extracts from masked regions
- Text methods: `get_txt_embedding` ([L116](../../ovo/entities/pe_generator.py#L116)), `get_embed_txt_similarity` ([L123](../../ovo/entities/pe_generator.py#L123))
- **Fixed**: `mask_res` ([L28](../../ovo/entities/pe_generator.py#L28)) auto-detects from `model.image_size`

## Implementation Status ✅

### 1. OVO class ([ovo.py](../../ovo/entities/ovo.py))
- [L43](../../ovo/entities/ovo.py#L43): `self.pe_generator` conditionally loaded
- [L50](../../ovo/entities/ovo.py#L50): `"ins_pe_descriptors": dict()` added to keyframes
- [L88-89, L99-100](../../ovo/entities/ovo.py#L88-L100): PE handled in `cpu()`/`cuda()`
- [L355-358](../../ovo/entities/ovo.py#L355-L358): Parallel PE extraction in `_compute_semantic_info()`
- [L384-386](../../ovo/entities/ovo.py#L384-L386): PE descriptors cleaned in `update_map()`
- [L429-434](../../ovo/entities/ovo.py#L429-L434): PE descriptors transferred during fusion
- [L487-531](../../ovo/entities/ovo.py#L487-L531): `_extract_pe()`, `_update_matched_objects_pe()`, `update_objects_pe()`
- [L610-612, L637-640](../../ovo/entities/ovo.py#L610-L640): PE save/restore in `capture_dict()`/`restore_dict()`

### 2. Instance3D class ([instance3d.py](../../ovo/entities/instance3d.py))
- [L33-34](../../ovo/entities/instance3d.py#L33-L34): `pe_feature`, `pe_feature_kf` attributes
- [L41](../../ovo/entities/instance3d.py#L41): `to_update_pe` flag
- [L86-88, L103, L107, L110](../../ovo/entities/instance3d.py#L86-L110): Set `to_update_pe=True` on updates
- [L161-195](../../ovo/entities/instance3d.py#L161-L195): `update_pe()` method with L1-norm aggregation
- [L207-208, L229-231](../../ovo/entities/instance3d.py#L207-L231): PE export/restore

## PE Data Storage

### In Memory
**OVO keyframes** ([ovo.py#L47-L52](../../ovo/entities/ovo.py#L47-L52)):
- `self.keyframes["ins_pe_descriptors"][kf_id][ins_id]` → Per-keyframe embeddings (tensor)

**Instance3D** ([instance3d.py#L33-34](../../ovo/entities/instance3d.py#L33-L34)):
- `self.pe_feature` → Aggregated descriptor (L1-norm across keyframes)
- `self.pe_feature_kf` → List of kf_ids with PE descriptors

### On Disk
**Debug mode** ([ovo.py#L610-612](../../ovo/entities/ovo.py#L610-L612)):
- `kf_{kf_id}_ins3d_{ins_id}_pe` → Per-keyframe embeddings

**Always saved** ([instance3d.py#L207-208](../../ovo/entities/instance3d.py#L207-L208)):
- `ins3d_{ins_id}_pe` → Aggregated descriptor
- `ins3d_{ins_id}_pe_kfs` → List of keyframe IDs

## Fusion Limitation & Solution ⚠️

### Current Issue
Loop-closure fusion ([ovo.py#L414](../../ovo/entities/ovo.py#L414)) uses `instance_utils.same_instance()` which **only checks CLIP similarity** (`th_cossim`). PE descriptors are transferred but NOT used for fusion decisions.

### Proposed Solution
Add config parameter to choose descriptor for fusion:
```python
config["fusion_descriptor"] = "clip"  # Options: "clip", "pe", "both"
```

Modify `instance_utils.same_instance()` to:
1. Accept `descriptor_type` parameter
2. Use `instance.pe_feature` when `descriptor_type == "pe"`
3. Combine both when `descriptor_type == "both"` (e.g., average cosine similarity)

**Files to modify**:
- [ovo/utils/instance_utils.py](../../ovo/utils/instance_utils.py): Update `same_instance()` signature
- [ovo.py#L414](../../ovo/entities/ovo.py#L414): Pass `config["fusion_descriptor"]` to `same_instance()`

## Config
```python
config["pe"] = {
    "model_card": "PE-Core-L14-336",  # or PE-T16-384, PE-G14-448, etc.
    "use_half": False
}
config["fusion_descriptor"] = "clip"  # TODO: Support "pe" or "both"
```

## Status
- ✅ Fix `mask_res` auto-detection
- ✅ Add PE to OVO: init, extract, update methods
- ✅ Extend Instance3D: storage + methods
- ✅ Fusion transfers PE descriptors
- ⚠️ Fusion decision still CLIP-only
- [ ] Add `fusion_descriptor` config option
- [ ] Test end-to-end
- [ ] Benchmark PE vs CLIP
