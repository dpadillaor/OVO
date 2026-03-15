# SAM3 Generator in OVO

## Summary

`SAM3Generator` is a **feature-extraction-only** wrapper around the SAM3 ViT backbone. It does NOT perform segmentation — it only extracts dense embeddings per masked crop. `fusion_encoders.py` provides the `FusionEncoderAdapter` ABC and concrete adapters (PE, DINO, SAM3) that integrate generators into the OVO pipeline.

---

## `SAM3Generator` (`sam3_generator.py:25-178`)

**Purpose**: ViT-based feature encoder. Produces L2-normalized embeddings per masked crop. Does not run the SAM3 segmentation pipeline.

### Constructor config keys (`sam3_generator.py:30-38`)

| Key | Default | Meaning |
|---|---|---|
| `components` | `"vit_only"` | Which sub-modules to load: `"vit_only"`, `"vit_neck"`, `"full"` |
| `load_from_hf` | `False` | Load checkpoint from HuggingFace |
| `checkpoint_path` | `None` | Local path to SAM3 checkpoint |
| `image_size` | `1008` | Resize target (square) before encoding |
| `use_half` | `False` | Cast model to FP16 |

### Model loading (`sam3_generator.py:64-81`)
Calls `build_sam3_image_model(checkpoint_path=..., load_from_HF=...)` from `thirdParty/sam3/sam3/model_builder.py`. Extracts sub-modules:
- `self.vit = model.backbone.vision_backbone.trunk` (always)
- `self.neck = model.backbone.vision_backbone.neck` (if `components` in `["vit_neck", "full"]`)
- `self.text_encoder = model.backbone.text_backbone` (if `components == "full"`)

The full segmentation decoder/mask head is discarded.

### Embedding dimensions (`sam3_generator.py:50-53`)
- `vit_only` → `embed_dim = 1024`
- `vit_neck` or `full` → `embed_dim = 256`

### Normalization (`sam3_generator.py:61-62`)
Resize: BICUBIC to `(image_size, image_size)`. Normalize: mean `(0.5, 0.5, 0.5)`, std `(0.5, 0.5, 0.5)` — **distinct from SAM2/CLIP normalization**.

### `encode_image` (`sam3_generator.py:108-147`)
- `@torch.no_grad()`
- Input: `torch.Tensor` shape `(B, 3, H, W)` or `(3, H, W)`, range `[0, 1]`
- Output: `torch.Tensor` shape `(B, embed_dim)`, L2-normalized
- For 4D spatial maps: applies `adaptive_avg_pool2d` to collapse spatial dims.

### `extract_sam3` (`sam3_generator.py:149-178`) — main entry point
- `@torch.no_grad()`
- Input:
  - `image`: shape `(3, H, W)`, range 0-255; or `np.ndarray (H, W, 3)` (auto-transposed)
  - `binary_maps`: `torch.Tensor` shape `(N, H, W)`, binary masks
- Output: `torch.Tensor` shape `(N, embed_dim)`
- Returns `torch.zeros((0, embed_dim))` for empty mask input.
- Calls `segment_utils.segmap2segimg(binary_maps, image, False, out_l=self.image_size)` to crop/resize each mask region.

### Key observation
`SAM3Generator` is NOT a segmentor — no `generate()` or `predict()` method. Masks must be provided externally. `thirdParty/sam3` is appended to `sys.path` at import time, failing fast if absent.

---

## `fusion_encoders.py`

### `FusionEncoderAdapter` (ABC) (`fusion_encoders.py:8-37`)
Abstract methods: `compute_and_update`, `update_objects`, `transfer_on_merge`, `cleanup_keyframe`.

### `SAM3FusionAdapter` (`fusion_encoders.py:150-233`)
- Wraps `SAM3Generator` as `self.generator`; storage key: `"ins_sam3_descriptors"`
- `compute_and_update`: calls `generator.extract_sam3(image, binary_maps).cpu()`, stores `sam3_embeds[idx:idx+1]` (shape `(1, embed_dim)`) per instance, calls `objects[ins_id].update_sam3(keyframes[storage_key])`
- `transfer_on_merge`: moves descriptor dict entry from `source_id` to `target_id`; last source wins on collision
- `cleanup_keyframe`: deletes `keyframes[storage_key][kf_id]`

### Storage key convention
`"ins_{encoder}_descriptors"` maps to top-level `keyframes` dict keys. Embeddings stored as shape `(1, embed_dim)` slices.

### Extension pattern
`DINOFusionAdapter` (`fusion_encoders.py:126-147`) is a no-op placeholder — confirms adding new adapters is low-friction. `PEFusionAdapter` is the reference implementation; SAM3 adapter mirrors it exactly.

---

## Recommendations

1. **SAM2 fusion adapter**: Mirror `SAM3FusionAdapter` exactly — replace `extract_sam3` with SAM2 generator method, change storage key and `obj.update_*` / `obj.to_update_*` attributes.
2. **SAM3 does not segment**: A segmentation wrapper for SAM3 requires `thirdParty/sam3` investigation — not covered here.
3. **Normalization difference**: SAM3 uses `(0.5, 0.5, 0.5)` mean/std. Handle separately from SAM2.
4. **`embed_dim` coupling**: A unified wrapper should expose `embed_dim` as a property.
5. **`Instance3D` dependency**: Verify `Instance3D` has `update_sam2` / `to_update_sam2` before implementing `SAM2FusionAdapter`.

---

## Files Not Analyzed (Out of Scope)
- `ovo/entities/instance3d.py` — needed to confirm `update_sam3` / `to_update_sam3` hooks
- `thirdParty/sam3/sam3/model_builder.py` — `build_sam3_image_model` full API
