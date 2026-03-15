# MaskGenerator Class

**File**: `ovo/entities/mask_generator.py`

## Summary

`MaskGenerator` is the sole wrapper between OVO and the underlying SAM backbone. It handles model loading (delegated to `segment_utils.load_sam`), optional precomputation of masks to disk, and per-frame mask retrieval via `get_masks()`. There is already a SAM-version detection branch keyed on `config["sam_version"]`, but it controls only dtype/TF32 flags — no behavioral branching of the generator call itself exists yet.

---

## Constructor `__init__` (`mask_generator.py:16-37`)
- `self.precomputed = config["precomputed"]` — controls disk load vs live generation.
- `self.masks_path` built from `config["masks_base_path"] + scene_name`.
- NMS thresholds from config: `nms_iou_th=0.8`, `nms_score_th=0.7`, `nms_inner_th=0.5`.
- `self.multi_crop = config.get("multi_crop", False)` — stored but never used (dead config).
- If masks directory already exists, `self.mask_generator = None` (model skipped); loaded lazily by `precompute()`.

## `load_mask_generator` (`mask_generator.py:39-53`)
The only existing SAM-version-specific branch — primary extension point for SAM3.
- `sam_version == ""` → `self.dtype = torch.float32` (SAM1/legacy)
- Any non-empty `sam_version` → TF32 flags + `self.dtype = torch.bfloat16` (SAM2)
- Delegates to `segment_utils.load_sam(config, device=self.device)` at line 48.
- Warm-up: two calls to `self.mask_generator.generate(np.random.rand(512,512,3).astype(np.float32))` at lines 52-53.

## `segment` (`mask_generator.py:101-120`) — core inference
- `@torch.no_grad`, runs inside `torch.autocast`.
- **Sole live `.generate(image)` call at line 113** — expects `(H, W, 3)` RGB numpy array.
- Empty result guard: returns `(np.array([]), np.array([]))` if no masks.
- Post-processing:
  1. `segment_utils.masks_update(masks, iou_thr, score_thr, inner_thr)` — NMS/filtering
  2. `segment_utils.mask2segmap(masks_default, image)` → `(seg_map, binary_maps)`

## `get_masks` (`mask_generator.py:81-99`) — public API
- Dispatches to disk load or `segment()`, returns tensors on `self.device`.
- `seg_map` shape: `(H, W)`, pixel values in `[-1, N)` (-1 = unassigned).
- `binary_maps` shape: `(N, H, W)`, values 0 or 1.

## `precompute` (`mask_generator.py:122-151`)
- Batch offline mask generation. Lazy-loads model if `self.mask_generator is None`.
- Frame IDs: `[i for i in range(len(dataset)) if i % segment_every == 0]`.

## Device management: `to / cpu / cuda` (`mask_generator.py:55-79`)
- All guard with `if self.mask_generator:` — safe when model is None.
- **Model access path**: `self.mask_generator.predictor.model` — SAM2-specific; fragile for SAM3.

## Disk I/O
- `_save_masks` (`mask_generator.py:153-168`): `{frame_id:04d}_seg_map_default.npy` + `{frame_id:04d}_bmap_default.npy`
- `_load_masks` (`mask_generator.py:170-195`): missing bmap → fallback reconstruction from seg_map (marked `TODO`).

---

## Key Risks for SAM3 Integration

1. **`.generate()` output format**: `masks_update` / `mask2segmap` expect SAM2's `List[Dict]` schema. If SAM3 returns differently, these break.
2. **Model access chain**: `self.mask_generator.predictor.model` is SAM2-specific. `to()` / `cpu()` / `cuda()` will silently fail for SAM3 if nesting differs.
3. **`segment()` doesn't guard `None` model**: unlike `precompute()`, no lazy-load — would raise `AttributeError`.

## Recommendations

1. Add `elif sam_version == "3":` branch in `load_mask_generator` for SAM3-specific init.
2. Abstract `predictor.model` access into a helper or version-branch it.
3. Add `None` guard or lazy-load in `segment()`.
4. Handle SAM3's stateful session API inside the SAM3 adapter so `segment()` callers see a stateless interface.
