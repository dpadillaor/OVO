# SAM2 Integration in OVO

## How SAM2 is instantiated (`segment_utils.py:269-309`)

`load_sam()` is the sole factory. Dispatches on `config["sam_version"]`:
- Empty string → SAM1 (`segment_anything` library)
- Any other value (e.g. `"2.1"`) → SAM2 (`sam2` library)

For SAM2:
```python
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
sam = build_sam2(model_cfg, checkpoint_path, device=device, mode="eval", apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(model=sam, points_per_side=..., pred_iou_thresh=..., ...)
```
Returns the `SAM2AutomaticMaskGenerator` object (not the raw model). `apply_postprocessing=False` — OVO handles post-processing itself.

## Config keys consumed (`config["sam"]`)
`sam_version`, `sam_encoder`, `sam_ckpt_path`, `points_per_side`, `nms_iou_th`, `stability_score_thresh`, `min_mask_region_area`, `use_m2m`

## Interface called on the model
Only the **automatic mask generator** interface is used — no point/box prompts. The `.generate(image)` call is made inside `MaskGenerator.get_masks()` in `mask_generator.py`.

## Output format consumed
SAM2 returns a list of dicts. Fields consumed per dict:
- `segmentation` — bool ndarray `(H, W)`
- `predicted_iou` — float
- `stability_score` — float

Processed by `mask2segmap()` (`segment_utils.py:12`) and `masks_update()` (`segment_utils.py:173`) into `(seg_map, binary_maps)` tensors.

## Orchestration entry points (`ovo.py`)
- `self.mask_generator = MaskGenerator(config["sam"], scene_name, device=device)` — init at `ovo.py:60`
- **Sole call site**: `OVO._get_masks()` at `ovo.py:231-242` calls `self.mask_generator.get_masks(image, frame_id)`
  - Input `image`: `np.ndarray (H, W, 3)`, RGB uint8
  - Called from `detect_and_track_objects()` at `ovo.py:203`
- Device management: `mask_generator.cpu()` / `.cuda()` at `ovo.py:149` / `162`

## Critical architectural distinction
**SAM2 = segmenter only** (produces masks, no descriptors).
**SAM3 = feature encoder only** (produces embeddings for fusion, no masks).
They are completely separate systems: `config["sam"]` vs `config["sam3"]`.

## Key file not yet analyzed
`ovo/entities/mask_generator.py` — contains the actual `.generate(image)` call and precomputed mask loading logic.
