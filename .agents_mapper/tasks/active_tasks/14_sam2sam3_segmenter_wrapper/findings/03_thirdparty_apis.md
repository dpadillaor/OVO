# ThirdParty APIs: SAM2 vs SAM3

## SAM2: `SAM2AutomaticMaskGenerator`

**File**: `thirdParty/segment-anything-2/sam2/automatic_mask_generator.py:36`

### Constructor (`lines 37-57`)
```python
SAM2AutomaticMaskGenerator(
    model: SAM2Base,
    points_per_side: Optional[int] = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.95,
    stability_score_offset: float = 1.0,
    mask_threshold: float = 0.0,
    box_nms_thresh: float = 0.7,
    crop_n_layers: int = 0,
    crop_nms_thresh: float = 0.7,
    crop_overlap_ratio: float = 512 / 1500,
    crop_n_points_downscale_factor: int = 1,
    point_grids: Optional[List[np.ndarray]] = None,
    min_mask_region_area: int = 0,
    output_mode: str = "binary_mask",  # or "uncompressed_rle" / "coco_rle"
    use_m2m: bool = False,
    multimask_output: bool = True,
    **kwargs,
)
```
Exactly one of `points_per_side` or `point_grids` must be set. Also has `from_pretrained(model_id)` classmethod.

### `generate(image)` (`line 169`)
- Input: `np.ndarray` — HWC uint8
- Returns: `List[Dict[str, Any]]`, one dict per mask:

| Key | Type | Notes |
|-----|------|-------|
| `segmentation` | `np.ndarray (H, W)` bool | Format controlled by `output_mode` |
| `bbox` | `list[float]` | XYWH format |
| `area` | `int` | Pixel count |
| `predicted_iou` | `float` | Model's own quality prediction |
| `point_coords` | `list[list[float]]` | Grid point that generated this mask |
| `stability_score` | `float` | Binarization stability measure |
| `crop_box` | `list[float]` | Crop region used, XYWH |

**Prompt model**: Fully automatic (point grid only). No text or box prompts.

---

## SAM3: No `automatic_mask_generator.py` Exists

`thirdParty/sam3/sam3/automatic_mask_generator.py` **does not exist**. SAM3 is a grounded detection/segmentation model — always prompt-conditioned.

### Closest equivalent: `Sam3Processor` (`sam3/model/sam3_image_processor.py:14`)

```python
Sam3Processor(
    model,                          # Sam3Image instance
    resolution: int = 1008,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
)
```

### Usage pattern (stateful session)
```python
state = processor.set_image(image)
state = processor.set_text_prompt("object", state)       # text-conditioned
# OR
state = processor.add_geometric_prompt([cx, cy, w, h], label=True, state)  # box prompt (normalized cx,cy,w,h)
```

### Output (written into `state` dict)
| Key | Type | Notes |
|-----|------|-------|
| `masks` | `torch.Tensor (N, 1, H, W)` bool | Binary masks at original resolution |
| `masks_logits` | `torch.Tensor (N, 1, H, W)` | Raw float logits |
| `boxes` | `torch.Tensor (N, 4)` | XYXY pixel coords |
| `scores` | `torch.Tensor (N,)` | `sigmoid(logit) * sigmoid(presence)` |

N = masks surviving `confidence_threshold`. Model construction: `build_sam3_image_model(...)` from `model_builder.py:560` (two-step: factory + `Sam3Processor(model)`).

---

## API Incompatibilities Summary

| Dimension | SAM2 | SAM3 |
|-----------|------|------|
| Prompt model | Fully automatic (point grid) | Always prompt-conditioned (text or box) |
| Interface style | Stateless `generate(image)` | Stateful session dict passed across calls |
| Output type | `List[Dict]`, one dict per mask | State dict with tensors (`masks`, `boxes`, `scores`) |
| Output bbox format | XYWH float list | XYXY pixel tensor |
| Score field | `predicted_iou` + `stability_score` | Single `score` (detection confidence) |
| Input image | HWC uint8 numpy | PIL, tensor, or numpy; resized to 1008×1008 |
| Missing SAM2 fields in SAM3 | — | `area`, `stability_score`, `point_coords`, `crop_box` |
| Model load | `from_pretrained(model_id)` classmethod | `build_sam3_image_model(...)` + `Sam3Processor(model)` |

---

## Wrapper Design Recommendation

Build a shared `BaseSegmenter` with a stateless `generate(image: np.ndarray, text_prompt: Optional[str] = None) -> List[Dict]` method:
- For SAM3: default `text_prompt` to `"object"`, manage state internally per call, convert tensors to numpy, compute `area` from mask, normalize bbox to XYWH, map `scores` → `predicted_iou`, stub `stability_score` / `point_coords` / `crop_box`.
- The stateful session must be hidden inside the SAM3 adapter — callers should see a stateless interface identical to SAM2.
