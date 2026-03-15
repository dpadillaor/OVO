# Tests — SAM2/SAM3 Segmenter Wrapper

## Unit Tests

### T1 — `SAM3AutomaticMaskGenerator.generate()` output format
**File**: `tests/test_sam3_automatic_mask_generator.py`
**What**: Mock `SAM3InteractiveImagePredictor` to return fake masks/iou_preds/low_res_masks.
Call `generator.generate(fake_image)` and assert the output is a `List[Dict]` with
keys `segmentation`, `area`, `bbox`, `predicted_iou`, `point_coords`,
`stability_score`, `crop_box` — identical to SAM2's output format.
**Why**: Confirms the interface contract is met before touching MaskGenerator.

### T2 — `SAM3AutomaticMaskGenerator` constructor
**What**: Assert that passing `points_per_side=16` builds `self.point_grids` of the
correct shape, and that the predictor is an instance of `SAM3InteractiveImagePredictor`.
**Why**: Guards against import or init regressions.

### T3 — `segment_utils.load_sam()` with `sam_version="3"`
**File**: `tests/test_segment_utils.py`
**What**: Mock `build_sam3_video_model` to return a fake video model with `.tracker`
and `.detector.backbone`. Assert `load_sam` returns a `SAM3AutomaticMaskGenerator`
instance when `sam_version == "3"`.
**Why**: Validates the factory branch without requiring a real checkpoint.

### T4 — `segment_utils.load_sam()` with `sam_version=""` and `"2.1"` still work
**What**: Regression test — existing SAM1/SAM2 paths are unaffected by the new branch.
**Why**: Prevents breaking existing experiments.

### T5 — `MaskGenerator` with `sam_version="3"` sets correct dtype
**File**: `tests/test_mask_generator.py`
**What**: Mock `load_sam` to return a dummy object. Instantiate `MaskGenerator` with
`config["sam_version"] = "3"`. Assert `self.dtype == torch.bfloat16` and TF32 flags
are enabled.
**Why**: Validates the dtype branch in `load_mask_generator`.

---

## Integration Tests (require checkpoint)

### T6 — End-to-end single frame segmentation with SAM3
**What**: Load a real SAM3 video model checkpoint. Run `MaskGenerator.get_masks(image)`
on a Replica frame. Assert output shapes: `seg_map (H, W)` and `binary_maps (N, H, W)`
with N > 0.
**Why**: Confirms the full pipeline works with real weights.

### T7 — SAM3 output is geometrically valid
**What**: For each mask in `binary_maps`, assert it is a non-empty boolean tensor,
that masks don't all overlap completely (basic over-segmentation sanity), and that
bboxes fit within image bounds.
**Why**: Catches silent failures where the model runs but produces degenerate output.

### T8 — SAM2 vs SAM3 mask count comparison (qualitative)
**What**: Run both SAM2 and SAM3 on the same Replica frame with `points_per_side=16`.
Log number of masks produced by each. No hard assertion — output is logged for manual
review.
**Why**: First-pass quality check to see if SAM3 over/under-segments vs SAM2.