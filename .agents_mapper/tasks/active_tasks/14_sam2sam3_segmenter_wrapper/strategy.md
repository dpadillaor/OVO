# Strategy — SAM2/SAM3 Segmenter Wrapper

## Goal

Allow OVO to use either SAM2 or SAM3 as the per-frame segmenter by setting
`config["sam"]["sam_version"]` to `"2.x"` or `"3"` respectively. No downstream
code changes required — the wrapper exposes the same `.generate(image)` interface.

---

## Architecture

```
MaskGenerator
  └── load_mask_generator()          ← extension point (sam_version branch)
        ├── sam_version == ""        → SAM1 (existing)
        ├── sam_version == "2.x"     → SAM2AutomaticMaskGenerator (existing)
        └── sam_version == "3"       → SAM3AutomaticMaskGenerator (NEW)

segment_utils.load_sam()             ← factory, returns .generate()-compatible object
  └── sam_version == "3"             → build_sam3_video_model + SAM3AutomaticMaskGenerator

ovo/entities/sam3_automatic_mask_generator.py   ← already created
  └── SAM3AutomaticMaskGenerator
        ├── SAM3InteractiveImagePredictor (sam3.model.sam1_task_predictor)
        └── AMG utilities from sam2.utils.amg (geometry only, model-agnostic)
```

---

## Phases

### Phase 1 — SAM3AutomaticMaskGenerator ✅ DONE
File created at `ovo/entities/sam3_automatic_mask_generator.py`.
Mirrors `SAM2AutomaticMaskGenerator` exactly, replacing:
- `SAM2Base` → `Sam3TrackerBase`
- `SAM2ImagePredictor` → `SAM3InteractiveImagePredictor`
AMG utilities reused from `sam2.utils.amg`.

### Phase 2 — `segment_utils.load_sam()` integration
Add `sam_version == "3"` branch in `ovo/utils/segment_utils.py`:

```python
elif sam_version == "3":
    from ovo.entities.sam3_automatic_mask_generator import SAM3AutomaticMaskGenerator
    from sam3.model_builder import build_sam3_video_model

    video_model = build_sam3_video_model(
        checkpoint_path=config.get("sam_ckpt_path") or None,
        load_from_HF=True,
        device=device,
    )
    tracker = video_model.tracker
    tracker.backbone = video_model.detector.backbone

    mask_generator = SAM3AutomaticMaskGenerator(
        model=tracker,
        points_per_side=config.get("points_per_side", 32),
        pred_iou_thresh=config.get("nms_iou_th", 0.8),
        stability_score_thresh=config.get("stability_score_thresh", 0.95),
        min_mask_region_area=config.get("min_mask_region_area", 0),
        use_m2m=config.get("use_m2m", False),
    )
    return mask_generator
```

### Phase 3 — `mask_generator.load_mask_generator()` integration
Add `sam_version == "3"` branch in `ovo/entities/mask_generator.py`:

```python
elif sam_version == "3":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    self.dtype = torch.bfloat16
```

The device management chain `self.mask_generator.predictor.model` is the same
attribute name in `SAM3InteractiveImagePredictor` — verify at integration time.

### Phase 4 — Config YAML
Add example config for SAM3 segmentor to `data/working/configs/`:

```yaml
sam:
  sam_version: "3"
  sam_ckpt_path: null       # null = auto-download from HuggingFace (facebook/sam3)
  points_per_side: 16
  nms_iou_th: 0.8
  stability_score_thresh: 0.95
  min_mask_region_area: 100
  use_m2m: false

sam3:                        # feature encoder — separate config, unchanged
  components: "vit_neck"
  checkpoint_path: "..."
```

---

## Key Design Decisions

- **Separate checkpoints**: SAM3 segmentor (`config["sam"]`) and SAM3 feature encoder
  (`config["sam3"]`) use separate checkpoints. The segmentor needs the full video model
  (detector + tracker); the feature encoder only needs the ViT backbone.
- **AMG utilities from SAM2**: `sam2.utils.amg` utilities are pure geometry/data ops
  with no model dependency — safe to reuse.
- **`build_sam3_video_model`**: Required for `SAM3InteractiveImagePredictor` which needs
  `Sam3TrackerBase`. The image model (`build_sam3_image_model`) does not expose the
  tracker interface.
- **`tracker.backbone = video_model.detector.backbone`**: Required so the tracker can
  run `forward_image` using the detector's shared vision backbone.

---

## Open Risks

1. **Warm-up calls**: Two warm-up `generate()` calls are made for SAM2 in
   `load_mask_generator`. Verify SAM3 tolerates them or skip for `sam_version == "3"`.
2. **`_bb_feat_sizes` mismatch**: `SAM3InteractiveImagePredictor` uses
   `[(288, 288), (144, 144), (72, 72)]` vs SAM2's sizes. Internal to the predictor
   but verify mask output resolution is correct at integration time.
3. **HuggingFace authentication**: `build_sam3_video_model(load_from_HF=True)` requires
   `hf auth login` with approved access to `facebook/sam3`.
