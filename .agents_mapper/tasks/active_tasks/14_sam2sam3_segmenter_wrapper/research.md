# Research Summary: SAM2/SAM3 Segmenter Wrapper

**Task**: Create a unified segmenter wrapper that abstracts over SAM2 and SAM3, allowing model selection via configuration without changing downstream code. Requires analyzing existing SAM2 integration, SAM3 API differences, and designing a common interface.
**Date**: 2026-03-15
**Sources**: `01_sam2_ovo_integration.md`, `02_sam3_generator.md`, `03_thirdparty_apis.md`, `04_mask_generator.md`

---

## Executive Summary

SAM2 is integrated into OVO as a fully automatic, stateless segmenter accessed through `MaskGenerator` -> `segment_utils.load_sam()`, producing `List[Dict]` masks consumed downstream. SAM3 has a dual role in the codebase: it is already used as a **feature encoder** (not a segmenter) via `SAM3Generator` and `SAM3FusionAdapter`, while its **segmentation capability** is entirely unexploited — and there is no `automatic_mask_generator.py` equivalent in `thirdParty/sam3`. SAM3 segmentation is always prompt-conditioned and stateful, which is fundamentally incompatible with SAM2's stateless `generate(image)` API. The wrapper must hide SAM3's stateful session management and prompt requirement behind a shared stateless interface while also carefully translating output formats so `masks_update` and `mask2segmap` continue working unchanged.

---

## Task Context

The goal is a unified `BaseSegmenter` abstraction that routes segmentation calls to either SAM2 or SAM3 based on config, with no changes required in `MaskGenerator`, `ovo.py`, or any downstream consumer. The task starts from analysis of the current SAM2 integration path and SAM3's available APIs, then derives the wrapper design from the delta between them.

---

## Key Components Identified

### SAM2 Integration Path

| Component | Location | Purpose | Source |
|-----------|----------|---------|--------|
| `load_sam()` | `segment_utils.py:269-309` | Sole factory; dispatches on `config["sam_version"]` | 01, 04 |
| `build_sam2()` | `thirdParty/segment-anything-2/sam2/build_sam.py` | Constructs raw SAM2 model | 01 |
| `SAM2AutomaticMaskGenerator` | `thirdParty/segment-anything-2/sam2/automatic_mask_generator.py:36` | Wraps SAM2 for automatic mask generation | 01, 03 |
| `MaskGenerator` | `ovo/entities/mask_generator.py` | OVO-level wrapper; manages precompute/live, device movement, NMS | 04 |
| `MaskGenerator.load_mask_generator` | `mask_generator.py:39-53` | Only existing SAM-version branch; primary extension point | 04 |
| `MaskGenerator.segment` | `mask_generator.py:101-120` | Sole live `.generate(image)` call site (line 113) | 04 |
| `MaskGenerator.get_masks` | `mask_generator.py:81-99` | Public API consumed by `ovo.py` | 04 |
| `masks_update()` | `segment_utils.py:173` | NMS/filtering on SAM2 `List[Dict]` output | 01, 04 |
| `mask2segmap()` | `segment_utils.py:12` | Converts filtered masks to `(seg_map, binary_maps)` tensors | 01, 04 |
| `OVO._get_masks()` | `ovo.py:231-242` | Calls `mask_generator.get_masks(image, frame_id)` | 01 |
| `OVO.__init__` | `ovo.py:60` | Instantiates `MaskGenerator` | 01 |
| Device helpers | `ovo.py:149, 162` | `mask_generator.cpu()` / `.cuda()` | 01 |

**Details:**
- `apply_postprocessing=False` is passed to `build_sam2` — OVO owns all post-processing.
- Config keys in `config["sam"]`: `sam_version`, `sam_encoder`, `sam_ckpt_path`, `points_per_side`, `nms_iou_th`, `stability_score_thresh`, `min_mask_region_area`, `use_m2m`.
- The image passed to `.generate()` is `np.ndarray (H, W, 3)` RGB uint8.

### SAM3 as Feature Encoder (Existing, Separate)

| Component | Location | Purpose | Source |
|-----------|----------|---------|--------|
| `SAM3Generator` | `ovo/entities/sam3_generator.py:25-178` | ViT-based feature encoder; no segmentation | 02 |
| `SAM3Generator.extract_sam3` | `sam3_generator.py:149-178` | Main entry point; takes image + binary_maps, returns embeddings | 02 |
| `SAM3Generator.encode_image` | `sam3_generator.py:108-147` | Encodes a cropped/resized image tensor to L2-normalized embedding | 02 |
| `build_sam3_image_model` | `thirdParty/sam3/sam3/model_builder.py` | Model factory for SAM3 | 02, 03 |
| `SAM3FusionAdapter` | `ovo/entities/fusion_encoders.py:150-233` | Plugs `SAM3Generator` into fusion pipeline | 02 |
| `FusionEncoderAdapter` (ABC) | `fusion_encoders.py:8-37` | Abstract base for all fusion encoder adapters | 02 |

**Details:**
- `SAM3Generator` discards SAM3's decoder/mask head — only ViT trunk (and optionally neck/text backbone) are retained.
- This is configured via `config["sam3"]`, completely separate from `config["sam"]`.
- Normalization: SAM3 uses mean/std `(0.5, 0.5, 0.5)`, distinct from SAM2 and CLIP.
- Embed dims: `vit_only` → 1024; `vit_neck` or `full` → 256.

### SAM3 as Segmenter (Not Yet Used in OVO)

| Component | Location | Purpose | Source |
|-----------|----------|---------|--------|
| `Sam3Processor` | `thirdParty/sam3/sam3/model/sam3_image_processor.py:14` | Stateful segmentation session | 03 |
| `Sam3Processor.set_image` | `sam3_image_processor.py` | Initializes state dict for a given image | 03 |
| `Sam3Processor.set_text_prompt` | `sam3_image_processor.py` | Text-conditioned mask prediction | 03 |
| `Sam3Processor.add_geometric_prompt` | `sam3_image_processor.py` | Box-conditioned mask prediction (normalized cx,cy,w,h) | 03 |
| `build_sam3_image_model` | `model_builder.py:560` | Two-step: factory + `Sam3Processor(model)` | 03 |

**Details:**
- There is **no `automatic_mask_generator.py`** for SAM3 — it is purely prompt-conditioned.
- Output lives in a mutable `state` dict, not a return value: keys `masks` (N,1,H,W bool tensor), `masks_logits` (N,1,H,W), `boxes` (N,4 XYXY tensor), `scores` (N,) tensor.
- N is masks surviving `confidence_threshold`.

---

## Architecture & Data Flow

### Current SAM2 Flow

```
ovo.py:60
  MaskGenerator.__init__
    └─> segment_utils.load_sam()
          └─> build_sam2() + SAM2AutomaticMaskGenerator(model, ...)

ovo.py:231 OVO._get_masks(image, frame_id)
  └─> MaskGenerator.get_masks(image, frame_id)      [mask_generator.py:81]
        ├─ [precomputed] _load_masks() from disk
        └─ [live] segment(image)                     [mask_generator.py:101]
              └─> self.mask_generator.generate(image) [line 113]  <── sole call site
                    └─> masks_update(masks, ...)      [segment_utils.py:173]
                    └─> mask2segmap(masks, image)     [segment_utils.py:12]
                          └─> returns (seg_map: H×W, binary_maps: N×H×W)
```

### SAM3 Feature Encoder Flow (Existing, Separate)

```
SAM3FusionAdapter.compute_and_update(image, binary_maps, ...)
  └─> SAM3Generator.extract_sam3(image, binary_maps)
        └─> segment_utils.segmap2segimg(binary_maps, image, ...)  [crop/resize]
        └─> encode_image(cropped_tensor)
              └─> vit / neck forward pass
                    └─> adaptive_avg_pool2d + L2 normalize
                          └─> returns (N, embed_dim) tensor
```

### Proposed Wrapper Insertion Point

```
MaskGenerator.load_mask_generator [mask_generator.py:39-53]
  ├─ sam_version == ""   → SAM1 (existing)
  ├─ sam_version != ""
  │    └─ sam_version != "3" → SAM2AutomaticMaskGenerator (existing)
  └─ sam_version == "3"  → SAM3SegmenterAdapter  [NEW]
         └─ wraps Sam3Processor, exposes stateless .generate(image)

MaskGenerator.segment [line 113]
  └─> self.mask_generator.generate(image)
        ├─ SAM2: returns List[Dict] directly
        └─ SAM3: internally manages state, returns same List[Dict] schema
```

---

## Critical Code References

### Must Understand

- `mask_generator.py:101-120` (`segment`) — sole `.generate(image)` call site; this is the exact point where the output schema contract is enforced
- `mask_generator.py:39-53` (`load_mask_generator`) — primary extension point; add `elif sam_version == "3":` branch here
- `segment_utils.py:173` (`masks_update`) — consumes `List[Dict]` with `segmentation`, `predicted_iou`, `stability_score`; SAM3 adapter must produce these keys
- `segment_utils.py:12` (`mask2segmap`) — consumes same `List[Dict]`; second consumer of the schema
- `thirdParty/sam3/sam3/model/sam3_image_processor.py:14` (`Sam3Processor`) — the stateful SAM3 session API to be wrapped
- `mask_generator.py:44` — device access chain `self.mask_generator.predictor.model` (SAM2-specific; will break for SAM3)

### Should Review

- `segment_utils.py:269-309` (`load_sam`) — factory to extend or bypass for SAM3
- `ovo/entities/sam3_generator.py:25-178` (`SAM3Generator`) — reference for how SAM3 model loading works; `build_sam3_image_model` call at lines 64-81
- `fusion_encoders.py:150-233` (`SAM3FusionAdapter`) — reference implementation showing the adapter pattern already established in OVO
- `ovo.py:149, 162` — `mask_generator.cpu()` / `.cuda()` calls that eventually hit the fragile device chain

### Nice to Know

- `mask_generator.py:55-79` — `to / cpu / cuda` implementations; guard on `if self.mask_generator:` is already present
- `sam3_generator.py:50-53` — embed dim constants (`1024` vs `256`) for the encoder role
- `mask_generator.py:15-16` — `multi_crop` config key stored but never used (dead code)
- `mask_generator.py:153-168`, `170-195` — disk save/load format (`.npy` files); precomputed masks unaffected by backend swap

---

## Patterns & Conventions

- **Adapter pattern via ABC**: `FusionEncoderAdapter` at `fusion_encoders.py:8-37` defines abstract methods and is extended by `PEFusionAdapter`, `SAM3FusionAdapter`, `DINOFusionAdapter`. New adapters are low-friction.
- **Config-keyed dispatch**: `load_sam()` already dispatches on `config["sam_version"]`; `load_mask_generator()` mirrors this. The SAM3 segmenter branch should follow the same `elif sam_version == "3":` idiom.
- **Stateless external interface**: Callers (`MaskGenerator.segment`, downstream) always see a stateless `.generate(image)` returning `List[Dict]`. Session management and prompt injection must be internal to the SAM3 adapter.
- **Lazy model loading with None guard**: `precompute()` handles `None` model; `segment()` does not — any new adapter must respect this asymmetry or fix it.
- **Output schema contract**: The `List[Dict]` with keys `segmentation (H,W bool ndarray)`, `predicted_iou (float)`, `stability_score (float)` is the hard contract between the segmenter and `masks_update` / `mask2segmap`. All other keys (`bbox`, `area`, `point_coords`, `crop_box`) exist in SAM2 output but are not consumed by OVO today.
- **Separate configs for separate roles**: `config["sam"]` → segmenter; `config["sam3"]` → feature encoder. This separation must be preserved in the wrapper.

---

## Dependencies & Integration Points

- **Upstream**: `ovo.py` (`OVO._get_masks`, `detect_and_track_objects`) feeds RGB uint8 numpy arrays into `MaskGenerator.get_masks`.
- **Downstream**: `masks_update` and `mask2segmap` in `segment_utils.py` consume the raw `List[Dict]` from `.generate()`. The `(seg_map, binary_maps)` tensors they produce are then consumed by `SAM3FusionAdapter` (and other fusion adapters) in the feature-encoding pipeline.
- **External — SAM2**: `thirdParty/segment-anything-2/sam2/` — `build_sam2`, `SAM2AutomaticMaskGenerator`.
- **External — SAM3 segmenter**: `thirdParty/sam3/sam3/model/sam3_image_processor.py` (`Sam3Processor`), `thirdParty/sam3/sam3/model_builder.py` (`build_sam3_image_model`).
- **External — SAM3 encoder**: Same `build_sam3_image_model`; `sys.path` manipulation at import time in `sam3_generator.py` — path must exist or import fails immediately.
- **`Instance3D`**: `SAM3FusionAdapter` calls `objects[ins_id].update_sam3(...)` — if a SAM2 fusion adapter is added, `Instance3D` must have a corresponding `update_sam2` / `to_update_sam2` hook. Not yet verified.

---

## Gaps & Open Questions

- [ ] `thirdParty/sam3/sam3/model_builder.py` full API — `build_sam3_image_model` signature and what `load_from_HF` / `checkpoint_path` actually require for the segmentation model (vs. encoder-only loading)
- [ ] `ovo/entities/instance3d.py` — whether `update_sam3` / `to_update_sam3` hooks exist; whether analogous `update_sam2` hooks are needed for a SAM2 fusion adapter (noted in findings but not investigated)
- [ ] `Sam3Processor` default text prompt behavior — whether `"object"` as a generic prompt produces recall comparable to SAM2's point-grid automatic mode, and at what confidence threshold
- [ ] `Sam3Processor.add_geometric_prompt` — whether box prompts (as an alternative to text) could produce denser coverage; not analyzed
- [ ] `segment_utils.masks_update` — exact fields accessed beyond `segmentation`, `predicted_iou`, `stability_score`; any additional required keys would widen the translation surface
- [ ] `segment_utils.mask2segmap` — whether it accesses any fields beyond `segmentation`; findings mention it consumes the mask list but don't enumerate all field accesses
- [ ] SAM3 warm-up calls — `load_mask_generator` issues two dummy `.generate()` calls (lines 52-53) for warm-up; SAM3's stateful session may require a different warm-up sequence
- [ ] Device management for `Sam3Processor` — `self.mask_generator.predictor.model` at `mask_generator.py:44` is SAM2-specific; the equivalent access path in SAM3 is unknown

---

## Implementation Hints

- The only code that must change for callers to remain untouched is inside `MaskGenerator.load_mask_generator` (add branch) and whatever new adapter class is introduced for SAM3. `segment()`, `get_masks()`, and everything above them in `ovo.py` should not need modification.
- The SAM3 adapter's `.generate(image)` must return `List[Dict]` with at minimum `segmentation` (bool HW numpy), `predicted_iou` (float), and `stability_score` (float). SAM3's `scores` tensor maps naturally to `predicted_iou`; `stability_score` has no SAM3 equivalent and must be stubbed (e.g., to `1.0` or mirrored from `scores`).
- SAM3 `masks` are `(N, 1, H, W)` bool tensors; the wrapper must squeeze dim 1 and convert to numpy to match the `segmentation` field expectation.
- SAM3 output bbox is XYXY pixel coords; SAM2 uses XYWH float list. OVO does not appear to consume `bbox` downstream, but the conversion should still be included for correctness.
- The stateful `state = processor.set_image(image); state = processor.set_text_prompt("object", state)` pattern must be encapsulated in the adapter's `.generate()` — a new state dict per call, discarded after.
- `self.mask_generator.predictor.model` in `mask_generator.py:44` (device management) is fragile. Either version-branch it alongside `load_mask_generator`, or introduce an abstract `get_model()` helper on the adapter.
- `SAM3Generator` already calls `build_sam3_image_model` for encoder use — the same factory is used for segmentation. The checkpoint path and component loading differ (full model vs. ViT-only extract), so they should remain separate instances even if the factory is shared.

---

## Appendix: Source Findings

### From: `01_sam2_ovo_integration.md`
- Analyzed: `ovo/entities/ovo.py`, `ovo/utils/segment_utils.py`
- Key contribution: Established the complete SAM2 integration path — config keys, factory function, sole call site (`OVO._get_masks`), output fields consumed, and the critical architectural note that SAM3 is a separate feature encoder (not a segmenter) in the current codebase.

### From: `02_sam3_generator.md`
- Analyzed: `ovo/entities/sam3_generator.py`, `ovo/entities/fusion_encoders.py`
- Key contribution: Documented `SAM3Generator` as an encoder-only component (no `generate()` method), its embed dimensions, normalization convention, and the `SAM3FusionAdapter` pattern that plugs it into fusion — providing the reference adapter pattern for any new wrapper.

### From: `03_thirdparty_apis.md`
- Analyzed: `thirdParty/segment-anything-2/sam2/automatic_mask_generator.py`, `thirdParty/sam3/sam3/model/sam3_image_processor.py`, `thirdParty/sam3/sam3/model_builder.py`
- Key contribution: Confirmed SAM3 has no automatic mask generator; documented `Sam3Processor`'s stateful session API and output schema in full; produced the side-by-side API incompatibility table that defines the exact translation surface for the wrapper.

### From: `04_mask_generator.md`
- Analyzed: `ovo/entities/mask_generator.py`
- Key contribution: Mapped the internal structure of `MaskGenerator` — the exact extension point (`load_mask_generator:39-53`), the sole live `.generate()` call (line 113), the fragile device access chain (`predictor.model`), the `None`-model gap in `segment()`, and disk I/O format for precomputed masks.
