# Research Findings: SAM3 Finetuned Perception Model

## Executive Summary

SAM3 repository is **complete** at `thirdParty/sam3/`. The "Perception Model" in SAM3 is a **ViT backbone (848M params)** with a feature pyramid neck. Finetuned checkpoint loading is fully supported.

---

## SAM3 Architecture Overview

### Vision Encoder (ViT Backbone)
**File**: `thirdParty/sam3/sam3/model/vitdet.py`

| Parameter | Value |
|-----------|-------|
| Input Size | 1008x1008 px |
| Patch Size | 14x14 |
| Embed Dim | **1024** |
| Depth | 32 blocks |
| Attention Heads | 16 |
| Parameters | 848M |
| Normalization | mean/std=[0.5, 0.5, 0.5] |

### Feature Pyramid Neck
**File**: `thirdParty/sam3/sam3/model/necks.py`

Converts ViT output to 4-level pyramid (all 256-dim):
- Scale 4.0x, 2.0x, 1.0x, 0.5x

### Visual-Language Fusion
**File**: `thirdParty/sam3/sam3/model/vl_combiner.py`

`SAM3VLBackbone` combines vision encoder with text encoder.

---

## How Finetuned Models Are Loaded

### Entry Point
**File**: `thirdParty/sam3/sam3/model_builder.py`

```python
from sam3.model_builder import build_sam3_image_model

model = build_sam3_image_model(
    checkpoint_path="/path/to/finetuned_sam3.pt",
    load_from_HF=False
)
```

### Default Checkpoint Source
- **HuggingFace**: `facebook/sam3` → `sam3.pt`

### Checkpoint Format
Keys are prefixed with `detector.`:
```python
checkpoint = torch.load("sam3.pt")
# Extract detector weights
sam3_weights = {
    k.replace("detector.", ""): v
    for k, v in checkpoint.items()
    if "detector" in k
}
```

---

## Extracting Just the Vision Encoder

```python
import torch

checkpoint = torch.load("sam3_finetuned.pt")
model_state = checkpoint.get("model", checkpoint)

# Extract ViT weights only
vit_weights = {
    k.replace("detector.backbone.vision_backbone.trunk.", ""): v
    for k, v in model_state.items()
    if "detector.backbone.vision_backbone.trunk." in k
}

torch.save(vit_weights, "finetuned_vit_weights.pt")
```

---

## Comparison: SAM3 vs Current OVO PE

| Aspect | OVO PE (Current) | SAM3 ViT |
|--------|------------------|----------|
| Model | PE-Core-L14-336 | ViT-32 |
| Input Size | 336px | 1008px |
| Output Dim | **512** | **1024** |
| Normalization | [0.5, 0.5, 0.5] | [0.5, 0.5, 0.5] |
| Text Encoder | Integrated | Integrated |

### Key Difference
Output dimension changes from **512 → 1024**. This affects fusion/merging logic.

---

## Key Files Reference

| Purpose | File Path |
|---------|-----------|
| ViT Backbone | `sam3/model/vitdet.py` |
| Feature Pyramid | `sam3/model/necks.py` |
| VL Combiner | `sam3/model/vl_combiner.py` |
| Model Builder | `sam3/model_builder.py` |
| Image Model | `sam3/model/sam3_image.py` |

---

## Integration Considerations

1. **Dimension Mismatch**: 1024 vs 512 output dims - fusion logic needs update
2. **Modular Extraction**: ViT can be extracted separately from full model
3. **Model Size**: 848M params - may need optimization for real-time use
4. **Preprocessing**: Same normalization as current PE (compatible)
5. **Text Encoder**: Tightly integrated but can be decoupled

---

## Recommended Integration Approach

**Option A: Full SAM3 Backbone**
- Use `build_sam3_image_model()` with custom checkpoint
- Benefits: Full feature pyramid + text encoder
- Drawback: Large model, may be overkill for OVO

**Option B: Extract ViT Only**
- Extract `detector.backbone.vision_backbone.trunk.*` weights
- Create wrapper similar to `PEGenerator`
- Benefits: Lighter, focused on vision features
- Drawback: Loses multi-scale pyramid

**Option C: Hybrid Approach**
- Extract ViT + Neck (pyramid features)
- Skip text encoder (OVO has its own)
- Benefits: Multi-scale features without redundant text encoding
