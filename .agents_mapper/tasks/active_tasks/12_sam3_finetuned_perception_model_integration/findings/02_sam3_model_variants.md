# Research Findings: SAM3 Model Variants & Architecture

## Executive Summary

While the **848M ViT Backbone** (ViT-Large) is the single foundational vision encoder for SAM3, the repository implements several specialized model variants that utilize this backbone for different tasks (detection, tracking, interactivity). This document details these variants and their specific architectures.

---

## 1. The Unified Backbone
**Core Component:** `ViT` (Vision Transformer)
- **Parameters:** ~848M
- **Config:** Depth=32, Embed Dim=1024, Heads=16, Patch Size=14x14
- **Input:** 1008x1008 pixels
- **Role:** Serves as the shared "Trunk" for all downstream tasks.

---

## 2. Model Variants

### A. SAM3 Image (The "Detector")
**Class:** `Sam3Image`
**File:** `sam3/model/sam3_image.py`
**Primary Use:** Open-vocabulary object detection and segmentation in static images.
- **Architecture:** 
    - **Backbone:** Shared ViT + Text Encoder (CLIP-like).
    - **Neck:** Feature Pyramid (scales 4.0x to 0.5x).
    - **Decoder:** Transformer Decoder (DETR-style) that queries objects based on text/visual prompts.
    - **Head:** `UniversalSegmentationHead` for mask generation.
- **Key Feature:** Can "exhaustively segment" all instances of a concept.

### B. SAM3 Tracker (The "Video Model")
**Class:** `Sam3TrackerPredictor` / `Sam3VideoPredictorMultiGPU`
**File:** `sam3/model/sam3_tracking_predictor.py`
**Primary Use:** Tracking objects across video frames.
- **Architecture:**
    - Inherits significantly from **SAM 2**.
    - **Memory:** Uses a "Memory Encoder" and "Memory Bank" to store past frame features.
    - **Interaction:** Can refine tracks interactively.
- **Integration:** It shares the same vision backbone as the Image model but adds temporal components to handle consistency over time.

### C. SAM3 Interactive Predictor
**Class:** `SAM3InteractiveImagePredictor`
**File:** `sam3/model/sam1_task_predictor.py`
**Primary Use:** Classic SAM1-style interactive segmentation (points, boxes).
- **Architecture:**
    - Wraps the core components to accept geometric prompts.
    - Often used as a submodule within the larger `Sam3Image` model to refine detections or allow user correction.

---

## 3. Specialized Components

### "Dual Neck" Architecture
**Class:** `Sam3DualViTDetNeck`
**File:** `sam3/model/necks.py`
**Insight:** 
The system can maintain **two separate sets of neck weights** attached to the single frozen backbone:
1.  **SAM3 Neck:** Optimized for the new open-vocabulary tasks.
2.  **SAM2 Neck:** (Optional) A clone of the neck optimized for the original SAM2 interactive tasks.
This allows the model to switch between "SAM3 mode" and "SAM2 mode" without reloading the massive backbone.

### Vision-Language Fusion
**Class:** `SAM3VLBackbone`
**File:** `sam3/model/vl_combiner.py`
**Insight:**
The backbone isn't just visual; it tightly integrates text features early on.
- **Text Encoder:** A 1024-width transformer (similar to CLIP's text tower).
- **Fusion:** Text and Image features are combined before entering the heavy decoder layers.

---

## Implications for OVO Integration

1.  **Backbone Reusability:** If we integrate the ViT backbone for our "Perception Model," we are technically equipping OVO with the capability to run *any* of these tasks if we add the corresponding heads/necks.
2.  **Memory Overhead:** The "Dual Neck" and "Tracker" components add parameters. For a pure "Perception Model" (feature extraction), we should strictly instantiate the **Backbone + Primary Neck** and avoid initializing the Tracker or Dual Neck weights to save VRAM.
3.  **Future Proofing:** Documenting the Tracker architecture is valuable if OVO moves towards video-based consistency (Track-and-Map) in the future.
