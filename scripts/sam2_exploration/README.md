# SAM2 Parameter Exploration with Layer Analysis

Isolated sandbox for testing SAM2 parameters with layer-by-layer visualization.

## Output Structure

Results saved in `scripts/sam2_exploration/output/`:

```
scripts/sam2_exploration/output/
├── Replica_office0_sam16_iou0.8_score0.1_inner0.5/
│   ├── metadata.json               ← Config used
│   ├── 000000/
│   │   ├── capa_0.png              ← Layer 0 overlay
│   │   ├── capa_1.png              ← Layer 1 overlay
│   │   ├── capa_2.png              ← Layer 2 overlay
│   │   └── final.png               ← Final filtered
│   ├── 000001/
│   └── ...0000099/
├── Replica_office0_sam32_iou0.5_score0.1_inner0.5/
└── Replica_office0_sam8_iou0.9_score0.1_inner0.5/
```

Easy comparison: Different SAM configs organized by dataset/scene/config name.

## Usage

### Basic (OVO defaults)

```bash
cd scripts/sam2_exploration
python test_sam2.py --frames "0:100:1"
```

Generates 4 images per frame (capa_0, capa_1, capa_2, final) in:
```
scripts/sam2_exploration/output/Replica_office0_sam16_iou0.8_score0.1_inner0.5/
```

### Override Parameters

```bash
# Different point density
python test_sam2.py --points-per-side 32
python test_sam2.py --points-per-side 8

# Different NMS thresholds
python test_sam2.py --nms-iou-th 0.5
python test_sam2.py --nms-iou-th 0.9

# Select specific frames
python test_sam2.py --frames "0:20:5"   # 0, 5, 10, 15
python test_sam2.py --frames 0,5,10     # specific indices

# Verbose output
python test_sam2.py --verbose
```

### Quick Test Multiple Configs

```bash
# Test 1
python test_sam2.py --points-per-side 16 --frames 0:5

# Test 2
python test_sam2.py --points-per-side 32 --frames 0:5

# Test 3
python test_sam2.py --points-per-side 8 --frames 0:5

# All results in output/, easy to compare visually with viewer
```

## Visualization

### Web UI (Recommended)

Interactive comparison of multiple configs:

```bash
streamlit run scripts/sam2_exploration/viewer.py
```

Opens http://localhost:8501

**Features:**
- Enter dataset pattern in sidebar (e.g., 'Replica_office0')
- Select 2-3 configs to compare side-by-side
- Slider to navigate frames
- For each frame, see 4 images (capa_0, capa_1, capa_2, final) per config
- Shows "Not available" if frame missing in config
- Display config metadata
- Handles different frame counts per config

**Workflow:**
1. Run `streamlit run scripts/sam2_exploration/viewer.py`
2. Enter dataset pattern in sidebar (default: `Replica_office0`)
3. Select configs to compare (checkboxes)
4. Use slider to navigate frames
5. View 4-column grid per config

### File Browser (Simple)

```bash
ls scripts/sam2_exploration/output/
# Browse results with file manager or terminal
```

## Layers Explained

Each frame generates 4 images:

**capa_0.png** — Base grid (image-wide)
- `points_per_side × points_per_side` grid
- Detects large objects
- Fast, coarse

**capa_1.png** — Cropped layer
- Grid divided by 2 (8x8 for points_per_side=16)
- Detects medium objects
- Adds detail to large areas

**capa_2.png** — Fine detail layer
- Grid divided by 4 (4x4 for points_per_side=16)
- Detects small objects
- Maximum detail

**final.png** — All layers combined + NMS filtering
- All 3 layers merged
- Duplicate masks removed (NMS)
- Clean, final result

## Config File

Edit `config.yaml` to change:

```yaml
dataset:
  dataset_path: "data/input/Datasets/Replica/office0/results"
  frame_indices: null  # null=all, "0:100:10"=specific

sam:
  points_per_side: 16      # Main parameter (8, 16, 32, 64)
  nms_iou_th: 0.8         # Main filter (0.3-0.9)
  nms_score_th: 0.1       # Usually leave default
  nms_inner_th: 0.5       # For nested objects
  sam_version: "2.1"      # SAM2 version
  sam_encoder: "hiera_l"  # Model size
```

## Parameters Explained

### points_per_side
- **8** → Fast, sparse, big objects only
- **16** → OVO default, balanced
- **32** → Dense, more masks, slower
- **64** → Very dense, overkill

### nms_iou_th
- **0.3** → Strict, eliminates many overlaps
- **0.8** → Default, balanced
- **0.9** → Relaxed, keeps overlapping masks

### nms_inner_th
- Lower (0.2) → Protect small objects inside large ones
- Higher (0.8) → Eliminate nested masks

## Typical Workflow

1. Quick test single frame
   ```bash
   python test_sam2.py --frames 0 --verbose
   ```

2. Generate 100 frames
   ```bash
   python test_sam2.py --frames "0:100:1"
   ```

3. Compare with viewer
   ```bash
   streamlit run scripts/sam2_exploration/viewer.py
   ```

4. Try different config
   ```bash
   python test_sam2.py --points-per-side 32 --frames "0:100:1"
   ```

5. View both in viewer (auto-loads all configs matching pattern)

## OVO Defaults (for reference)

From `data/working/configs/ovo.yaml`:
```yaml
sam:
  points_per_side: 16
  sam_version: "2.1"
  sam_encoder: hiera_l
  nms_iou_th: 0.8
  stability_score_th: 0.95
```

## Tips

- Script prints GPU/CPU device at startup
- Use `watch` to monitor progress: `watch "ls scripts/sam2_exploration/output/Replica_office0_sam16*/  | wc -l"`
- Outputs saved frame-by-frame (recoverable if interrupted)
- Metadata JSON saved automatically with each run
- No OVO pipeline touched — fully isolated
