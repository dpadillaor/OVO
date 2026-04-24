# config["semantic"] usage — OVO module

## Direct accesses in ovo.py

| Key path | file:line | Usage |
|---|---|---|
| `config["sam"]["multi_crop"]` | ovo.py:31 | Set based on clip embed_type; controls SAM multi-crop behavior |
| `config["clip"]["embed_type"]` | ovo.py:31 | Read to determine multi_crop flag (="vanilla" check) |
| `config["clip"]` | ovo.py:45 | Passed entire to CLIPGenerator.__init__ |
| `config["clip"].get("k_top_views", 0)` | ovo.py:38 | Read to set self.n_top_views; passed to Instance3D.n_top_kf |
| `config["clip"]["mask_res"]` | ovo.py:43 | Set from config["sam"]["mask_res"] if missing |
| `config["sam"]["mask_res"]` | ovo.py:43 | Read and potentially copied to config["clip"]["mask_res"] |
| `config["sam"]` | ovo.py:60 | Passed entire to MaskGenerator.__init__ |
| `config["pe"]` | ovo.py:46 | Passed entire to PEGenerator.__init__ if "pe" in config |
| `config["sam3"]` | ovo.py:52 | Passed entire to SAM3Generator.__init__ if "sam3" in config |
| `config.get("debug_info", False)` | ovo.py:35 | Stored as self.debug_info; controls debug logging |
| `config.get("fusion_method", "CLIP")` | ovo.py:36 | Stored as self.fusion_method; determines fusion encoder type |
| `config.get("th_centroid", 1.5)` | ovo.py:79 | Stored as self.th_centroid; fusion strategy threshold |
| `config.get("th_cossim", 0.81)` | ovo.py:80 | Stored as self.th_cossim; fusion strategy similarity threshold |
| `config.get("th_points", 0.1)` | ovo.py:81 | Stored as self.th_points; fusion strategy point overlap threshold |
| `config.get("verbose", True)` | ovo.py:90 | Controls PrettyPrinter output of full config |
| `config.get("log", False)` | ovo.py:173 (profil decorator) | Controls timing profiling in @profil decorator |
| `config.get("match_distance_th")` | ovo.py:298 | Passed to geometry_utils.match_3d_points_to_2d_pixels |
| `config.get("track_th")` | ovo.py:307 | Passed to _track_objects method |
| `config.get("depth_filter", False)` | ovo.py:295 | Conditional depth filtering in _match_and_track_instances |
| `config.get("debug_info", False)` | ovo.py:313 | Conditional debug info storage (ins_maps) |
| `config.get("kf_queue_delay", 0)` | ovo.py:409 | Minimum queue size before computing semantic info |
| `config.get("log", False)` | ovo.py:230, 439 | Logging stats (frame_id, n_obj, timing) |
| `config.get("return_all_clips", False)` | ovo.py:602 | Passed to clip_generator.extract_clip() |

## Config subdict passthroughs to classes

| Config section | Recipient class | file | Notes |
|---|---|---|---|
| `config["clip"]` | CLIPGenerator | clip_generator.py:12 | Full config dict passed; accesses: embed_type, mask_res, model_card, weights_predictor_path, use_half, w_masked, w_global, logit_scale, logit_bias |
| `config["sam"]` | MaskGenerator | mask_generator.py:16 | Full config dict passed; accesses: precomputed, masks_base_path, nms_iou_th, nms_score_th, nms_inner_th, multi_crop, sam_version, precompute |
| `config["pe"]` | PEGenerator | pe_generator.py:21 | Full config dict passed if present; accesses: model_card, mask_res, use_half |
| `config["sam3"]` | SAM3Generator | sam3_generator.py:30 | Full config dict passed if "sam3" in config; accesses: components, load_from_hf, checkpoint_path, image_size, use_half |
| `config` (entire) | create_fusion_strategy() | fusion.py:146 | Factory function that reads fusion_method, creates SemanticGeometricFusion or GeometricOnlyFusion with config |

## Fusion strategy initialization (fusion.py)

Fusion strategies receive full config dict and extract:

| Key | Accessed in | Usage |
|---|---|---|
| `config.get("fusion_method", "clip")` | create_fusion_strategy:159 | Determines strategy class (SemanticGeometricFusion vs GeometricOnlyFusion) and feature_attr ("clip_feature", "dino_feature", "pe_feature", "sam3_feature") |
| `config.get("th_centroid", 1.5)` | SemanticGeometricFusion.__init__:63, GeometricOnlyFusion.__init__:114 | Instance centroid distance threshold for fusion |
| `config.get("th_cossim", 0.81)` | SemanticGeometricFusion.__init__:64 | Cosine similarity threshold for semantic fusion (GeometricOnly doesn't use) |
| `config.get("th_points", 0.1)` | SemanticGeometricFusion.__init__:65, GeometricOnlyFusion.__init__:115 | Point cloud overlap threshold |

## Summary of key flows

1. **Init (ovo.py:27-93)**:
   - `config["clip"]` → CLIPGenerator
   - `config["sam"]` → MaskGenerator
   - `config["pe"]` (optional) → PEGenerator
   - `config["sam3"]` (optional) → SAM3Generator
   - `config` → create_fusion_strategy() → SemanticGeometricFusion/GeometricOnlyFusion
   - Scalar threshold keys (th_centroid, th_cossim, th_points) stored locally in OVO

2. **Frame processing (ovo.py:185-320)**:
   - `config["match_distance_th"]` → geometry_utils.match_3d_points_to_2d_pixels()
   - `config["track_th"]` → _track_objects()
   - `config["depth_filter"]` → conditional depth_filter call

3. **Semantic computation (ovo.py:408-459)**:
   - `config["kf_queue_delay"]` controls queue drain threshold
   - `config["return_all_clips"]` → clip_generator.extract_clip()

4. **No direct fusion.py config usage from OVO context** - fusion.py creates strategies via factory using full config dict
