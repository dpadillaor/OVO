from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SemanticConfig:
    """Typed configuration for the OVO semantic pipeline."""
    # Tracking
    match_distance_th: float = 0.05
    track_th: float = 100
    # Fusion strategy
    fusion_method: str = "clip"
    th_centroid: float = 1.5
    th_cossim: float = 0.81
    th_points: float = 0.1
    n_top_views: int = 0
    # Runtime flags
    log: bool = False
    debug_info: bool = False
    depth_filter: bool = False
    kf_queue_delay: int = 0
    return_all_clips: bool = False
    verbose: bool = True
    # Generator sub-configs
    clip_config: Dict = field(default_factory=dict)
    sam_config: Dict = field(default_factory=dict)
    pe_config: Optional[Dict] = None
    sam3_config: Optional[Dict] = None

    @classmethod
    def from_config(cls, config: dict) -> "SemanticConfig":
        clip_config = dict(config.get("clip", {}))
        sam_config = dict(config.get("sam", {}))

        # Propagate mask_res from sam to clip if needed
        if "mask_res" in sam_config and "mask_res" not in clip_config:
            clip_config["mask_res"] = sam_config["mask_res"]

        sam_config["multi_crop"] = clip_config.get("embed_type") != "vanilla"

        return cls(
            match_distance_th=config.get("match_distance_th", 0.05),
            track_th=config.get("track_th", 100),
            fusion_method=config.get("fusion_method", "clip"),
            th_centroid=config.get("th_centroid", 1.5),
            th_cossim=config.get("th_cossim", 0.81),
            th_points=config.get("th_points", 0.1),
            n_top_views=clip_config.get("k_top_views", 0),
            log=config.get("log", False),
            debug_info=config.get("debug_info", False),
            depth_filter=config.get("depth_filter", False),
            kf_queue_delay=config.get("kf_queue_delay", 0),
            return_all_clips=config.get("return_all_clips", False),
            verbose=config.get("verbose", True),
            clip_config=clip_config,
            sam_config=sam_config,
            pe_config=config.get("pe"),
            sam3_config=config.get("sam3"),
        )
