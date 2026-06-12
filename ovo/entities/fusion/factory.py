from typing import Dict, Any, List, Optional

from ...utils.cooccurrence_graph import CooccurrenceGraph
from .criteria import (
    Criterion,
    CooccurrenceCriterion,
    CentroidDistanceCriterion,
    AabbDistanceCriterion,
    CosSimilarityCriterion,
    PointOverlapCriterion,
    PointOverlapOldCriterion,
)
from .strategy import FusionStrategy


_DEFAULT_CHAINS: Dict[str, List[str]] = {
    "clip":  ["cooccurrence", "centroid", "cos_sim", "overlap"],
    "dino":  ["cooccurrence", "centroid", "cos_sim", "overlap"],
    "pe":    ["cooccurrence", "centroid", "cos_sim", "overlap"],
    "sam3":  ["cooccurrence", "centroid", "cos_sim", "overlap"],
}

_FEATURE_ATTRS: Dict[str, str] = {
    "clip": "clip_feature",
    "dino": "dino_feature",
    "pe":   "pe_feature",
    "sam3": "sam3_feature",
}


def _build_criterion(name: str, config: Dict[str, Any], feature_attr: Optional[str], cooccurrence_graph: CooccurrenceGraph) -> Criterion:
    if name == "cooccurrence":
        return CooccurrenceCriterion(cooccurrence_graph, config.get("cooccurrence_veto_threshold", 5))
    if name == "centroid":
        return CentroidDistanceCriterion(config.get("th_centroid", 1.5))
    if name == "aabb":
        return AabbDistanceCriterion(config.get("th_aabb", 0.3))
    if name == "cos_sim":
        if feature_attr is None:
            raise ValueError("cos_sim criterion requires a semantic fusion_method (not geometric)")
        return CosSimilarityCriterion(feature_attr, config.get("th_cossim", 0.81))
    if name == "overlap":
        return PointOverlapCriterion(config.get("th_points", 0.1))
    if name == "overlap_old":
        return PointOverlapOldCriterion(config.get("th_points", 0.1))
    raise ValueError(f"Unknown criterion: {name!r}. Valid: cooccurrence, centroid, aabb, cos_sim, overlap, overlap_old")


def create_fusion_strategy(config: Dict[str, Any], cooccurrence_graph: CooccurrenceGraph) -> FusionStrategy:
    """
    Build a FusionStrategy from config.

    fusion_method selects the default criterion chain and feature attribute.
    Override the chain by setting fusion_criteria (list of criterion names) in config.
    """
    fusion_method = config.get("fusion_method", "clip").lower()
    if fusion_method not in _DEFAULT_CHAINS:
        raise ValueError(f"Unknown fusion_method: {fusion_method!r}. Valid: {list(_DEFAULT_CHAINS)}")

    chain = config.get("fusion_criteria", _DEFAULT_CHAINS[fusion_method])
    feature_attr = _FEATURE_ATTRS.get(fusion_method)

    criteria = [_build_criterion(name, config, feature_attr, cooccurrence_graph) for name in chain]
    return FusionStrategy(criteria)
