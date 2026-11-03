"""Accept rule: given how similar two instances look and how much they overlap, are they the same?

    semantics:  cos_sim >= min_cos, otherwise rejected before measuring any overlap
    strong:     the overlap alone is enough            overlap > strong
    weak:       less overlap is enough if they look    overlap > weak  and  cos_sim > weak_min_cos
                very similar

    accept: {min_cos: 0.81, strong: 0.5, weak: 0.2, weak_min_cos: 0.9}
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AcceptCfg:
    min_cos: float = 0.81
    strong: float = 0.5
    weak: float = 0.2
    weak_min_cos: float = 0.9


def similar_enough(cos_sim: float, cfg: AcceptCfg) -> bool:
    """Semantic check, done before measuring overlap (overlap can be expensive)."""
    return cos_sim >= cfg.min_cos


def accept(overlap: float, cos_sim: float, cfg: AcceptCfg) -> Optional[str]:
    """Return the branch that accepts ("strong", "weak" or "both"), or None to reject.

    Assumes `similar_enough` already passed.
    """
    strong = overlap > cfg.strong
    weak = cos_sim > cfg.weak_min_cos and overlap > cfg.weak
    if strong and weak:
        return "both"
    if strong:
        return "strong"
    if weak:
        return "weak"
    return None
