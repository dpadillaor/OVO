"""Instance fusion: decide which instances are the same object and merge them."""

from .config import FusionCfg
from .pipeline import FusionPipeline, FusionReport

__all__ = ["FusionCfg", "FusionPipeline", "FusionReport"]
