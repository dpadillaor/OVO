"""Instance fusion: decide which instances are the same object and merge them."""

from .config import FusionCfg
from .log import FusionLog
from .pipeline import FusionPipeline, FusionReport
from .timing import TimingLog

__all__ = ["FusionCfg", "FusionLog", "FusionPipeline", "FusionReport", "TimingLog"]
