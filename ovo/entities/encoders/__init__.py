from .base import ImageEncoder
from .crops import CropStrategy, SegCrop, BBoxCrop, build_crop
from .clip import CLIPEncoder
from .pe import PEEncoder

# SAM3Encoder is intentionally NOT imported here: it pulls the SAM3 submodule at import
# time, which need not be present when SAM3 is unused. Import it lazily where needed:
#     from .encoders.sam3 import SAM3Encoder

__all__ = [
    "ImageEncoder",
    "CropStrategy", "SegCrop", "BBoxCrop", "build_crop",
    "CLIPEncoder", "PEEncoder",
]
