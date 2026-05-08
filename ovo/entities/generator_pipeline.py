from typing import Optional
from .clip_generator import CLIPGenerator
from .mask_generator import MaskGenerator
from .pe_generator import PEGenerator
from .semantic_config import SemanticConfig


class GeneratorPipeline:
    """Owns and manages all feature generators. CLIP always active; others optional by config."""

    def __init__(self, semantic_config: SemanticConfig, device: str, scene_name: Optional[str] = None, eval: bool = False):
        self.clip = CLIPGenerator(semantic_config.clip_config, device=device)
        self.pe: Optional[PEGenerator] = (
            PEGenerator(semantic_config.pe_config, device=device) if semantic_config.pe_config is not None else None
        )
        self.sam3 = self._init_sam3(semantic_config.sam3_config, device)
        self.mask = MaskGenerator(semantic_config.sam_config, scene_name, device=device) if not eval else None

    @staticmethod
    def _init_sam3(sam3_config, device: str):
        if sam3_config is None:
            return None
        try:
            from .sam3_generator import SAM3Generator
            return SAM3Generator(sam3_config, device=device)
        except ImportError as e:
            raise ImportError(
                f"Failed to import SAM3Generator. SAM3 dependencies may not be installed: {e}"
            ) from e

    def cpu(self) -> None:
        self.clip.cpu()
        if self.pe is not None:
            self.pe.cpu()
        if self.sam3 is not None:
            self.sam3.cpu()
        if self.mask is not None:
            self.mask.cpu()

    def cuda(self) -> None:
        self.clip.cuda()
        if self.pe is not None:
            self.pe.cuda()
        if self.sam3 is not None:
            self.sam3.cuda()
        if self.mask is not None:
            self.mask.cuda()

    def to(self, device: str) -> None:
        if "cuda" in device:
            self.cuda()
        else:
            self.cpu()
