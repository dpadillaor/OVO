"""Contract shared by every image encoder (CLIP, PE, SAM3).

An encoder turns an RGB image + per-instance masks into one descriptor per instance
(`extract`), and, when it has a text tower, scores those descriptors against text queries
(`get_embed_txt_similarity`) for the final open-vocabulary labels. How the crop is built
before encoding is a separate concern (see `crops.py`), so the same encoder can run on a
masked segment, a bbox with context, or the whole frame without changing this contract.
"""

from abc import ABC, abstractmethod
from typing import List

import torch


class ImageEncoder(ABC):
    def to(self, device: str) -> None:
        """Move the encoder to `device`, dispatching to cpu()/cuda()."""
        return self.cuda() if "cuda" in device else self.cpu()

    @abstractmethod
    def cpu(self) -> None: ...

    @abstractmethod
    def cuda(self) -> None: ...

    @abstractmethod
    def encode_image(self, input: torch.Tensor) -> torch.Tensor:
        """Encode a batch of RGB crops (B,3,H,W in [0,1]) into normalized descriptors."""
        ...

    def get_embed_txt_similarity(self, ins_descriptors: torch.Tensor, txt_queries: List[str], templates="{}") -> torch.Tensor:
        """Similarity between instance descriptors and text queries (for final labels).

        Only encoders with a text tower (CLIP, PE-Core) implement it; a fusion-only encoder
        (SAM3, vision-only PE) leaves this raising, since labeling never runs through it.
        """
        raise NotImplementedError(f"{type(self).__name__} has no text tower for labeling")

    # Each encoder exposes its own descriptor extractor (extract_clip / extract_pe /
    # extract_sam3): the crop and return shape differ per model, so the name stays specific
    # while the shared contract above (device, encode, text similarity) is enforced here.
