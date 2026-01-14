from typing import Dict, List, Union, Optional
import torch
import torch.nn.functional as F
import sys
import os
import torchvision.transforms as T
import numpy as np

# Add perception_models to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
pm_path = os.path.join(project_root, "thirdParty/perception_models")
if pm_path not in sys.path:
    sys.path.append(pm_path)

from core.vision_encoder import pe
from core.vision_encoder import transforms as pe_transforms
from ..utils import segment_utils

class PEGenerator:
    def __init__(self, config: Dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model_card = config.get("model_card", "PE-Core-L14-336")

        # Detect if this is a vision-only model (PE-Spatial, PE-Lang)
        self.is_vision_only = "Spatial" in self.model_card or "Lang" in self.model_card

        if self.is_vision_only:
            # PE-Spatial/PE-Lang are vision-only encoders, use VisionTransformer directly
            print(f"Loading vision-only model: {self.model_card}")
            self.model = pe.VisionTransformer.from_config(self.model_card, pretrained=True)
        else:
            # PE-Core models are full CLIP models with text encoder
            self.model = pe.CLIP.from_config(self.model_card, pretrained=True)

        # Auto-detect mask_res from model or use config override
        self.mask_res = config.get("mask_res", self.model.image_size)
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer (only for CLIP models with text encoder)
        if not self.is_vision_only:
            self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)
        else:
            self.tokenizer = None

        # Dimensions - handle both CLIP and VisionTransformer
        if self.is_vision_only:
            # VisionTransformer: width is the embedding dimension
            self.embed_dim = self.model.width
        elif hasattr(self.model, 'visual') and hasattr(self.model.visual, 'output_dim'):
            self.embed_dim = self.model.visual.output_dim
        else:
            raise ValueError("Cannot determine embedding dimension from model")

        if config.get("use_half", False):
            self.model.half()
            
        # Transform for Tensor inputs (B, 3, H, W)
        # PE expects: Resize, Normalize([0.5]*3, [0.5]*3)
        self.resize = T.Resize((self.model.image_size, self.model.image_size), interpolation=T.InterpolationMode.BICUBIC)
        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    @property
    def get_pe_dim(self) -> int:
        return self.embed_dim
            
    def to(self, device: str) -> None:
        if "cuda" in device:
            return self.cuda()
        else:
            return self.cpu()

    def cpu(self) -> None:
        self.device = "cpu"
        self.model.cpu()

    def cuda(self) -> None:
        self.device = "cuda"
        self.model.cuda()

    @torch.no_grad()
    def encode_image(self, input: torch.Tensor) -> torch.Tensor:
        """ Compute PE descriptor of an RGB image
        Args:
            - input (torch.Tensor): RGB image as tensor with shape (B, 3, H, W) or (3, H, W) in range [0, 1]
        Return:
            - pe_descriptor (torch.Tensor): as tensor with shape (B, self.embed_dim)
        """
        if input.ndim == 3:
            input = input.unsqueeze(0)

        # Resize
        processed_input = self.resize(input)
        # Normalize
        processed_input = self.normalize(processed_input)

        if self.config.get("use_half", False):
            processed_input = processed_input.half()

        if self.is_vision_only:
            # VisionTransformer: direct forward call
            features = self.model(processed_input)
            # PE-Spatial/Lang models with pool_type="none" return (B, num_patches, width)
            # We need to pool to get (B, embed_dim)
            if features.ndim == 3:
                # Use CLS token if available (first token), otherwise mean pool
                if self.model.use_cls_token:
                    features = features[:, 0, :]  # CLS token
                else:
                    features = features.mean(dim=1)  # Mean pool
            return features
        else:
            # CLIP: use encode_image method
            return self.model.encode_image(processed_input)

    @torch.no_grad()
    def extract_pe(self, image: Union[torch.Tensor, np.ndarray], binary_maps: torch.Tensor) -> torch.Tensor:
        """ Computes a PE vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor | np.ndarray): Full source RGB image with dimensions (3,H,W) and range 0-255.
            - binary_maps (torch.Tensor): A tensor of (N, H, W) containing N binary maps.
        Return:
            - pe_embeds: tensor with dim (N, self.embed_dim).        
        """
        
        if isinstance(image, np.ndarray):
             if image.ndim == 3 and image.shape[2] == 3:
                 image = image.transpose((2, 0, 1))
             image = torch.from_numpy(image).to(self.device)
             
        seg_images = segment_utils.segmap2segimg(binary_maps, image.squeeze(), False, out_l=self.mask_res)
        
        if len(seg_images) == 0:
            return torch.tensor([], device = self.device)
        
        if not isinstance(seg_images, torch.Tensor):
            seg_images = torch.from_numpy(seg_images).to(self.device)
        else:
            seg_images = seg_images.to(self.device)
            
        # Normalize to 0-1 as expect by encode_image
        seg_images = seg_images.float() / 255.0
        
        # encode_image expects (B, 3, H, W)
        # segmap2segimg returns (N, C, H, W)
        pe_embed = torch.nn.functional.normalize(self.encode_image(seg_images), p=2, dim=-1)
        
        if self.config.get("use_half", False):
            pe_embed = pe_embed.half()
            
        return pe_embed

    @torch.no_grad()
    def get_txt_embedding(self, text_list: List[str]) -> torch.Tensor:
        if self.is_vision_only:
            raise NotImplementedError(f"Text encoding not supported for vision-only model: {self.model_card}")
        # PE tokenizer
        tok_phrases = self.tokenizer(text_list).to(self.device)
        embeds = self.model.encode_text(tok_phrases)
        embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds

    @torch.no_grad()
    def get_embed_txt_similarity(self, ins_descriptors: torch.Tensor, txt_queries: List[str], templates: str | List[str] = ['{}']) -> torch.Tensor:
        if self.is_vision_only:
            raise NotImplementedError(f"Text similarity not supported for vision-only model: {self.model_card}")
        n_queries = len(txt_queries)
        txt_embeds = torch.zeros((n_queries, ins_descriptors.shape[1]), device = ins_descriptors.device)
        if isinstance(templates, str):
            templates = [templates]
        queries = [[template.format(query) for template in templates] for query in txt_queries]

        for j in range(n_queries):
            embed = self.get_txt_embedding(queries[j]).mean(0, keepdim=True).float()
            txt_embeds[j] = torch.nn.functional.normalize(embed, p=2, dim=-1)

        sim_map = (ins_descriptors @ txt_embeds.T) * self.model.logit_scale.exp()
        return sim_map
