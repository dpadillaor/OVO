from typing import Dict, List, Union, Optional
import torch
import torch.nn.functional as F
import sys
import os
import torchvision.transforms as T

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
        self.mask_res = config.get("mask_res", 336) 

        # Load PE model
        self.model = pe.CLIP.from_config(self.model_card, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)
        
        # Dimensions
        if hasattr(self.model.visual, 'output_dim'):
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
            
        return self.model.encode_image(processed_input)

    @torch.no_grad()
    def extract_pe(self, image: torch.Tensor, binary_maps: torch.Tensor) -> torch.Tensor:
        """ Computes a PE vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor): Full source RGB image with dimensions (3,H,W) and range 0-255.
            - binary_maps (torch.Tensor): A tensor of (N, H, W) containing N binary maps.
        Return:
            - pe_embeds: tensor with dim (N, self.embed_dim).        
        """
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
        # PE tokenizer
        tok_phrases = self.tokenizer(text_list).to(self.device)
        embeds = self.model.encode_text(tok_phrases)
        embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds

    @torch.no_grad()
    def get_embed_txt_similarity(self, ins_descriptors: torch.Tensor, txt_queries: List[str], templates: str | List[str] = ['{}']) -> torch.Tensor:
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
