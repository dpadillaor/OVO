from typing import Dict, List, Union, Optional, Any
import torch
import torch.nn.functional as F
import sys
import os
import torchvision.transforms as T

# Add SAM3 to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sam3_path = os.path.join(project_root, "thirdParty/sam3")

if not os.path.exists(sam3_path):
    raise FileNotFoundError(
        f"SAM3 repository not found at {sam3_path}. "
        "Make sure the thirdParty submodules are correctly initialized and the path exists."
    )

if sam3_path not in sys.path:
    sys.path.append(sam3_path)

from sam3.model_builder import build_sam3_image_model
from ..utils import segment_utils

class SAM3Generator:
    """
    SAM3 Feature Generator for OVO.
    Extracts semantic features using the finetuned SAM3 Vision Transformer.
    """
    def __init__(self, config: Dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.components = config.get("components", "vit_only")
        self.load_from_hf = config.get("load_from_hf", False)
        self.checkpoint_path = config.get("checkpoint_path", None)
        self.image_size = config.get("image_size", 1008)
        self.use_half = config.get("use_half", False)

        valid_components = ["vit_only", "vit_neck", "full"]
        if self.components not in valid_components:
            raise ValueError(f"Invalid components: {self.components}. Must be one of {valid_components}")

        if not self.checkpoint_path and not self.load_from_hf:
            raise ValueError("Either checkpoint_path must be provided or load_from_hf must be True")

        # Load the model using SAM3 builder
        self._load_model()

        # Set embedding dimension based on architecture
        if self.components == "vit_only":
            self.embed_dim = 1024
        else:
            self.embed_dim = 256

        self.to(self.device)
        
        if self.use_half and self.device != "cpu":
            self.model_to_half()

        # Normalization (SAM3 uses [0.5, 0.5, 0.5] mean/std)
        self.resize = T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC)
        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def _load_model(self):
        """Loads the SAM3 model and extracts required components."""
        model = build_sam3_image_model(
            checkpoint_path=self.checkpoint_path,
            load_from_HF=self.load_from_hf
        )
        
        self.vit = model.backbone.vision_backbone.trunk
        
        if self.components in ["vit_neck", "full"]:
            self.neck = model.backbone.vision_backbone.neck
        else:
            self.neck = None
            
        if self.components == "full":
            self.text_encoder = model.backbone.text_backbone
        else:
            self.text_encoder = None

    def model_to_half(self):
        """Converts model components to half precision."""
        if self.vit is not None:
            self.vit.half()
        if self.neck is not None:
            self.neck.half()
        if self.text_encoder is not None:
            self.text_encoder.half()

    def to(self, device: str) -> None:
        """Moves model components to the specified device."""
        self.device = device
        if self.vit is not None:
            self.vit.to(device)
        if self.neck is not None:
            self.neck.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)

    def cpu(self) -> None:
        self.to("cpu")

    def cuda(self) -> None:
        self.to("cuda")

    @torch.no_grad()
    def encode_image(self, input: torch.Tensor) -> torch.Tensor:
        """ Compute SAM3 descriptor of an RGB image.
        Args:
            - input (torch.Tensor): RGB image as tensor with shape (B, 3, H, W) or (3, H, W) in range [0, 1]
        Return:
            - sam3_descriptor (torch.Tensor): as tensor with shape (B, self.embed_dim)
        """
        if input.ndim == 3:
            input = input.unsqueeze(0)
            
        input = input.to(self.device)
        processed_input = self.normalize(self.resize(input))
        
        if self.use_half and self.device != "cpu":
            processed_input = processed_input.half()
            
        if self.components == "vit_only":
            features = self.vit(processed_input)
            # Handle potential dict return from ViT trunk
            if isinstance(features, dict):
                last_key = list(features.keys())[-1]
                features = features[last_key]
            # Handle potential list return from ViT trunk (multi-scale features)
            elif isinstance(features, list):
                features = features[-1]  # Take the last (highest level) feature

            if features.ndim == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            return F.normalize(features, p=2, dim=-1)
            
        elif self.components in ["vit_neck", "full"]:
            trunk_features = self.vit(processed_input)
            neck_features = self.neck(trunk_features)
            
            # Neck returns list of multi-scale features, we take the highest level
            top_feature = neck_features[-1]
            if top_feature.ndim == 4:
                top_feature = F.adaptive_avg_pool2d(top_feature, (1, 1)).flatten(1)
            return F.normalize(top_feature, p=2, dim=-1)

    @torch.no_grad()
    def extract_sam3(self, image: torch.Tensor, binary_maps: torch.Tensor) -> torch.Tensor:
        """ Computes a SAM3 vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor): Full source RGB image with dimensions (3,H,W) and range 0-255.
            - binary_maps (torch.Tensor): A tensor of (N, H, W) containing N binary maps.
        Return:
            - sam3_embeds: tensor with dim (N, self.embed_dim).
        """
        if binary_maps.shape[0] == 0:
            return torch.zeros((0, self.embed_dim), device=self.device)

        # Convert numpy array to tensor if needed (image comes as np.ndarray with shape (H,W,3))
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose((2,0,1))).to(self.device)

        seg_images = segment_utils.segmap2segimg(binary_maps, image.squeeze(), False, out_l=self.image_size)
        
        if not isinstance(seg_images, torch.Tensor):
            seg_images = torch.from_numpy(seg_images).to(self.device)
        else:
            seg_images = seg_images.to(self.device)
            
        seg_images = seg_images.float() / 255.0
        sam3_embed = self.encode_image(seg_images)
        
        if self.use_half and self.device != "cpu":
            sam3_embed = sam3_embed.half()
            
        return sam3_embed