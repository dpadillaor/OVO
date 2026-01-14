from typing import Dict, List
import torch
import yaml
import os

# Assuming a dino_utils.py will be created for DINO-specific utilities
# from ..utils import dino_utils
from ..utils import segment_utils

class DINOGenerator:
    def __init__(self, config: Dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model_card = config.get("model_card", "dinov2_vitg14") # Default DINOv2 model
        
        # Placeholder for DINO model loading
        # self.model, self.preprocess, dino_dim = dino_utils.load_dino_model(self.model_card, config.get("use_half", False))
        # self.dino_dim = dino_dim
        
        # Placeholder values for now
        self.dino_dim = 1024 # Example dimension for DINOv2-Giant
        self.model = None # Placeholder
        self.preprocess = None # Placeholder

        if config.get("use_half", False):
            # self.model.half() # If model supports half precision
            pass
        
        self.to(self.device)

    @property
    def get_dino_dim(self) -> int:
        return self.dino_dim
            
    def to(self, device: str) -> None:
        """
        Move predictor model to either 'cpu' or 'cuda' device.
        Args:
            device (str): device to mode the model to.
        """
        if "cuda" in device:
            return self.cuda()
        else:
            return self.cpu()

    def cpu(self) -> None:
        """
        Move predictor model to cpu device.
        """
        self.device = "cpu"
        if self.model:
            self.model.cpu()

    def cuda(self) -> None:
        """
        Move predictor model to cuda default device.
        """
        self.device = "cuda"
        if self.model:
            self.model.cuda()

    @torch.no_grad
    def encode_image(self, input: torch.Tensor) -> torch.Tensor:
        """ Compute DINO descriptor of an RGB image
        Args:
            - input (torch.Tensor): RGB image as tensor with shape (3,H,W) in range [0,1]
        Return:
            - dino_descriptor (torch.Tensor): as tensor with shape (self.dino_dim)
        """
        if self.preprocess:
            processed_input = self.preprocess(input)
        else:
            processed_input = input # Placeholder if no preprocess
        
        # Placeholder for actual DINO model inference
        # return self.model(processed_input)
        
        # Return dummy tensor for now
        return torch.randn(input.shape[0], self.dino_dim, device=self.device)

    @torch.no_grad
    def extract_dino(self, image: torch.Tensor, binary_maps: torch.Tensor) -> torch.Tensor:
        """ Computes a DINO vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor): Full source RGB image with dimensions (3,H,W) and range 0-255.
            - binary_maps (torch.Tensor): A tensor of (N, H, W) containing N binary maps, one for each segmented instance.
        Return:
            - dino_embeds: list of numpy arrays with dim (N, self.dino_dim).        
        """
        if len(image.shape) == 3:
            image = image[None, ...] # Add batch dimension if missing

        # DINO typically works on full images or patches. For mask-specific features,
        # we might need to extract features from the original image and then pool them
        # based on the masks, or crop the image for each mask.
        # For now, let's assume we can get a feature for each mask by cropping.

        dino_embeds = []
        for i in range(binary_maps.shape[0]):
            mask = binary_maps[i]
            # Crop image based on mask bounding box
            y_indices, x_indices = torch.where(mask)
            if y_indices.numel() == 0: # Empty mask
                dino_embeds.append(torch.zeros(self.dino_dim, device=self.device))
                continue
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            cropped_image = image[:, :, y_min:y_max+1, x_min:x_max+1]
            
            # Resize cropped image to a fixed size for DINO model if necessary
            # For now, just pass it to encode_image (which is a placeholder)
            
            # Assuming encode_image can handle a batch of 1 cropped image
            embed = self.encode_image(cropped_image.float() / 255.0).squeeze(0)
            dino_embeds.append(embed)
        
        if len(dino_embeds) == 0:
            return torch.tensor([], device=self.device)

        return torch.stack(dino_embeds)
