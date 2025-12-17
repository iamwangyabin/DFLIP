import clip
import os
import torch
import torch.nn as nn
from typing import Dict

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "RN50x64": 1024,
    "ViT-L/14@336px": 768,
}


class OjhaModel(nn.Module):
    """
    Ojha method for fake detection using CLIP features.
    Based on the approach from Ojha et al.
    """
    def __init__(self, name="ViT-L/14", num_classes=2):
        super(OjhaModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(CHANNELS[name], num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, **kwargs):
        features = self.model.encode_image(x)

        return {'logits': self.fc(features),
                'features': features}


class OjhaClassifier(nn.Module):
    """
    Ojha binary classifier adapted for the DFLIP framework.
    Uses CLIP features for fake detection.
    """
    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        num_classes: int = 2,  # Binary classification: real/fake
        verbose: bool = True,
    ):
        super().__init__()
        
        self.clip_model_name = clip_model_name
        
        # Create Ojha model
        self.model = OjhaModel(name=clip_model_name, num_classes=num_classes)
        
        if verbose:
            self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Ojha Binary Classifier ({self.clip_model_name}):")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) tensor of images
            
        Returns:
            dict with keys:
                - logits: (B, 2) binary classification logits
                - features: (B, feature_dim) CLIP features
        """
        # Forward through Ojha model
        outputs = self.model(pixel_values)
        
        # Return in the expected format for the framework
        return {
            "logits": outputs['logits'],
            "features": outputs['features'],
        }


def create_ojha(config: Dict, verbose: bool = True) -> OjhaClassifier:
    """Create an Ojha binary classifier."""
    model_config = config.get("model", {})
    
    model = OjhaClassifier(
        clip_model_name=model_config.get("clip_model_name", "ViT-L/14"),
        num_classes=2,  # Binary classification
        verbose=verbose,
    )
    
    if verbose:
        print(f"Created Ojha Binary Classifier for real/fake detection")
    return model