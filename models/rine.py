import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict

import clip
from utils.registry import MODELS


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class RINEModel(nn.Module):
    def __init__(self, backbone0="ViT-L/14", backbone1=768, nproj=2, proj_dim=512, **kwargs):
        super().__init__()

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone0, device="cpu")
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone1 if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]

        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g

        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return {'logits': p, 'z': z}


@MODELS.register_module()
class RINEClassifier(nn.Module):
    """
    RINE (Residual Intermediate Network Embedding) binary classifier adapted for the DFLIP framework.
    Uses CLIP intermediate features with attention mechanism for fake detection.
    """
    def __init__(
        self,
        backbone0: str = "ViT-L/14",
        backbone1: int = 768,
        nproj: int = 2,
        proj_dim: int = 512,
        num_classes: int = 2,  # Binary classification: real/fake
        verbose: bool = True,
    ):
        super().__init__()
        
        self.backbone0 = backbone0
        self.backbone1 = backbone1
        self.nproj = nproj
        self.proj_dim = proj_dim
        
        # Create RINE model
        self.model = RINEModel(
            backbone0=backbone0,
            backbone1=backbone1,
            nproj=nproj,
            proj_dim=proj_dim
        )
        
        # Add final classification layer if needed for multi-class
        if num_classes > 1:
            self.classifier = nn.Linear(1, num_classes)
        else:
            self.classifier = None
        
        if verbose:
            self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"RINE Binary Classifier ({self.backbone0}):")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) tensor of images
            
        Returns:
            dict with keys:
                - logits: (B, num_classes) classification logits
                - features: (B, feature_dim) extracted features
        """
        # Forward through RINE model
        outputs = self.model(pixel_values)
        
        logits = outputs['logits']
        features = outputs['z']
        
        # Apply final classifier if multi-class
        if self.classifier is not None:
            logits = self.classifier(logits)
        
        # Return in the expected format for the framework
        return {
            "logits": logits,
            "features": features,
        }


def create_rine(config: Dict, verbose: bool = True) -> RINEClassifier:
    """Create a RINE binary classifier."""
    model_config = config.get("model", {})
    
    # Extract backbone configuration
    backbone0 = model_config.get("backbone0", "ViT-L/14")
    backbone1 = model_config.get("backbone1", 768)
    nproj = model_config.get("nproj", 2)
    proj_dim = model_config.get("proj_dim", 512)
    
    model = RINEClassifier(
        backbone0=backbone0,
        backbone1=backbone1,
        nproj=nproj,
        proj_dim=proj_dim,
        num_classes=2,  # Binary classification
        verbose=verbose,
    )
    
    if verbose:
        print(f"Created RINE Binary Classifier for real/fake detection")
        print(f"  - Backbone: {backbone0}")
        print(f"  - Feature dim: {backbone1}")
        print(f"  - Projection layers: {nproj}")
        print(f"  - Projection dim: {proj_dim}")
    
    return model