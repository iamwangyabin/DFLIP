import torch
import torch.nn as nn
import timm
from typing import Dict


class BaselineClassifier(nn.Module):
    """
    Simple baseline classifier with three independent classification heads.
    Uses timm in the most traditional way - just the final pooled features.
    One backbone with three MLP heads: detection, family, version.
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2",
        num_families: int = 27,
        num_versions: int = 1386,
        freeze_encoder: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Simple timm backbone - traditional baseline approach
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classifier head, get features only
        )
        
        # Get feature dimension from the model
        self.vision_hidden_size = self.backbone.num_features
        
        if freeze_encoder:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        
        # Detection head (real/fake binary classification)
        self.detection_head = nn.Sequential(
            nn.Linear(self.vision_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary: real or fake
        )
        
        # Family classification head
        self.family_head = nn.Sequential(
            nn.Linear(self.vision_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_families)
        )
        
        # Version classification head
        self.version_head = nn.Sequential(
            nn.Linear(self.vision_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_versions)
        )
        
        self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) tensor of images
            
        Returns:
            dict with keys:
                - detection_logits: (B, 2) binary classification logits
                - family_logits: (B, num_families) family classification logits
                - version_logits: (B, num_versions) version classification logits
        """
        # Extract features using simple timm approach
        features = self.backbone(pixel_values)  # (B, num_features)
        
        # Pass through three independent heads
        detection_logits = self.detection_head(features)  # (B, 2)
        family_logits = self.family_head(features)  # (B, num_families)
        version_logits = self.version_head(features)  # (B, num_versions)
        
        return {
            "detection_logits": detection_logits,
            "family_logits": family_logits,
            "version_logits": version_logits,
        }


def create_baseline(config: Dict) -> BaselineClassifier:
    """Create a baseline classifier with three heads."""
    model_config = config.get("model", {})
    bhep_config = config.get("bhep", {})
    
    model = BaselineClassifier(
        model_name=model_config.get("base_model", "vit_small_patch14_dinov2"),
        num_families=bhep_config.get("num_families", 27),
        num_versions=bhep_config.get("num_versions", 1386),
        freeze_encoder=model_config.get("freeze_encoder", True),
        hidden_dim=512,
        dropout=0.3,
    )
    
    print(f"Created Baseline Classifier with 3 heads: "
          f"detection (2), family ({bhep_config.get('num_families', 27)}), "
          f"version ({bhep_config.get('num_versions', 1386)})")
    return model