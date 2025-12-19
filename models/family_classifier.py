import torch
import torch.nn as nn
import timm
from typing import Dict


class FamilyClassifier(nn.Module):
    """
    Simple family classifier with timm backbone and single classification head.
    Only performs family classification on fake images (family_id >= 0).
    Real images (family_id = -1) are ignored during training.
    """
    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        num_families: int = 27,
        freeze_encoder: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        verbose: bool = True,
    ):
        super().__init__()
        
        # Simple timm backbone - traditional approach
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
        
        if verbose:
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
                - family_logits: (B, num_families) family classification logits
        """
        # Extract features using simple timm approach
        features = self.backbone(pixel_values)  # (B, num_features)
        
        # Pass through family classification head
        family_logits = self.family_head(features)  # (B, num_families)
        
        return {
            "family_logits": family_logits,
        }



class FamilyClassifierLoss(nn.Module):
    """
    Loss function for family classifier.
    Only computes loss on fake images (family_id >= 0).
    Real images (family_id = -1) are ignored.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs containing 'family_logits'
            targets: Targets containing 'family_ids' and 'is_fake'
        
        Returns:
            Dict containing loss and metrics
        """
        family_ids = targets['family_ids']
        is_fake = targets.get('is_fake', None)
        
        # Only compute loss on fake images with valid family_ids
        if is_fake is not None:
            # Use both is_fake and family_ids >= 0 to filter
            valid_mask = (is_fake == 1) & (family_ids >= 0)
        else:
            # If is_fake not provided, just use family_ids >= 0
            valid_mask = family_ids >= 0
        
        if valid_mask.sum() == 0:
            # No valid samples in this batch
            device = outputs['family_logits'].device
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "family_loss": 0.0,
                "num_valid_samples": 0,
            }
        
        # Extract valid samples
        valid_logits = outputs['family_logits'][valid_mask]
        valid_targets = family_ids[valid_mask]
        
        # Compute cross-entropy loss
        family_loss = self.criterion(valid_logits, valid_targets)
        
        return {
            "loss": family_loss,
            "family_loss": family_loss.item(),
            "num_valid_samples": valid_mask.sum().item(),
        }


def create_family_classifier(config: Dict, verbose: bool = True) -> FamilyClassifier:
    """Create a family classifier."""
    model_config = config.get("model", {})
    bhep_config = config.get("bhep", {})
    
    model = FamilyClassifier(
        model_name=model_config.get("base_model", "vit_base_patch14_dinov2"),
        num_families=bhep_config.get("num_families", 27),
        freeze_encoder=model_config.get("freeze_encoder", True),
        hidden_dim=512,
        dropout=0.3,
        verbose=verbose,
    )
    
    if verbose:
        print(f"Created Family Classifier with {bhep_config.get('num_families', 27)} families")
    
    return model