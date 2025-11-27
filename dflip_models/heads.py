"""
Task-specific heads for DFLIP Stage 1 (Profiler).
Includes Detection, Identification, and Localization heads.

NOTE: SimpleTaskHead and SpatialTaskHead have been moved to unified_fusion.py
These classes are kept here for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import new simplified heads
from .unified_fusion import SimpleTaskHead, SpatialTaskHead


class DetectionHead(nn.Module):
    """
    Binary classification head for real/fake detection.
    
    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    """
    
    def __init__(self, in_features: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)  # Binary: [Real, Fake]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) pooled vision features
        
        Returns:
            logits: (B, 2) classification logits
        """
        return self.classifier(features)


class IdentificationHead(nn.Module):
    """
    Multi-class classification head for generator model identification.
    
    Args:
        in_features: Input feature dimension
        num_classes: Number of generator models + 1 (for Real)
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) pooled vision features
        
        Returns:
            logits: (B, num_classes) classification logits
        """
        return self.classifier(features)


class LocalizationHead(nn.Module):
    """
    Segmentation head for forgery localization (heatmap generation).
    Uses a simple decoder to upsample features to image resolution.
    
    Args:
        in_channels: Input feature channels
        hidden_dim: Hidden channel dimension
        output_size: Output spatial size (H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        output_size: tuple = (448, 448)
    ):
        super().__init__()
        
        self.output_size = output_size
        
        # Progressive upsampling decoder
        self.decoder = nn.ModuleList([
            # Stage 1: Reduce channels
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
            # Stage 2: Upsample
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
            ),
            # Stage 3: Upsample
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 4),
                nn.ReLU(inplace=True),
            ),
            # Stage 4: Upsample
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 8),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) spatial vision features
        
        Returns:
            mask: (B, 1, H_out, W_out) forgery probability heatmap
        """
        x = features
        
        # Apply decoder stages
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        # Final prediction
        mask = self.final_conv(x)
        
        # Resize to target output size if needed
        if x.shape[-2:] != self.output_size:
            mask = F.interpolate(
                mask,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )
        
        return mask


class MultiTaskLoss(nn.Module):
    """
    Combined loss for Stage 1 multi-task learning.
    
    Combines:
    - Binary Cross Entropy for detection
    - Cross Entropy for identification
    - Dice Loss for localization
    - MSE Loss for decoder reconstruction (optional)
    
    Args:
        detection_weight: Weight for detection loss
        identification_weight: Weight for identification loss
        localization_weight: Weight for localization loss
        reconstruction_weight: Weight for reconstruction loss
    """
    
    def __init__(
        self,
        detection_weight: float = 1.0,
        identification_weight: float = 1.0,
        localization_weight: float = 0.5,
        reconstruction_weight: float = 0.3
    ):
        super().__init__()
        
        self.detection_weight = detection_weight
        self.identification_weight = identification_weight
        self.localization_weight = localization_weight
        self.reconstruction_weight = reconstruction_weight
        
        self.bce_loss = nn.CrossEntropyLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """
        Dice loss for segmentation.
        
        Args:
            pred: (B, 1, H, W) predicted mask logits
            target: (B, 1, H, W) ground truth binary mask
            smooth: Smoothing factor
        
        Returns:
            dice_loss: Scalar loss value
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred_sigmoid.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return 1.0 - dice.mean()
    
    def forward(
        self,
        detection_logits: torch.Tensor,
        identification_logits: torch.Tensor,
        localization_pred: torch.Tensor,
        is_fake_labels: torch.Tensor,
        generator_labels: torch.Tensor,
        mask_labels: torch.Tensor,
        decoder_output: torch.Tensor = None,
        reconstruction_target: torch.Tensor = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            detection_logits: (B, 2) from DetectionHead
            identification_logits: (B, num_classes) from IdentificationHead
            localization_pred: (B, 1, H, W) from LocalizationHead
            is_fake_labels: (B,) binary labels
            generator_labels: (B,) generator class labels
            mask_labels: (B, 1, H, W) binary masks
            decoder_output: (B, D) from Decoder (optional)
            reconstruction_target: (B, D) target features for reconstruction (optional)
        
        Returns:
            Dictionary with total loss and individual components
        """
        # Detection loss (Binary CE)
        det_loss = self.bce_loss(detection_logits, is_fake_labels)
        
        # Identification loss (Multi-class CE)
        iden_loss = self.ce_loss(identification_logits, generator_labels)
        
        # Localization loss (Dice)
        # Only compute for fake images with masks
        has_mask = (mask_labels.sum(dim=(1, 2, 3)) > 0)
        if has_mask.any():
            loc_loss = self.dice_loss(
                localization_pred[has_mask],
                mask_labels[has_mask]
            )
        else:
            loc_loss = torch.tensor(0.0, device=detection_logits.device)
        
        # Reconstruction loss (MSE)
        # Only compute if decoder output is provided
        if decoder_output is not None and reconstruction_target is not None:
            recon_loss = self.mse_loss(decoder_output, reconstruction_target)
        else:
            recon_loss = torch.tensor(0.0, device=detection_logits.device)
        
        # Combined loss
        total_loss = (
            self.detection_weight * det_loss +
            self.identification_weight * iden_loss +
            self.localization_weight * loc_loss +
            self.reconstruction_weight * recon_loss
        )
        
        return {
            'loss': total_loss,
            'detection_loss': det_loss.item(),
            'identification_loss': iden_loss.item(),
            'localization_loss': loc_loss.item() if isinstance(loc_loss, torch.Tensor) else 0.0,
            'reconstruction_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0.0
        }
