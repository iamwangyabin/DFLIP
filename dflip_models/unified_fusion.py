"""
Unified Feature Fusion Module with Task Token Mechanism
Simplifies multi-task learning by using learnable task tokens to query multi-scale features.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class UnifiedFeatureFusion(nn.Module):
    """
    Unified multi-scale feature fusion module using task tokens.
    
    This module takes multi-scale features extracted from different layers
    and fuses them based on a task-specific token using cross-attention.
    
    Architecture:
        1. Project each scale's features to common dimension
        2. Use task token as Query, multi-scale features as Key/Value
        3. Cross-attention to get task-specific weighted features
        4. Output pooled features for downstream MLP
    
    Args:
        hidden_size: Feature dimension (from Qwen VIT)
        num_scales: Number of feature scales to fuse
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_scales = num_scales
        self.num_heads = num_heads
        
        # Project features from each scale to common dimension
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            )
            for _ in range(num_scales)
        ])
        
        # Cross-attention: task token queries multi-scale features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        
        # Feed-forward network for refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        task_token: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-scale features using task token.
        
        Args:
            multi_scale_features: List of (B, N, D) tensors from different layers
            task_token: (B, 1, D) learnable task token
        
        Returns:
            task_features: (B, D) task-specific fused features
        """
        assert len(multi_scale_features) == self.num_scales, \
            f"Expected {self.num_scales} scales, got {len(multi_scale_features)}"
        
        B = task_token.shape[0]
        
        # Project each scale's features
        projected_features = []
        for i, features in enumerate(multi_scale_features):
            # features: (B, N, D)
            projected = self.scale_projections[i](features)
            projected_features.append(projected)
        
        # Concatenate all scales along sequence dimension
        # This creates a unified feature "memory" from all scales
        concat_features = torch.cat(projected_features, dim=1)  # (B, N_total, D)
        
        # Cross-attention: task token attends to multi-scale features
        # Query: task_token (B, 1, D)
        # Key/Value: concat_features (B, N_total, D)
        attn_output, attn_weights = self.cross_attention(
            query=task_token,
            key=concat_features,
            value=concat_features
        )  # attn_output: (B, 1, D)
        
        # Residual connection + normalization
        task_features = self.output_norm(task_token + self.output_dropout(attn_output))
        
        # Feed-forward network with residual
        ffn_output = self.ffn(task_features)
        task_features = self.ffn_norm(task_features + ffn_output)  # (B, 1, D)
        
        # Remove sequence dimension
        task_features = task_features.squeeze(1)  # (B, D)
        
        return task_features


class SimpleTaskHead(nn.Module):
    """
    Simple MLP-based task head for classification.
    
    Takes fused task-specific features and produces class logits.
    Much simpler than the previous DetectionHead/IdentificationHead.
    
    Args:
        in_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) task-specific features
        
        Returns:
            logits: (B, num_classes) classification logits
        """
        return self.mlp(features)


class SpatialTaskHead(nn.Module):
    """
    Spatial task head for localization/segmentation tasks.
    
    Uses task token to query spatial features and produce a heatmap.
    Alternative to the complex LocalizationHead.
    
    Args:
        in_channels: Input feature channels
        hidden_dim: Hidden dimension for decoder
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
        
        # Simple convolutional decoder
        self.decoder = nn.Sequential(
            # Reduce channels
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Upsample stages
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            
            # Final prediction
            nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)
        )
    
    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: (B, C, H, W) spatial features
        
        Returns:
            mask: (B, 1, H_out, W_out) segmentation mask logits
        """
        mask = self.decoder(spatial_features)
        
        # Resize if needed
        if mask.shape[-2:] != self.output_size:
            mask = torch.nn.functional.interpolate(
                mask,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )
        
        return mask
