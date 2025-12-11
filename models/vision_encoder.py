import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText
import timm
from typing import Dict, List, Tuple

# class Qwen3_VisionEncoder(nn.Module):
#     def __init__(self, model_path, freeze=True):
#         super().__init__()
#         full_model = AutoModelForImageTextToText.from_pretrained(
#             model_path,
#             device_map="cpu",
#             trust_remote_code=True,
#             torch_dtype=torch.float16
#         )
#
#         self.visual = full_model.visual
#         cfg = full_model.config.vision_config
#
#         self.patch_size = getattr(cfg, "patch_size", 14)
#         self.merge_size = getattr(cfg, "spatial_merge_size", 2)
#         self.temp_patch_size = getattr(cfg, "temporal_patch_size", 2)
#         self.stride = self.patch_size * self.merge_size  # 28
#         self.pixel_dim = 3 * self.temp_patch_size * self.patch_size ** 2  # 1176
#
#         del full_model
#         import gc
#         gc.collect()
#
#         if freeze:
#             self.visual.requires_grad_(False)
#             self.visual.eval()
#
#     def forward(self, x):
#         """
#         Args:
#             x: (Batch, 3, H, W) - H, W 必须是 28 的倍数
#         Returns:
#             feat: (Batch, Dim, H/28, W/28) - 类似 ResNet 的 Feature Map
#         """
#         B, C, H, W = x.shape
#
#         if self.temp_patch_size > 1:
#             x_pad = x.unsqueeze(2).repeat(1, 1, self.temp_patch_size, 1, 1)
#         else:
#             x_pad = x.unsqueeze(2)
#
#         x_pad = x_pad.view(B, -1, H, W)
#         patches = F.unfold(x_pad, kernel_size=self.patch_size, stride=self.patch_size)
#         pixel_values = patches.transpose(1, 2).reshape(-1, self.pixel_dim).to(self.visual.dtype)
#
#         grid_thw = torch.tensor([1, H, W], device=x.device).unsqueeze(0).repeat(B, 1)
#
#         outputs = self.visual(pixel_values, grid_thw=grid_thw)
#         hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
#
#         feat_h, feat_w = H // self.stride, W // self.stride
#         dim = hidden_states.shape[-1]
#
#         return hidden_states.view(B, feat_h * feat_w, dim).permute(0, 2, 1).reshape(B, dim, feat_h, feat_w)

class TimmMultiLevelEncoder(nn.Module):
    
    def __init__(self, model_name: str = "vit_base_patch14_dinov2", freeze: bool = True, pretrained: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Create model with features_only=True to get intermediate features
        self.backbone = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=(0, 1, 2, 3),  # Extract all 4 levels
        )
        
        # Get the model's expected input size
        # Most timm models store this in pretrained_cfg or default_cfg
        if hasattr(self.backbone, 'pretrained_cfg') and self.backbone.pretrained_cfg is not None:
            input_size = self.backbone.pretrained_cfg.get('input_size', (3, 224, 224))
        elif hasattr(self.backbone, 'default_cfg') and self.backbone.default_cfg is not None:
            input_size = self.backbone.default_cfg.get('input_size', (3, 224, 224))
        else:
            input_size = (3, 224, 224)
        
        # Get feature dimensions for each level
        # Create dummy input to infer feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_size)
            dummy_features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in dummy_features]
            self.num_levels = len(dummy_features)
        
        # Total dimension after concatenation
        self.fused_dim = sum(self.feature_dims)
        
        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
    
    def forward(self, x: torch.Tensor,) -> torch.Tensor:

        multi_level_features = self.backbone(x)
        
        # Global average pooling for each level
        pooled_features = []
        for level_feat in multi_level_features:
            # level_feat shape: (B, C, H, W)
            # Global average pooling: (B, C, H, W) -> (B, C)
            pooled = level_feat.mean(dim=[2, 3])  # Global average pooling
            pooled_features.append(pooled)

        fused_features = torch.cat(pooled_features, dim=1)  # (B, sum(dims))
        return fused_features
