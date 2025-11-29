import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText

"""
================================================================================
Qwen3-VL Vision Encoder 独立封装 (Standard Backbone Wrapper)
================================================================================

【功能说明】
该类将 Qwen3-VL 的 Vision Tower 提取并封装为一个标准的计算机视觉 Backbone。
它屏蔽了原模型内部复杂的动态分辨率（Dynamic Resolution）、Grid 计算、
以及 Flatten 操作。

你可以像使用 ResNet 或标准的 ViT (timm style) 一样使用这个模块：
输入一个 Batch 的图像张量，输出保持空间结构的特征图 (Feature Map)。

【关键参数】
- Patch Size: 14 (Qwen3-VL 默认)
- Merge Size: 2 (空间合并倍数)
- Stride (步长): 28 (14 * 2) -> 这意味着输入分辨率必须是 28 的倍数。

【输入输出详解】

1. 输入 (Input):
   - 格式: Tensor
   - 形状: (Batch_Size, 3, Height, Width)
   - 约束: 
     1. Height 和 Width 必须能被 28 整除 (例如 224, 336, 448, 1024...)。
     2. 建议使用 float16 或 bfloat16 数据类型以匹配模型权重。
     3. 位于 GPU (或其他模型所在设备) 上。

2. 输出 (Output):
   - 格式: Tensor
   - 形状: (Batch_Size, Hidden_Dim, Feat_H, Feat_W)
   - 维度解释:
     - Hidden_Dim: 视觉模型的输出维度 (例如 7B 模型通常为 3584, 2B 模型可能不同)。
     - Feat_H: 输入 Height / 28
     - Feat_W: 输入 Width / 28
   - 物理含义: 这是一个保持了 2D 空间结构的特征图，适合接入检测头、分割头或池化分类层。

【使用示例】

    # 1. 初始化
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct" # 或 Qwen3 路径
    encoder = Qwen3_VisionEncoder(model_path, freeze=True).cuda()

    # 2. 准备数据 (Batch=4, 分辨率 336x336)
    # 注意: 336 / 28 = 12
    dummy_input = torch.randn(4, 3, 336, 336).cuda().half()

    # 3. 前向传播
    features = encoder(dummy_input)

    # 4. 检查输出
    # 预期形状: (4, 3584, 12, 12)
    print(features.shape) 

================================================================================
"""

class Qwen3_VisionEncoder(nn.Module):
    def __init__(self, model_path, freeze=True):
        super().__init__()
        full_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        self.visual = full_model.visual
        cfg = full_model.config.vision_config

        self.patch_size = getattr(cfg, "patch_size", 14)
        self.merge_size = getattr(cfg, "spatial_merge_size", 2)
        self.temp_patch_size = getattr(cfg, "temporal_patch_size", 2)
        self.stride = self.patch_size * self.merge_size  # 28
        self.pixel_dim = 3 * self.temp_patch_size * self.patch_size ** 2  # 1176

        del full_model
        import gc;
        gc.collect()

        if freeze:
            self.visual.requires_grad_(False)
            self.visual.eval()

    def forward(self, x):
        """
        Args:
            x: (Batch, 3, H, W) - H, W 必须是 28 的倍数
        Returns:
            feat: (Batch, Dim, H/28, W/28) - 类似 ResNet 的 Feature Map
        """
        B, C, H, W = x.shape

        if self.temp_patch_size > 1:
            x_pad = x.unsqueeze(2).repeat(1, 1, self.temp_patch_size, 1, 1)
        else:
            x_pad = x.unsqueeze(2)

        x_pad = x_pad.view(B, -1, H, W)
        patches = F.unfold(x_pad, kernel_size=self.patch_size, stride=self.patch_size)
        pixel_values = patches.transpose(1, 2).reshape(-1, self.pixel_dim).to(self.visual.dtype)

        grid_thw = torch.tensor([1, H, W], device=x.device).unsqueeze(0).repeat(B, 1)

        outputs = self.visual(pixel_values, grid_thw=grid_thw)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        feat_h, feat_w = H // self.stride, W // self.stride
        dim = hidden_states.shape[-1]

        return hidden_states.view(B, feat_h * feat_w, dim).permute(0, 2, 1).reshape(B, dim, feat_h, feat_w)