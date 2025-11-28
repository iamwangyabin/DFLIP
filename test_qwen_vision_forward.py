"""
Qwen Vision 模型前向传播测试脚本
不需要训练，仅测试随机输入的输出

这个脚本用来验证DFLIPProfiler模型的forward pass是否正确运行
"""

import torch
import torch.nn as nn
from dflip_models.profiler import DFLIPProfiler

model = DFLIPProfiler(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    num_generators=10,
)

model.eval()


batch_size = 2
num_channels = 3
height = 448
width = 448
pixel_values = torch.randn(batch_size, num_channels, height, width)
outputs = model.forward(pixel_values, return_features=True)


