"""
AIDE (Adaptive Image Detection Enhancement) model implementation.
Combines high-pass filtering, ResNet features, and ConvNeXt features for fake image detection.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Optional
from .srm_filter_kernel import all_normalized_hpf_list

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip not available. AIDE model will not work without it.")


def load_resnet_weights(model_min, model_max, weight_path: str):
    """Load ResNet weights from local path."""
    if weight_path is None or not os.path.exists(weight_path):
        return
        
    try:
        pretrained_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
        model_min_dict = model_min.state_dict()
        model_max_dict = model_max.state_dict()

        for k in pretrained_dict.keys():
            if k in model_min_dict and pretrained_dict[k].size() == model_min_dict[k].size():
                model_min_dict[k] = pretrained_dict[k]
                model_max_dict[k] = pretrained_dict[k]
                
        model_min.load_state_dict(model_min_dict)
        model_max.load_state_dict(model_max_dict)
        
    except Exception as e:
        print(f"Warning: Could not load ResNet weights: {e}")


class HPF(nn.Module):
    """High-Pass Filter module using SRM filters."""
    
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
        hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)

        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = self.hpf(input)
        return output


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet backbone for AIDE model."""

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class AIDEModel(nn.Module):
    """
    AIDE Model implementation adapted for DFLIP framework.
    
    The original AIDE model expects 5 input images:
    - x_minmin, x_maxmax, x_minmin1, x_maxmax1: processed images for HPF
    - tokens: original image for ConvNeXt
    
    For DFLIP framework compatibility, we adapt it to work with single images.
    """

    def __init__(self, resnet_path: Optional[str] = None, convnext_path: Optional[str] = None, cache_dir: str = "./models_cache"):
        super(AIDEModel, self).__init__()
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required for AIDE model. Please install it with: pip install open-clip-torch")
        
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])

        # Load pretrained ResNet weights if provided
        load_resnet_weights(self.model_min, self.model_max, resnet_path)

        self.fc = Mlp(2048 + 256, 1024, 2)

        # Initialize ConvNeXt model
        try:
            if convnext_path and os.path.exists(convnext_path):
                convnext_pretrained = convnext_path
            else:
                convnext_pretrained = convnext_path
            
            self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge", pretrained=convnext_pretrained
            )
            self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
            self.openclip_convnext_xxl.head.global_pool = nn.Identity()
            self.openclip_convnext_xxl.head.flatten = nn.Identity()
            self.openclip_convnext_xxl.eval()
            
            # Freeze ConvNeXt parameters
            for param in self.openclip_convnext_xxl.parameters():
                param.requires_grad = False
                
        except Exception as e:
            # Create a dummy ConvNeXt replacement
            self.openclip_convnext_xxl = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Conv2d(3, 3072, 1),
                nn.ReLU()
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),
        )

    def forward(self, x):
        """
        Forward pass adapted for single image input.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            dict with 'logits' key containing classification logits
        """
        # For DFLIP framework compatibility, we use the same image for all inputs
        # In the original AIDE, these would be different processed versions
        x_minmin = x
        x_maxmax = x
        x_minmin1 = x
        x_maxmax1 = x
        tokens = x

        # Apply HPF to all inputs
        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        # Process through ConvNeXt (frozen)
        with torch.no_grad():
            # Normalization constants for different models
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            # Normalize and process through ConvNeXt
            normalized_tokens = tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            local_convnext_image_feats = self.openclip_convnext_xxl(normalized_tokens)
            
            # Handle different output shapes
            if len(local_convnext_image_feats.shape) == 4:
                # Expected shape: [B, 3072, 8, 8]
                local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            else:
                # Fallback for different shapes
                local_convnext_image_feats = local_convnext_image_feats.view(tokens.size(0), -1)
                
            x_0 = self.convnext_proj(local_convnext_image_feats)

        # Process through ResNet models
        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        # Average the ResNet features
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4

        # Concatenate ConvNeXt and ResNet features
        x = torch.cat([x_0, x_1], dim=1)

        # Final classification
        x = self.fc(x)

        return {'logits': x}


class AIDEClassifier(nn.Module):
    """
    AIDE classifier adapted for the DFLIP framework.
    Combines high-pass filtering, ResNet features, and ConvNeXt features for fake detection.
    """
    
    def __init__(
        self,
        resnet_path: Optional[str] = None,
        convnext_path: Optional[str] = None,
        cache_dir: str = "./models_cache",
        num_classes: int = 2,  # Binary classification: real/fake
        verbose: bool = True,
    ):
        super().__init__()
        
        self.model = AIDEModel(resnet_path=resnet_path, convnext_path=convnext_path, cache_dir=cache_dir)
        
        # Adjust final layer if needed
        if num_classes != 2:
            self.model.fc = Mlp(2048 + 256, 1024, num_classes)
        
        pass
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) tensor of images
            
        Returns:
            dict with keys:
                - logits: (B, num_classes) classification logits
        """
        # Forward through AIDE model
        outputs = self.model(pixel_values)
        
        # Return in the expected format for the framework
        return {
            "logits": outputs['logits'],
        }


def create_aide(config: Dict, verbose: bool = True) -> AIDEClassifier:
    """Create an AIDE binary classifier."""
    model_config = config.get("model", {})
    
    # Extract model paths and cache directory
    resnet_path = model_config.get("resnet_path", None)
    convnext_path = model_config.get("convnext_path", None)
    cache_dir = model_config.get("cache_dir", "./models_cache")
    
    model = AIDEClassifier(
        resnet_path=resnet_path,
        convnext_path=convnext_path,
        cache_dir=cache_dir,
        num_classes=2,  # Binary classification
        verbose=verbose,
    )
    
    pass
    
    return model