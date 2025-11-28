import torch
import torch.nn as nn
from typing import Dict

from .vision_encoder import Qwen3_VisionEncoder


class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DFLIPProfiler(nn.Module):
    """
    Simplified DFLIPProfiler with frozen vision encoder,
    simple MLP classifiers for binary and multiclass tasks,
    localization (forgery area) head is omitted for now.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        num_generators: int = 10,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.vision_encoder = Qwen3_VisionEncoder(model_name, freeze=freeze_encoder)
        self.vision_hidden_size = self.vision_encoder.visual.config.hidden_size

        # Simple MLP classifiers replacing previous task heads
        self.detection_head = SimpleMLPClassifier(
            input_dim=self.vision_hidden_size,
            output_dim=2,  # binary classification: real vs fake
        )
        self.identification_head = SimpleMLPClassifier(
            input_dim=self.vision_hidden_size,
            output_dim=num_generators,  # multiclass identification
        )

        # No localization head for now

        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")

    def forward(self, pixel_values: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, C, H, W) input images, H and W multiples of 28
            return_features: whether to return intermediate pooled features
        
        Returns:
            Dictionary with detection_logits and identification_logits
            optionally pooled_features if return_features=True
        """
        features = self.vision_encoder(pixel_values)  # (B, D, H', W')
        pooled_features = features.mean(dim=[2, 3])  # global avg pooling over H' and W'

        detection_logits = self.detection_head(pooled_features)
        identification_logits = self.identification_head(pooled_features)

        outputs = {
            "detection_logits": detection_logits,
            "identification_logits": identification_logits,
        }

        if return_features:
            outputs["pooled_features"] = pooled_features

        return outputs


def create_profiler(config: Dict) -> DFLIPProfiler:
    model_config = config.get("model", {})
    profiler = DFLIPProfiler(
        model_name=model_config.get("base_model", "Qwen/Qwen3-VL-2B-Instruct"),
        num_generators=config.get("num_generators", 10),
        freeze_encoder=model_config.get("freeze_encoder", True),
    )
    return profiler
