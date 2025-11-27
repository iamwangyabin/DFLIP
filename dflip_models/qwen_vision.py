"""
Stage 1: The Profiler (Refactored with Task Token Architecture)
Multi-task model with frozen Qwen VIT backbone + Unified Feature Fusion + Task Tokens.
"""

import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, Optional, List

from .unified_fusion import UnifiedFeatureFusion, SimpleTaskHead, SpatialTaskHead
from .heads import MultiTaskLoss


class DFLIPProfiler(nn.Module):
    """
    Stage 1: The Profiler (Refactored with Task Token Architecture)
    
    Simplified multi-task model that performs:
    1. Detection: Real vs Fake classification (using binary_task_token)
    2. Identification: Generator model classification (using multiclass_task_token)
    3. Localization: Forgery region segmentation (using localization_task_token)
    
    Architecture:
        Frozen Qwen2.5-VL → Extract Multi-Scale Features → 
        Task Tokens + Unified Fusion → Simple MLP Heads → Predictions
    
    Args:
        model_name: Hugging Face model name
        num_generators: Number of generator classes (including Real=0)
        extract_layers: List of layer indices to extract intermediate features from
        device: Target device
        cache_dir: Model cache directory
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        num_generators: int = 10,
        extract_layers: List[int] = [6, 12, 18],
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_generators = num_generators
        self.extract_layers = extract_layers
        
        # Load base Qwen VL model
        print(f"Loading {model_name}...")
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype="auto",
            device_map="auto"
        )
       
        # Get vision encoder feature dimension
        self.vision_hidden_size = self.base_model.config.vision_config.hidden_size
        
        # Freeze the entire Qwen VIT model
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Frozen all Qwen VIT parameters")


        # Initialize Unified Feature Fusion module
        self.feature_fusion = UnifiedFeatureFusion(
            hidden_size=self.vision_hidden_size,
            num_scales=len(extract_layers),
            num_heads=8,
            dropout=0.1
        )
        
        # Initialize learnable task tokens
        # These are the key innovation - each task has its own query token
        self.binary_task_token = nn.Parameter(
            torch.randn(1, 1, self.vision_hidden_size) * 0.02
        )
        self.multiclass_task_token = nn.Parameter(
            torch.randn(1, 1, self.vision_hidden_size) * 0.02
        )
        self.localization_task_token = nn.Parameter(
            torch.randn(1, 1, self.vision_hidden_size) * 0.02
        )
        
        # Initialize simple task heads
        self.detection_head = SimpleTaskHead(
            in_dim=self.vision_hidden_size,
            num_classes=2,  # Real, Fake
            hidden_dim=256,
            dropout=0.1
        )
        
        self.identification_head = SimpleTaskHead(
            in_dim=self.vision_hidden_size,
            num_classes=num_generators,
            hidden_dim=256,
            dropout=0.1
        )
        

        self.localization_head = SpatialTaskHead(
            in_channels=self.vision_hidden_size,
            hidden_dim=256,
            output_size=(448, 448)
        )
        
        # Storage for intermediate features
        self.intermediate_features = {}
        self._register_feature_hooks()
        
        print(f"Initialized DFLIP Profiler with Task Token Architecture")
        print(f"  - Extract layers: {extract_layers}")
        print(f"  - Generator classes: {num_generators}")
        print(f"  - Task tokens: binary, multiclass, localization")
        self._print_trainable_parameters()

    
    def _register_feature_hooks(self):
        """Register forward hooks to extract intermediate layer features."""
        
        def get_hook(name):
            def hook(module, input, output):
                self.intermediate_features[name] = output
            return hook
        
        # Register hooks on specified layers
        if hasattr(self.base_model, 'visual'):
            visual_encoder = self.base_model.visual
            if hasattr(visual_encoder, 'blocks'):
                for idx in self.extract_layers:
                    if idx < len(visual_encoder.blocks):
                        layer = visual_encoder.blocks[idx]
                        layer.register_forward_hook(get_hook(f'layer_{idx}'))
                        print(f"Registered hook on layer {idx}")
                    else:
                        print(f"Warning: Layer {idx} does not exist, skipping")
    
    def _print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")


    def forward(
        self,
        pixel_values: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:

        B = pixel_values.shape[0]
        
        # Clear previous intermediate features
        self.intermediate_features = {}
        
        # Extract vision features using Qwen's vision encoder
        # The hooks will automatically capture intermediate layers
        with torch.no_grad():
            vision_outputs = self.base_model.visual(pixel_values)
        
        # Collect intermediate features from hooks
        multi_scale_features = []
        for idx in self.extract_layers:
            layer_name = f'layer_{idx}'
            if layer_name in self.intermediate_features:
                multi_scale_features.append(self.intermediate_features[layer_name])


        # Expand task tokens to batch size
        binary_token = self.binary_task_token.expand(B, -1, -1)
        multiclass_token = self.multiclass_task_token.expand(B, -1, -1)
        localization_token = self.localization_task_token.expand(B, -1, -1)
        

        binary_features = self.feature_fusion(multi_scale_features, binary_token)  # (B, D)
        multiclass_features = self.feature_fusion(multi_scale_features, multiclass_token)  # (B, D)
        

        last_scale_features = multi_scale_features[-1]  # (B, N, D)
        N, D = last_scale_features.shape[1], last_scale_features.shape[2]
        H = W = int(N ** 0.5)
        spatial_features = last_scale_features.transpose(1, 2).reshape(B, D, H, W)
        
        # Task-specific predictions
        detection_logits = self.detection_head(binary_features)
        identification_logits = self.identification_head(multiclass_features)
        localization_mask = self.localization_head(spatial_features)
        
        outputs = {
            'detection_logits': detection_logits,
            'identification_logits': identification_logits,
            'localization_mask': localization_mask
        }
        
        if return_features:
            outputs['binary_features'] = binary_features
            outputs['multiclass_features'] = multiclass_features
            outputs['spatial_features'] = spatial_features
        
        return outputs
    
    def predict(
        self,
        pixel_values: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict:
        """
        Inference mode prediction with post-processing.
        
        Args:
            pixel_values: (B, C, H, W) input images
            threshold: Detection threshold
        
        Returns:
            Dictionary with predictions:
                - is_fake: (B,) binary predictions
                - fake_probs: (B,) fake probabilities
                - generator_ids: (B,) predicted generator IDs
                - generator_probs: (B, num_generators) generator probabilities
                - forgery_masks: (B, 1, H, W) segmentation masks
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            
            # Detection predictions
            detection_probs = torch.softmax(outputs['detection_logits'], dim=-1)
            fake_probs = detection_probs[:, 1]  # Probability of being fake
            is_fake = (fake_probs > threshold).long()
            
            # Identification predictions
            identification_probs = torch.softmax(outputs['identification_logits'], dim=-1)
            generator_ids = torch.argmax(identification_probs, dim=-1)
            
            # Localization predictions
            forgery_masks = torch.sigmoid(outputs['localization_mask'])
            forgery_masks_binary = (forgery_masks > threshold).float()
        
        predictions = {
            'is_fake': is_fake,
            'fake_probs': fake_probs,
            'generator_ids': generator_ids,
            'generator_probs': identification_probs,
            'forgery_masks': forgery_masks,
            'forgery_masks_binary': forgery_masks_binary
        }
        
        return predictions
    
    def save_weights(self, save_path: str):
        """Save trainable component weights."""
        checkpoint = {
            'feature_fusion': self.feature_fusion.state_dict(),
            'binary_task_token': self.binary_task_token,
            'multiclass_task_token': self.multiclass_task_token,
            'localization_task_token': self.localization_task_token,
            'detection_head': self.detection_head.state_dict(),
            'identification_head': self.identification_head.state_dict(),
            'localization_head': self.localization_head.state_dict(),
            'num_generators': self.num_generators,
            'extract_layers': self.extract_layers
        }
        
        torch.save(checkpoint, f"{save_path}/profiler_weights.pt")
        print(f"Saved Profiler weights to {save_path}/profiler_weights.pt")
    
    def load_weights(self, load_path: str):
        """Load trainable component weights."""
        checkpoint = torch.load(f"{load_path}/profiler_weights.pt", map_location='cpu')
        
        self.feature_fusion.load_state_dict(checkpoint['feature_fusion'])
        self.binary_task_token.data = checkpoint['binary_task_token']
        self.multiclass_task_token.data = checkpoint['multiclass_task_token']
        self.localization_task_token.data = checkpoint['localization_task_token']
        self.detection_head.load_state_dict(checkpoint['detection_head'])
        self.identification_head.load_state_dict(checkpoint['identification_head'])
        self.localization_head.load_state_dict(checkpoint['localization_head'])
        
        print(f"Loaded Profiler weights from {load_path}/profiler_weights.pt")


def create_profiler(config: Dict) -> DFLIPProfiler:
    """
    Factory function to create DFLIPProfiler from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized DFLIPProfiler model
    """
    model_config = config.get('model', {})
    
    profiler = DFLIPProfiler(
        model_name=model_config.get('base_model', 'Qwen/Qwen2.5-VL-7B-Instruct'),
        num_generators=config.get('num_generators', 10),
        extract_layers=model_config.get('extract_layers', [6, 12, 18]),
        device=config.get('hardware', {}).get('device', 'cuda'),
        cache_dir=model_config.get('cache_dir')
    )
    
    return profiler
