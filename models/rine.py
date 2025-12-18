import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict

import clip


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class RINEModel(nn.Module):
    def __init__(self, backbone0="ViT-L/14", backbone1=768, nproj=2, proj_dim=512, **kwargs):
        super().__init__()
        
        # Store configuration for debugging
        self.backbone0 = backbone0
        self.backbone1 = backbone1
        self.nproj = nproj
        self.proj_dim = proj_dim

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone0, device="cpu")
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone1 if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]

        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g

        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return {'logits': p, 'z': z}


class RINEClassifier(nn.Module):
    """
    RINE (Residual Intermediate Network Embedding) binary classifier adapted for the DFLIP framework.
    Uses CLIP intermediate features with attention mechanism for fake detection.
    """
    def __init__(
        self,
        backbone0: str = "ViT-L/14",
        backbone1: int = 768,
        nproj: int = 2,
        proj_dim: int = 512,
        num_classes: int = 2,  # Binary classification: real/fake
        verbose: bool = True,
    ):
        super().__init__()
        
        self.backbone0 = backbone0
        self.backbone1 = backbone1
        self.nproj = nproj
        self.proj_dim = proj_dim
        
        # Create RINE model
        self.model = RINEModel(
            backbone0=backbone0,
            backbone1=backbone1,
            nproj=nproj,
            proj_dim=proj_dim
        )
        
        # Add final classification layer if needed for multi-class
        if num_classes > 1:
            self.classifier = nn.Linear(1, num_classes)
        else:
            self.classifier = None
        
        if verbose:
            self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"RINE Binary Classifier ({self.backbone0}):")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) tensor of images
            
        Returns:
            dict with keys:
                - logits: (B, num_classes) classification logits
                - features: (B, feature_dim) extracted features
        """
        # Forward through RINE model
        outputs = self.model(pixel_values)
        
        logits = outputs['logits']
        features = outputs['z']
        
        # Apply final classifier if multi-class
        if self.classifier is not None:
            logits = self.classifier(logits)
        
        # Return in the expected format for the framework
        return {
            "logits": logits,
            "features": features,
        }


def create_rine(config: Dict, verbose: bool = True) -> RINEClassifier:
    """Create a RINE binary classifier."""
    model_config = config.get("model", {})
    
    # Extract backbone configuration
    backbone0 = model_config.get("backbone0", "ViT-L/14")
    backbone1 = model_config.get("backbone1", 768)
    nproj = model_config.get("nproj", 2)
    proj_dim = model_config.get("proj_dim", 512)
    
    model = RINEClassifier(
        backbone0=backbone0,
        backbone1=backbone1,
        nproj=nproj,
        proj_dim=proj_dim,
        num_classes=2,  # Binary classification
        verbose=verbose,
    )
    
    if verbose:
        print(f"Created RINE Binary Classifier for real/fake detection")
        print(f"  - Backbone: {backbone0}")
        print(f"  - Feature dim: {backbone1}")
        print(f"  - Projection layers: {nproj}")
        print(f"  - Projection dim: {proj_dim}")
    
    return model