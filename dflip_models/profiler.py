import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from pathlib import Path

from peft import get_peft_model, LoraConfig, PeftModel
from .vision_encoder import TimmMultiLevelEncoder


class LocalizationHead(nn.Module):
    """Localization head for pixel-level prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class BayesianEvidenceProfiler(nn.Module):
    """
    Bayesian Hierarchical Evidence Profiler (BHEP).
    Uses Evidential Deep Learning (EDL) to provide uncertainty estimation
    and hierarchical reasoning (Base -> Family -> Version).
    """
    def __init__(self, feature_dim: int, num_families: int, num_versions: int, hierarchy_mask: torch.Tensor):
        super().__init__()
        self.num_families = num_families
        self.num_versions = num_versions
        self.register_buffer('hierarchy_mask', hierarchy_mask)
        
        # ==========================================================
        # 1. Base Branch (The Generalist)
        # ==========================================================
        # Standard MLP for general forgery traces
        self.base_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Output Logit
        )

        # ==========================================================
        # 2. Family Branch (The Specialist - Coarse)
        # ==========================================================
        # Predicts Dirichlet parameters alpha for families
        self.family_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_families),
            nn.Softplus()  # Ensure Evidence >= 0
        )
        
        # Opinion Projection: Map family belief to real/fake score
        self.op_fam = nn.Linear(num_families, 1)

        # ==========================================================
        # 3. Version Branch (The Specialist - Fine)
        # ==========================================================
        # Conditional EDL: Input features + family belief
        self.version_head = nn.Sequential(
            nn.Linear(feature_dim + num_families, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_versions),
            nn.Softplus()  # Ensure Evidence >= 0
        )
        
        # Opinion Projection: Map version belief to real/fake score
        self.op_ver = nn.Linear(num_versions, 1)

    def forward(self, features: torch.Tensor):
        """
        features: (Batch, feature_dim) - Global pooled features
        """
        batch_size = features.size(0)
        device = features.device

        # --- Step 1: Base Branch ---
        logit_base = self.base_head(features)
        # Base uncertainty (fixed neutral prior)
        u_base = torch.full((batch_size, 1), 0.5, device=device)

        # --- Step 2: Family Branch ---
        evidence_fam = self.family_head(features)
        alpha_fam = evidence_fam + 1.0
        
        S_fam = torch.sum(alpha_fam, dim=1, keepdim=True)
        b_fam = alpha_fam / S_fam  # Belief
        u_fam = self.num_families / S_fam  # Uncertainty
        
        logit_fam = self.op_fam(b_fam)

        # --- Step 3: Version Branch ---
        # Conditional Input: Features + Family Belief
        ver_input = torch.cat([features, b_fam], dim=1)
        evidence_ver_raw = self.version_head(ver_input)
        
        # Hierarchical Gating
        # Suppress versions belonging to families with low belief
        gate = torch.matmul(b_fam, self.hierarchy_mask)
        evidence_ver = evidence_ver_raw * gate
        alpha_ver = evidence_ver + 1.0
        
        S_ver = torch.sum(alpha_ver, dim=1, keepdim=True)
        b_ver = alpha_ver / S_ver
        u_ver = self.num_versions / S_ver
        
        logit_ver = self.op_ver(b_ver)

        # --- Step 4: Bayesian Fusion ---
        eps = 1e-5
        uncertainties = torch.cat([u_base, u_fam, u_ver], dim=1)
        
        # Weighting: Inverse of uncertainty
        raw_weights = 1.0 / (uncertainties + eps)
        norm_weights = F.softmax(raw_weights, dim=1)  # (B, 3)
        
        w_base = norm_weights[:, 0:1]
        w_fam  = norm_weights[:, 1:2]
        w_ver  = norm_weights[:, 2:3]
        
        final_logit = (
            w_base * logit_base +
            w_fam  * logit_fam +
            w_ver  * logit_ver
        )

        return {
            "detection_logits": torch.cat([-final_logit, final_logit], dim=1), # For compatibility (B, 2)
            "final_logit": final_logit,   # (B, 1)
            "alpha_fam": alpha_fam,
            "alpha_ver": alpha_ver,
            "weights": norm_weights,
            "uncertainties": uncertainties,
            "b_fam": b_fam,
            "b_ver": b_ver
        }


class DFLIPProfiler(nn.Module):
    """
    DFLIPProfiler with BHEP integration.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        config: Dict = None,
        freeze_encoder: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        
        self.vision_encoder = TimmMultiLevelEncoder(
            model_name=model_name,
            freeze=freeze_encoder,
            pretrained=True
        )
        self.vision_hidden_size = self.vision_encoder.fused_dim

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            self.vision_encoder.backbone = get_peft_model(
                self.vision_encoder.backbone, lora_config
            )
            self.use_lora = True
        else:
            self.use_lora = False

        # BHEP Configuration
        bhep_config = config.get('bhep', {})
        if bhep_config.get('enabled', False):
            self.use_bhep = True
            num_families = bhep_config['num_families']
            num_versions = bhep_config['num_versions']
            hierarchy_list = bhep_config['hierarchy']
            
            # Build hierarchy mask from config list
            # hierarchy_list[i] contains indices of versions belonging to family i
            mask = torch.zeros(num_families, num_versions)
            for fam_idx, ver_indices in enumerate(hierarchy_list):
                for ver_idx in ver_indices:
                    if ver_idx < num_versions:
                        mask[fam_idx, ver_idx] = 1.0
            
            self.bhep_head = BayesianEvidenceProfiler(
                feature_dim=self.vision_hidden_size,
                num_families=num_families,
                num_versions=num_versions,
                hierarchy_mask=mask
            )
        else:
            self.use_bhep = False
            # Legacy simple head
            self.detection_head = nn.Sequential(
                nn.Linear(self.vision_hidden_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 2)
            )

        self.localization_head = LocalizationHead(
            input_dim=self.vision_hidden_size,
            hidden_dim=256
        )

        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")

    def forward(self, pixel_values: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        B, C, H, W = pixel_values.shape
        pooled_features = self.vision_encoder(pixel_values)

        outputs = {}
        
        if self.use_bhep:
            bhep_outputs = self.bhep_head(pooled_features)
            outputs.update(bhep_outputs)
            # Add identification logits for compatibility (using version belief)
            # Map version belief to num_versions logits (mocking)
            outputs["identification_logits"] = torch.log(bhep_outputs["b_ver"] + 1e-9)
        else:
            detection_logits = self.detection_head(pooled_features)
            outputs["detection_logits"] = detection_logits
            # Mock identification logits for legacy
            outputs["identification_logits"] = torch.zeros(B, 10).to(pixel_values.device)

        # Localization is shared
        localization_logits = self.localization_head(pooled_features)
        localization_mask = localization_logits.view(B, 1, 1, 1)
        localization_mask = F.interpolate(
            localization_mask, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        outputs["localization_mask"] = localization_mask

        if return_features:
            outputs["pooled_features"] = pooled_features

        return outputs


class BHEPLoss(nn.Module):
    def __init__(self, lambda_edl=0.1, lambda_kl=0.5):
        super().__init__()
        self.lambda_edl = lambda_edl
        self.lambda_kl = lambda_kl
        self.bce = nn.BCEWithLogitsLoss()
        self.localization_loss_fn = nn.BCEWithLogitsLoss()

    def edl_loss(self, alpha, y):
        """EDL-MSE Loss for labeled fake samples."""
        S = torch.sum(alpha, dim=1, keepdim=True)
        b = alpha / S 
        
        y_onehot = F.one_hot(y, num_classes=alpha.shape[1]).float()
        
        loss_mse = torch.sum((y_onehot - b) ** 2, dim=1, keepdim=True)
        loss_var = torch.sum(b * (1 - b), dim=1, keepdim=True) / (S + 1)
        
        return torch.mean(loss_mse + loss_var)

    def max_entropy_loss(self, alpha):
        """KL Divergence to Uniform for real/unlabeled samples."""
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        log_prob = torch.log(prob + 1e-8)
        
        num_classes = alpha.shape[1]
        target_prob = torch.ones_like(prob) / num_classes
        
        loss_kl = F.kl_div(log_prob, target_prob, reduction='batchmean')
        return loss_kl

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Detection Loss (Main Task)
        label_det = targets['is_fake'].float().view(-1, 1)
        loss_det = self.bce(outputs['final_logit'], label_det)
        
        # 2. Localization Loss
        loss_loc = 0.0
        if 'mask_labels' in targets:
             loss_loc = self.localization_loss_fn(
                 outputs['localization_mask'], 
                 targets['mask_labels'].float()
             )

        # 3. BHEP Specific Losses
        loss_fam = 0.0
        loss_ver = 0.0
        
        # Check if we have EDL outputs
        if 'alpha_fam' in outputs:
            is_fake = (label_det.squeeze() == 1)
            is_real = (label_det.squeeze() == 0)
            
            # --- Handle Fake Samples ---
            if is_fake.sum() > 0:
                # If we have fine-grained labels, use EDL loss
                if 'label_fam' in targets and 'label_ver' in targets:
                    alpha_fam = outputs['alpha_fam'][is_fake]
                    target_fam = targets['label_fam'][is_fake]
                    loss_fam += self.edl_loss(alpha_fam, target_fam)
                    
                    alpha_ver = outputs['alpha_ver'][is_fake]
                    target_ver = targets['label_ver'][is_fake]
                    loss_ver += self.edl_loss(alpha_ver, target_ver)
                else:
                    # If no fine-grained labels (current dataset), 
                    # we can't supervise family/version directly for fake.
                    # Optionally, we could treat them as OOD or just ignore.
                    # Here we ignore to avoid noise.
                    pass

            # --- Handle Real Samples ---
            # Real samples should have high uncertainty (Max Entropy) in generator branches
            if is_real.sum() > 0:
                alpha_fam_real = outputs['alpha_fam'][is_real]
                loss_fam += self.max_entropy_loss(alpha_fam_real)
                
                alpha_ver_real = outputs['alpha_ver'][is_real]
                loss_ver += self.max_entropy_loss(alpha_ver_real)

        total_loss = (
            loss_det + 
            loss_loc + 
            self.lambda_edl * (loss_fam + loss_ver) * 0.0 # Temporarily disable EDL weight if no labels
            + self.lambda_kl * (loss_fam + loss_ver)      # Apply KL for real samples
        )
        
        # If we have labels, enable EDL weight
        if 'label_fam' in targets:
             total_loss += self.lambda_edl * (loss_fam + loss_ver)

        return {
            "loss": total_loss,
            "detection_loss": loss_det.item(),
            "localization_loss": loss_loc.item() if isinstance(loss_loc, torch.Tensor) else loss_loc,
            "bhep_fam_loss": loss_fam.item() if isinstance(loss_fam, torch.Tensor) else loss_fam,
            "bhep_ver_loss": loss_ver.item() if isinstance(loss_ver, torch.Tensor) else loss_ver
        }


def create_profiler(config: Dict) -> DFLIPProfiler:
    model_config = config.get("model", {})
    profiler = DFLIPProfiler(
        model_name=model_config.get("base_model", "dinov2_vitb14"),
        config=config,  # Pass full config for BHEP
        freeze_encoder=model_config.get("freeze_encoder", True),
    )
    return profiler
