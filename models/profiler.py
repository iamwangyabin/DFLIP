import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from peft import get_peft_model, LoraConfig, PeftModel
from .vision_encoder import TimmMultiLevelEncoder

class BayesianEvidenceClassifier(nn.Module):
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

        # Standard MLP for general forgery traces
        self.base_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Output Logit
        )

        # Predicts Dirichlet parameters alpha for families
        self.family_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_families),
            nn.Softplus()
        )
        
        # Opinion Projection: Map family belief to real/fake score
        self.op_fam = nn.Linear(num_families, 1)

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
        batch_size = features.size(0)
        device = features.device

        logit_base = self.base_head(features)
        u_base = torch.full((batch_size, 1), 0.5, device=device)

        evidence_fam = self.family_head(features)
        alpha_fam = evidence_fam + 1.0
        S_fam = torch.sum(alpha_fam, dim=1, keepdim=True)
        b_fam = alpha_fam / S_fam  # Belief
        u_fam = self.num_families / S_fam  # Uncertainty
        
        logit_fam = self.op_fam(b_fam)

        ver_input = torch.cat([features, b_fam], dim=1)
        evidence_ver_raw = self.version_head(ver_input)
        gate = torch.matmul(b_fam, self.hierarchy_mask)
        evidence_ver = evidence_ver_raw * gate
        alpha_ver = evidence_ver + 1.0
        S_ver = torch.sum(alpha_ver, dim=1, keepdim=True)
        b_ver = alpha_ver / S_ver
        u_ver = self.num_versions / S_ver
        
        logit_ver = self.op_ver(b_ver)

        eps = 1e-5
        uncertainties = torch.cat([u_base, u_fam, u_ver], dim=1)
        
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

        # Generate family and version logits for direct multi-class classification
        family_logits = torch.log(b_fam + 1e-9)   # (B, num_families)
        version_logits = torch.log(b_ver + 1e-9)  # (B, num_versions)

        return {
            "detection_logits": torch.cat([-final_logit, final_logit], dim=1), # For compatibility (B, 2)
            "final_logit": final_logit,   # (B, 1)
            "family_logits": family_logits,
            "version_logits": version_logits,
            "alpha_fam": alpha_fam,
            "alpha_ver": alpha_ver,
            "b_fam": b_fam,
            "b_ver": b_ver,
            "weights": norm_weights,
            "uncertainties": uncertainties,
        }


class DFLIPProfiler(nn.Module):
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

        self.bhep_head = BayesianEvidenceClassifier(
            feature_dim=self.vision_hidden_size,
            num_families=num_families,
            num_versions=num_versions,
            hierarchy_mask=mask
        )

        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")

    def forward(self, pixel_values: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        pooled_features = self.vision_encoder(pixel_values)
        outputs = self.bhep_head(pooled_features)
        return outputs



class BHEPLoss(nn.Module):
    def __init__(self, lambda_edl=0.1, lambda_kl=0.5):
        super().__init__()
        self.lambda_edl = lambda_edl
        self.lambda_kl = lambda_kl
        self.bce = nn.BCEWithLogitsLoss()

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
        
        label_det = targets['is_fake'].float().view(-1, 1)

        loss_det = self.bce(outputs['final_logit'], label_det)

        loss_fam = 0.0
        loss_ver = 0.0

        is_fake = (label_det.squeeze() == 1)
        is_real = (label_det.squeeze() == 0)

        if is_fake.sum() > 0:
            alpha_fam = outputs['alpha_fam'][is_fake]
            target_fam = targets['label_fam'][is_fake]
            loss_fam += self.edl_loss(alpha_fam, target_fam)

            alpha_ver = outputs['alpha_ver'][is_fake]
            target_ver = targets['label_ver'][is_fake]
            loss_ver += self.edl_loss(alpha_ver, target_ver)

        if is_real.sum() > 0:
            alpha_fam_real = outputs['alpha_fam'][is_real]
            loss_fam += self.max_entropy_loss(alpha_fam_real)

            alpha_ver_real = outputs['alpha_ver'][is_real]
            loss_ver += self.max_entropy_loss(alpha_ver_real)

        total_loss = (loss_det + self.lambda_edl * (loss_fam + loss_ver))

        return {
            "loss": total_loss,
            "detection_loss": loss_det.item(),
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
