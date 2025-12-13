"""Training utilities for DFLIP Profiler (BHEP, single-stage training).

This module contains optimizer / scheduler creation, one-epoch training,
validation, and checkpoint saving logic, so they can be reused from
scripts or notebooks.
"""
from typing import Dict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm  # local import to keep script dependency light

from models import DFLIPProfiler
from utils.logger import BaseLogger


def create_optimizer(model: nn.Module, config: Dict):
    """Create optimizer with separate parameter groups for LoRA and others."""
    train_config = config["training"]

    lora_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name.lower():
            lora_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": train_config["learning_rate"]})
    if other_params:
        # 头部稍微大学习率也可以按需调整
        param_groups.append({"params": other_params, "lr": train_config["learning_rate"] * 2})

    optimizer = AdamW(
        param_groups,
        weight_decay=train_config["weight_decay"],
        betas=(train_config["adam_beta1"], train_config["adam_beta2"]),
        eps=train_config["adam_epsilon"],
    )
    return optimizer


def create_scheduler(optimizer, num_training_steps: int, config: Dict):
    """Create learning rate scheduler (currently cosine with warmup)."""
    train_config = config["training"]
    num_warmup_steps = int(num_training_steps * train_config.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


def train_epoch(
    model: DFLIPProfiler,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    loss_fn: nn.Module,
    epoch: int,
    config: Dict,
    debug: bool = False,
    max_steps: int | None = None,
    logger: BaseLogger | None = None,
):
    """Train for a single epoch."""

    model.train()
    train_config = config["training"]
    grad_accum = train_config["gradient_accumulation_steps"]

    total_loss = 0.0
    total_det_loss = 0.0
    total_bhep_fam_loss = 0.0
    total_bhep_ver_loss = 0.0
    
    # Diagnostic accumulators for version analysis
    total_gate_mean = 0.0
    total_evidence_ver_ratio = 0.0
    total_alpha_ver_mean = 0.0
    diag_count = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        if max_steps is not None and step >= max_steps:
            break

        pixel_values = batch["pixel_values"].to(model.device)
        is_fake = batch["is_fake"].to(model.device)

        targets = {"is_fake": is_fake}
        if "family_ids" in batch:
            targets["label_fam"] = batch["family_ids"].to(model.device)
        if "version_ids" in batch:
            targets["label_ver"] = batch["version_ids"].to(model.device)

        outputs = model(pixel_values)
        loss_dict = loss_fn(outputs, targets)

        loss = loss_dict["loss"] / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 统计指标（全部用 float）
        total_loss += loss_dict["loss"].item()
        total_det_loss += loss_dict["detection_loss"]
        if "bhep_fam_loss" in loss_dict:
            total_bhep_fam_loss += loss_dict["bhep_fam_loss"]
        if "bhep_ver_loss" in loss_dict:
            total_bhep_ver_loss += loss_dict["bhep_ver_loss"]
        
        # Collect diagnostic statistics
        if "diagnostics" in outputs:
            diag = outputs["diagnostics"]
            total_gate_mean += diag["gate_mean"]
            total_evidence_ver_ratio += diag["evidence_ver_ratio"]
            total_alpha_ver_mean += diag["alpha_ver_mean"]
            diag_count += 1

        pbar_dict = {
            "loss": f"{loss_dict['loss'].item():.4f}",
            "det": f"{loss_dict['detection_loss']:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if "bhep_fam_loss" in loss_dict:
            pbar_dict["fam"] = f"{loss_dict['bhep_fam_loss']:.4f}"
        if "bhep_ver_loss" in loss_dict:
            pbar_dict["ver"] = f"{loss_dict['bhep_ver_loss']:.4f}"
        progress_bar.set_postfix(pbar_dict)

        if logger is not None and logger.is_active() and (step + 1) % train_config["logging_steps"] == 0:
            log_dict = {
                "train/loss": loss_dict["loss"].item(),
                "train/detection_loss": loss_dict["detection_loss"],
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": step,
            }
            if "bhep_fam_loss" in loss_dict:
                log_dict["train/bhep_fam_loss"] = loss_dict["bhep_fam_loss"]
            if "bhep_ver_loss" in loss_dict:
                log_dict["train/bhep_ver_loss"] = loss_dict["bhep_ver_loss"]
            
            # Add diagnostic information to logs
            if "diagnostics" in outputs:
                diag = outputs["diagnostics"]
                log_dict.update({
                    "diagnostics/gate_mean": diag["gate_mean"],
                    "diagnostics/gate_std": diag["gate_std"],
                    "diagnostics/evidence_ver_raw_mean": diag["evidence_ver_raw_mean"],
                    "diagnostics/evidence_ver_mean": diag["evidence_ver_mean"],
                    "diagnostics/evidence_ver_ratio": diag["evidence_ver_ratio"],
                    "diagnostics/alpha_ver_mean": diag["alpha_ver_mean"],
                    "diagnostics/alpha_ver_std": diag["alpha_ver_std"],
                    "diagnostics/alpha_ver_max": diag["alpha_ver_max"],
                    "diagnostics/S_ver_mean": diag["S_ver_mean"],
                })
            
            logger.log(log_dict)

    num_steps = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    num_steps = max(num_steps, 1)
    metrics = {
        "loss": total_loss / num_steps,
        "detection_loss": total_det_loss / num_steps,
    }
    if total_bhep_fam_loss > 0:
        metrics["bhep_fam_loss"] = total_bhep_fam_loss / num_steps
    if total_bhep_ver_loss > 0:
        metrics["bhep_ver_loss"] = total_bhep_ver_loss / num_steps
    
    # Add diagnostic summary to metrics
    if diag_count > 0:
        metrics.update({
            "diag_gate_mean": total_gate_mean / diag_count,
            "diag_evidence_ver_ratio": total_evidence_ver_ratio / diag_count,
            "diag_alpha_ver_mean": total_alpha_ver_mean / diag_count,
        })
        print(f"\n[DIAGNOSTICS] Gate mean: {metrics['diag_gate_mean']:.4f}, "
              f"Evidence ratio: {metrics['diag_evidence_ver_ratio']:.4f}, "
              f"Alpha_ver mean: {metrics['diag_alpha_ver_mean']:.4f}")
    
    return metrics


@torch.no_grad()
def evaluate(model: DFLIPProfiler, dataloader: DataLoader, loss_fn: nn.Module):
    """Evaluate on validation set: detection + family / version classification."""
    from tqdm import tqdm  # local import to keep script dependency light

    model.eval()

    total_loss = 0.0
    total_det_loss = 0.0
    correct_det = 0
    total_samples = 0

    # family / version 多分类统计（只在 fake 且有有效 label 的样本上计算）
    correct_fam = 0
    total_fam = 0
    correct_ver = 0
    total_ver = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(model.device)
        is_fake = batch["is_fake"].to(model.device)

        targets = {"is_fake": is_fake}
        if "family_ids" in batch:
            targets["label_fam"] = batch["family_ids"].to(model.device)
        if "version_ids" in batch:
            targets["label_ver"] = batch["version_ids"].to(model.device)

        outputs = model(pixel_values)
        loss_dict = loss_fn(outputs, targets)

        total_loss += loss_dict["loss"].item()
        total_det_loss += loss_dict["detection_loss"]

        # 二分类真实/伪造
        if "final_logit" in outputs:
            preds_det = (torch.sigmoid(outputs["final_logit"]) > 0.5).long().squeeze()
        else:
            preds_det = torch.argmax(outputs["detection_logits"], dim=1)

        correct_det += (preds_det == is_fake).sum().item()
        total_samples += is_fake.numel()

        # family 多分类，只统计 fake 且有有效 family_id 的样本
        if "family_logits" in outputs and "family_ids" in batch:
            fam_labels = batch["family_ids"].to(model.device)
            mask_fam = (is_fake == 1) & (fam_labels >= 0)
            if mask_fam.any():
                fam_logits = outputs["family_logits"][mask_fam]
                fam_labels_masked = fam_labels[mask_fam]
                fam_preds = torch.argmax(fam_logits, dim=1)
                correct_fam += (fam_preds == fam_labels_masked).sum().item()
                total_fam += fam_labels_masked.numel()

        # version 多分类，只统计 fake 且有有效 version_id 的样本
        if "version_logits" in outputs and "version_ids" in batch:
            ver_labels = batch["version_ids"].to(model.device)
            mask_ver = (is_fake == 1) & (ver_labels >= 0)
            if mask_ver.any():
                ver_logits = outputs["version_logits"][mask_ver]
                ver_labels_masked = ver_labels[mask_ver]
                ver_preds = torch.argmax(ver_logits, dim=1)
                correct_ver += (ver_preds == ver_labels_masked).sum().item()
                total_ver += ver_labels_masked.numel()

    num_batches = max(len(dataloader), 1)
    total_samples = max(total_samples, 1)

    family_accuracy = correct_fam / max(total_fam, 1) if total_fam > 0 else 0.0
    version_accuracy = correct_ver / max(total_ver, 1) if total_ver > 0 else 0.0

    return {
        "loss": total_loss / num_batches,
        "detection_loss": total_det_loss / num_batches,
        "detection_accuracy": correct_det / total_samples,
        "family_accuracy": family_accuracy,
        "version_accuracy": version_accuracy,
    }


def save_checkpoint(
    model: DFLIPProfiler,
    optimizer,
    scheduler,
    epoch: int,
    config: Dict,
    is_best: bool = False,
):
    """Save model / optimizer / scheduler states for the given epoch.

    If LoRA adapters are used, we first try to save them via
    ``model.vision_encoder.backbone.save_pretrained``; otherwise, fall back
    to full model state_dict.
    """
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_dir = output_dir / f"checkpoint-epoch-{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.vision_encoder.backbone.save_pretrained(str(epoch_dir))
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not save LoRA weights: {e}, saving full model state_dict instead.")
        torch.save(model.state_dict(), epoch_dir / "model.pt")

    trainer_state = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(trainer_state, output_dir / f"trainer_state-epoch-{epoch}.pt")

    if is_best:
        best_dir = output_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.vision_encoder.backbone.save_pretrained(str(best_dir))
        except Exception:
            torch.save(model.state_dict(), best_dir / "model.pt")
        torch.save(trainer_state, output_dir / "best_trainer_state.pt")

    print(f"Saved checkpoint to {output_dir}")


# ============================================================================
# Baseline Classifier Training Functions
# ============================================================================

def train_epoch_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    criterions: Dict[str, nn.Module],
    epoch: int,
    config: Dict,
    loss_weights: Dict[str, float],
    debug: bool = False,
    max_steps: int | None = None,
    logger: BaseLogger | None = None,
):
    """Train baseline classifier for a single epoch."""
    
    model.train()
    train_config = config["training"]
    grad_accum = train_config["gradient_accumulation_steps"]
    
    total_loss = 0.0
    total_det_loss = 0.0
    total_fam_loss = 0.0
    total_ver_loss = 0.0
    
    det_correct = 0
    fam_correct = 0
    ver_correct = 0
    total_samples = 0
    fake_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        if max_steps is not None and step >= max_steps:
            break
        
        pixel_values = batch["pixel_values"].to(model.device)
        is_fake = batch["is_fake"].to(model.device)
        family_ids = batch["family_ids"].to(model.device)
        version_ids = batch["version_ids"].to(model.device)
        
        # Forward pass
        outputs = model(pixel_values)
        
        # Compute losses for three heads
        det_loss = criterions['detection'](outputs['detection_logits'], is_fake)
        
        # Only compute family/version loss for fake samples
        fake_mask = is_fake == 1
        if fake_mask.sum() > 0:
            fam_loss = criterions['family'](
                outputs['family_logits'][fake_mask],
                family_ids[fake_mask]
            )
            ver_loss = criterions['version'](
                outputs['version_logits'][fake_mask],
                version_ids[fake_mask]
            )
        else:
            fam_loss = torch.tensor(0.0, device=model.device)
            ver_loss = torch.tensor(0.0, device=model.device)
        
        # Combined loss with weights
        loss = (loss_weights['detection'] * det_loss +
                loss_weights['family'] * fam_loss +
                loss_weights['version'] * ver_loss)
        
        loss = loss / grad_accum
        loss.backward()
        
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Statistics
        total_loss += (loss.item() * grad_accum)
        total_det_loss += det_loss.item()
        total_fam_loss += fam_loss.item() if isinstance(fam_loss, torch.Tensor) else fam_loss
        total_ver_loss += ver_loss.item() if isinstance(ver_loss, torch.Tensor) else ver_loss
        
        # Accuracies
        _, det_pred = outputs['detection_logits'].max(1)
        det_correct += det_pred.eq(is_fake).sum().item()
        
        if fake_mask.sum() > 0:
            _, fam_pred = outputs['family_logits'][fake_mask].max(1)
            fam_correct += fam_pred.eq(family_ids[fake_mask]).sum().item()
            
            _, ver_pred = outputs['version_logits'][fake_mask].max(1)
            ver_correct += ver_pred.eq(version_ids[fake_mask]).sum().item()
            
            fake_samples += fake_mask.sum().item()
        
        total_samples += is_fake.size(0)
        
        # Update progress bar
        pbar_dict = {
            'loss': f'{loss.item() * grad_accum:.4f}',
            'det_acc': f'{100.*det_correct/total_samples:.1f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
        }
        if fake_samples > 0:
            pbar_dict['fam_acc'] = f'{100.*fam_correct/fake_samples:.1f}%'
        progress_bar.set_postfix(pbar_dict)
        
        # Logging
        if logger is not None and logger.is_active() and (step + 1) % train_config["logging_steps"] == 0:
            log_dict = {
                "train/loss": loss.item() * grad_accum,
                "train/det_loss": det_loss.item(),
                "train/fam_loss": total_fam_loss / (step + 1),
                "train/ver_loss": total_ver_loss / (step + 1),
                "train/det_acc": det_correct / total_samples,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": step,
            }
            if fake_samples > 0:
                log_dict["train/fam_acc"] = fam_correct / fake_samples
                log_dict["train/ver_acc"] = ver_correct / fake_samples
            logger.log(log_dict)
    
    num_steps = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    num_steps = max(num_steps, 1)
    
    metrics = {
        'loss': total_loss / num_steps,
        'det_loss': total_det_loss / num_steps,
        'fam_loss': total_fam_loss / num_steps,
        'ver_loss': total_ver_loss / num_steps,
        'det_acc': det_correct / total_samples if total_samples > 0 else 0.0,
        'fam_acc': fam_correct / fake_samples if fake_samples > 0 else 0.0,
        'ver_acc': ver_correct / fake_samples if fake_samples > 0 else 0.0,
    }
    
    return metrics


@torch.no_grad()
def evaluate_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    criterions: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
):
    """Evaluate baseline classifier on validation set."""
    
    model.eval()
    
    total_loss = 0.0
    total_det_loss = 0.0
    total_fam_loss = 0.0
    total_ver_loss = 0.0
    
    det_correct = 0
    fam_correct = 0
    ver_correct = 0
    total_samples = 0
    fake_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(model.device)
        is_fake = batch["is_fake"].to(model.device)
        family_ids = batch["family_ids"].to(model.device)
        version_ids = batch["version_ids"].to(model.device)
        
        # Forward pass
        outputs = model(pixel_values)
        
        # Compute losses
        det_loss = criterions['detection'](outputs['detection_logits'], is_fake)
        
        fake_mask = is_fake == 1
        if fake_mask.sum() > 0:
            fam_loss = criterions['family'](
                outputs['family_logits'][fake_mask],
                family_ids[fake_mask]
            )
            ver_loss = criterions['version'](
                outputs['version_logits'][fake_mask],
                version_ids[fake_mask]
            )
        else:
            fam_loss = torch.tensor(0.0, device=model.device)
            ver_loss = torch.tensor(0.0, device=model.device)
        
        loss = (loss_weights['detection'] * det_loss +
                loss_weights['family'] * fam_loss +
                loss_weights['version'] * ver_loss)
        
        # Statistics
        total_loss += loss.item()
        total_det_loss += det_loss.item()
        total_fam_loss += fam_loss.item() if isinstance(fam_loss, torch.Tensor) else fam_loss
        total_ver_loss += ver_loss.item() if isinstance(ver_loss, torch.Tensor) else ver_loss
        
        # Accuracies
        _, det_pred = outputs['detection_logits'].max(1)
        det_correct += det_pred.eq(is_fake).sum().item()
        
        if fake_mask.sum() > 0:
            fake_samples += fake_mask.sum().item()
            _, fam_pred = outputs['family_logits'][fake_mask].max(1)
            fam_correct += fam_pred.eq(family_ids[fake_mask]).sum().item()
            
            _, ver_pred = outputs['version_logits'][fake_mask].max(1)
            ver_correct += ver_pred.eq(version_ids[fake_mask]).sum().item()
        
        total_samples += is_fake.size(0)
    
    num_batches = max(len(dataloader), 1)
    
    return {
        'loss': total_loss / num_batches,
        'det_loss': total_det_loss / num_batches,
        'fam_loss': total_fam_loss / num_batches,
        'ver_loss': total_ver_loss / num_batches,
        'det_acc': det_correct / total_samples if total_samples > 0 else 0.0,
        'fam_acc': fam_correct / fake_samples if fake_samples > 0 else 0.0,
        'ver_acc': ver_correct / fake_samples if fake_samples > 0 else 0.0,
    }


def save_checkpoint_baseline(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    config: Dict,
    is_best: bool = False,
):
    """Save baseline model checkpoint."""
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    # Save epoch checkpoint
    torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")
    
    # Save best model if applicable
    if is_best:
        torch.save(checkpoint, output_dir / "best_model.pt")
        print(f"Saved best model to {output_dir / 'best_model.pt'}")
    
    print(f"Saved checkpoint to {output_dir}")
