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
