#!/usr/bin/env python3
"""Train DFLIP Profiler (BHEP, 单阶段训练脚本)."""
import yaml
import argparse
from typing import Dict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel

from utils.transforms import build_train_val_transforms

from utils.seed import seed_everything

from models import create_profiler
from models.profiler import BHEPLoss
from dataset import create_profiling_dataloaders
from utils.training_funcs import (
    create_optimizer,
    create_scheduler,
    train_epoch,
    evaluate,
    save_checkpoint,
)
from utils.logger import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFLIP Profiler (BHEP)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dflip_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps per epoch (mainly for debug)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config["hardware"]["seed"])
    logger = create_logger(config)

    train_transform, val_transform = build_train_val_transforms(config)

    print("Creating datasets...")
    train_loader, val_loader = create_profiling_dataloaders(config, train_transform, val_transform)
    print("Creating model...")
    model = create_profiler(config)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.device = device

    print("Using BHEP Loss...")
    loss_fn = BHEPLoss(
        lambda_edl=config["bhep"].get("lambda_edl", 0.1),
        lambda_kl=config["bhep"].get("lambda_kl", 0.5),
        lambda_aux=config["bhep"].get("lambda_aux", 0.1),  # Auxiliary loss weight
    )

    train_config = config["training"]
    num_training_steps = (
        len(train_loader) // train_config["gradient_accumulation_steps"] * train_config["num_epochs"]
    )

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, num_training_steps, config)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        # 简化版：优先尝试作为 LoRA adapter 目录加载
        try:
            model.vision_encoder.backbone = PeftModel.from_pretrained(
                model.vision_encoder.backbone, args.resume
            )
            print("Loaded LoRA adapter from", args.resume)
        except Exception as e:  # noqa: BLE001
            print("Could not load LoRA adapter, trying full model state_dict.")
            checkpoint = torch.load(args.resume, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, train_config["num_epochs"]):
        print("\n" + "=" * 50)
        print(f"Epoch {epoch + 1}/{train_config['num_epochs']}")
        print("=" * 50)

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            epoch,
            config,
            args.debug,
            args.max_steps,
            logger=logger,
        )
        print(f"\nTrain metrics: {train_metrics}")

        if not args.debug:
            val_metrics = evaluate(model, val_loader, loss_fn)
            print(f"Val metrics: {val_metrics}")

            if logger.is_active():
                logger.log(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/detection_loss": val_metrics["detection_loss"],
                        "val/detection_accuracy": val_metrics["detection_accuracy"],
                        "val/family_accuracy": val_metrics["family_accuracy"],
                        "val/version_accuracy": val_metrics["version_accuracy"],
                        "epoch": epoch,
                    }
                )

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                print(f"New best model! Val loss: {best_val_loss:.4f}")

            save_checkpoint(model, optimizer, scheduler, epoch, config, is_best)
        else:
            # debug 模式下每个 epoch 也保存一份，方便快速检查
            save_checkpoint(model, optimizer, scheduler, epoch, config, False)

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config['training']['output_dir']}")
    print("=" * 50)

    if logger is not None and logger.is_active():
        logger.finish()


if __name__ == "__main__":
    main()
