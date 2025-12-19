#!/usr/bin/env python3
"""Train Family Classifier with single-node multi-GPU (DDP).

This script trains a simple family classifier with a single classification head
for family classification only. Real images (family_id = -1) are ignored during training.

Usage examples (single node with 4 GPUs):

    torchrun --nproc_per_node=4 scripts/train_family_classifier.py \
        --config configs/family_classifier_config.yaml

    # Or single GPU:
    python scripts/train_family_classifier.py --config configs/family_classifier_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.transforms import build_train_val_transforms  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from models.family_classifier import create_family_classifier, FamilyClassifierLoss  # noqa: E402
from dataset import create_profiling_dataloaders_ddp  # noqa: E402
from utils.training_funcs import (  # noqa: E402
    create_optimizer,
    create_scheduler,
    train_epoch_family_classifier,
    evaluate_family_classifier,
    save_checkpoint_family_classifier,
)
from utils.logger import create_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Family Classifier (DDP)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/family_classifier_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (few steps, still DDP)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps per epoch (mainly for debug)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by torchrun (do not set manually)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed(args):
    """Initialize torch.distributed for single-node multi-GPU."""
    # Check if distributed training is requested
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = args.local_rank
        if local_rank < 0:
            # Single GPU mode
            return -1, 0, 1
    
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = local_rank
    
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = torch.cuda.device_count()
    
    # Only initialize distributed if world_size > 1
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    local_rank, rank, world_size = setup_distributed(args)
    is_main_process = rank == 0
    use_ddp = world_size > 1
    
    if is_main_process:
        if use_ddp:
            print(f"Running DDP training on {world_size} GPUs")
        else:
            print("Running single GPU training")
    
    # Set random seed (add rank for different data augmentation)
    seed_everything(config["hardware"]["seed"] + rank)
    
    # Only create logger on main process
    logger = create_logger(config) if is_main_process else None
    
    if is_main_process:
        print("Building transforms...")
    train_transform, val_transform = build_train_val_transforms(config)
    
    if is_main_process:
        print("Creating datasets...")
    
    if use_ddp:
        train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_profiling_dataloaders_ddp(
            config, train_transform, val_transform, world_size=world_size, rank=rank
        )
    else:
        # Single GPU: use regular dataloaders
        from dataset import create_profiling_dataloaders
        train_loader, val_loader, test_loader = create_profiling_dataloaders(
            config, train_transform, val_transform
        )
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    if is_main_process:
        print("Creating family classifier model...")
    
    model = create_family_classifier(config, verbose=is_main_process)
    
    # Setup device
    if use_ddp:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.device = device
    
    # Wrap with DDP if using distributed training
    if use_ddp:
        # Add parameter usage diagnostics (only on main process)
        if is_main_process:
            print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # All parameters are used - no need for extra traversal
        )
    
    if is_main_process:
        print("Setting up loss function...")
    
    # Setup loss function
    criterion = FamilyClassifierLoss()
    
    # Calculate training steps
    train_config = config["training"]
    num_training_steps = (
        len(train_loader) // train_config["gradient_accumulation_steps"] 
        * train_config["num_epochs"]
    )
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, num_training_steps, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and is_main_process:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        if use_ddp:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Broadcast model parameters to all ranks
    if use_ddp:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, train_config["num_epochs"]):
        # Set epoch for distributed sampler
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print("\n" + "=" * 50)
            print(f"Epoch {epoch + 1}/{train_config['num_epochs']}")
            print("=" * 50)
        
        # Training
        train_metrics = train_epoch_family_classifier(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            epoch,
            config,
            args.debug,
            args.max_steps,
            logger=logger,
        )
        
        # Only main process prints and logs
        if is_main_process:
            print(f"\nTrain metrics: {train_metrics}")
        
        # Validation and test evaluation (all processes participate to avoid NCCL timeout)
        if not args.debug:
            # Unwrap DDP model for evaluation to avoid synchronization issues
            eval_model = model.module if use_ddp else model
            
            # All processes run evaluation (each on their data shard via DistributedSampler)
            # This prevents NCCL timeout by keeping all processes busy
            val_metrics = evaluate_family_classifier(
                eval_model,
                val_loader,
                criterion,
                use_ddp=use_ddp,  # Enable aggregation to get complete metrics
            )
            
            # Test evaluation (for observation only)
            test_metrics = evaluate_family_classifier(
                eval_model,
                test_loader,
                criterion,
                use_ddp=use_ddp,  # Enable aggregation to get complete metrics
            )
            
            # Only main process handles logging and checkpoint saving
            if is_main_process:
                print(f"Val metrics: {val_metrics}")
                print(f"Test metrics: {test_metrics}")
                
                # Log to logger
                if logger is not None and logger.is_active():
                    logger.log(
                        {
                            "val/loss": val_metrics["loss"],
                            "val/family_loss": val_metrics["family_loss"],
                            "val/family_acc": val_metrics["family_acc"],
                            "val/valid_samples_ratio": val_metrics["valid_samples_ratio"],
                            "test/loss": test_metrics["loss"],
                            "test/family_loss": test_metrics["family_loss"],
                            "test/family_acc": test_metrics["family_acc"],
                            "test/valid_samples_ratio": test_metrics["valid_samples_ratio"],
                            "epoch": epoch,
                        }
                    )
                
                # Determine if this is the best model based on validation accuracy
                is_best = val_metrics["family_acc"] > best_val_acc
                
                if is_best:
                    best_val_acc = val_metrics["family_acc"]
                    best_val_loss = val_metrics["loss"]
                    print(f"New best model! Val Family Acc: {best_val_acc:.4f}")
                
                # Save checkpoint (use unwrapped model)
                save_checkpoint_family_classifier(
                    eval_model,
                    optimizer,
                    scheduler,
                    epoch,
                    config,
                    is_best
                )
            
        elif args.debug and is_main_process:
            # Debug mode: save checkpoint each epoch
            model_to_save = model.module if use_ddp else model
            save_checkpoint_family_classifier(
                model_to_save,
                optimizer,
                scheduler,
                epoch,
                config,
                False
            )
    
    # Training completion summary
    if is_main_process and not args.debug:
        print("\n" + "=" * 50)
        print("Training completed!")
        print("=" * 50)
    
    if is_main_process:
        print(f"Best validation family accuracy: {best_val_acc:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {config['training']['output_dir']}")
        print("=" * 50)
        
        if logger is not None and logger.is_active():
            logger.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()