#!/usr/bin/env python3
"""Train Ojha Binary Classifier with single-node multi-GPU (DDP).

This script trains an Ojha binary classifier for real/fake detection only.
This is a comparison method using CLIP features for fake detection.

Usage examples (single node with 4 GPUs):

    torchrun --nproc_per_node=4 scripts/train_ojha.py \
        --config configs/ojha_train_config.yaml

    # Or single GPU:
    python scripts/train_ojha.py --config configs/ojha_train_config.yaml
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
from models.ojha import create_ojha  # noqa: E402
from dataset import create_profiling_dataloaders_ddp, create_profiling_dataloaders  # noqa: E402
from utils.logger import create_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Ojha Binary Classifier (DDP)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ojha_train_config.yaml",
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


def create_optimizer(model: nn.Module, config: Dict):
    """Create optimizer for Ojha."""
    train_config = config["training"]
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        betas=(train_config["adam_beta1"], train_config["adam_beta2"]),
        eps=train_config["adam_epsilon"],
    )
    return optimizer


def create_scheduler(optimizer, num_training_steps: int, config: Dict):
    """Create learning rate scheduler."""
    from transformers import get_cosine_schedule_with_warmup
    
    train_config = config["training"]
    num_warmup_steps = int(num_training_steps * train_config.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


def train_epoch_ojha(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    criterion: nn.Module,
    epoch: int,
    config: Dict,
    debug: bool = False,
    max_steps: int = None,
    logger=None,
):
    """Train Ojha for a single epoch."""
    from tqdm import tqdm
    
    model.train()
    train_config = config["training"]
    grad_accum = train_config["gradient_accumulation_steps"]
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        if max_steps is not None and step >= max_steps:
            break
        
        pixel_values = batch["pixel_values"].to(model.device)
        is_fake = batch["is_fake"].to(model.device)
        
        # Forward pass
        outputs = model(pixel_values)
        loss = criterion(outputs['logits'], is_fake)
        
        loss = loss / grad_accum
        loss.backward()
        
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Statistics
        total_loss += (loss.item() * grad_accum)
        _, predicted = outputs['logits'].max(1)
        correct += predicted.eq(is_fake).sum().item()
        total_samples += is_fake.size(0)
        
        # Update progress bar
        accuracy = 100. * correct / total_samples
        pbar_dict = {
            'loss': f'{loss.item() * grad_accum:.4f}',
            'acc': f'{accuracy:.1f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
        }
        progress_bar.set_postfix(pbar_dict)
        
        # Logging
        if logger is not None and logger.is_active() and (step + 1) % train_config["logging_steps"] == 0:
            log_dict = {
                "train/loss": loss.item() * grad_accum,
                "train/accuracy": accuracy,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": step,
            }
            logger.log(log_dict)
    
    num_steps = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    num_steps = max(num_steps, 1)
    
    metrics = {
        'loss': total_loss / num_steps,
        'accuracy': correct / total_samples if total_samples > 0 else 0.0,
    }
    
    return metrics


@torch.no_grad()
def evaluate_ojha(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    use_ddp: bool = False,
):
    """Evaluate Ojha on validation set."""
    from tqdm import tqdm
    import torch.distributed as dist
    
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Check if we should show progress bar (only main process in DDP)
    show_progress = True
    if use_ddp and dist.is_initialized():
        show_progress = dist.get_rank() == 0
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    # Only show tqdm on main process to avoid multiple progress bars
    iterator = tqdm(dataloader, desc="Evaluating") if show_progress else dataloader
    for batch in iterator:
        pixel_values = batch["pixel_values"].to(device)
        is_fake = batch["is_fake"].to(device)
        
        # Forward pass
        outputs = model(pixel_values)
        loss = criterion(outputs['logits'], is_fake)
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs['logits'].max(1)
        correct += predicted.eq(is_fake).sum().item()
        total_samples += is_fake.size(0)
    
    # Aggregate metrics across all DDP processes if needed
    if use_ddp and dist.is_initialized():
        # Convert to tensors for all_reduce
        metrics_tensor = torch.tensor([
            total_loss, correct, total_samples
        ], dtype=torch.float32, device=device)
        
        # Sum across all processes
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Extract aggregated values
        total_loss, correct, total_samples = metrics_tensor.tolist()
        
        # Convert back to int for counts
        correct = int(correct)
        total_samples = int(total_samples)
        
        # For loss averaging, we need to divide by total number of batches across all processes
        world_size = dist.get_world_size()
        num_batches = len(dataloader) * world_size
    else:
        num_batches = max(len(dataloader), 1)
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': correct / total_samples if total_samples > 0 else 0.0,
    }


def save_checkpoint_ojha(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    config: Dict,
    is_best: bool = False,
):
    """Save Ojha model checkpoint."""
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
        train_loader, val_loader, test_loader = create_profiling_dataloaders(
            config, train_transform, val_transform
        )
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    if is_main_process:
        print("Creating Ojha binary classifier model...")
    
    model = create_ojha(config, verbose=is_main_process)
    
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
            find_unused_parameters=False
        )
    
    if is_main_process:
        print("Setting up binary cross-entropy loss...")
    
    # Setup loss for binary classification
    criterion = nn.CrossEntropyLoss()
    
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
        train_metrics = train_epoch_ojha(
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
            val_metrics = evaluate_ojha(
                eval_model,
                val_loader,
                criterion,
                use_ddp=use_ddp,
            )
            
            # Test evaluation (for observation only)
            test_metrics = evaluate_ojha(
                eval_model,
                test_loader,
                criterion,
                use_ddp=use_ddp,
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
                            "val/accuracy": val_metrics["accuracy"],
                            "test/loss": test_metrics["loss"],
                            "test/accuracy": test_metrics["accuracy"],
                            "epoch": epoch,
                        }
                    )
                
                # Determine if this is the best model based on validation accuracy
                is_best = val_metrics["accuracy"] > best_val_acc
                
                if is_best:
                    best_val_acc = val_metrics["accuracy"]
                    best_val_loss = val_metrics["loss"]
                    print(f"New best model! Val Acc: {best_val_acc:.4f}")
                
                # Save checkpoint (use unwrapped model)
                save_checkpoint_ojha(
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
            save_checkpoint_ojha(
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
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {config['training']['output_dir']}")
        print("=" * 50)
        
        if logger is not None and logger.is_active():
            logger.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()