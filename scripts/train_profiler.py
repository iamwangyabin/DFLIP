#!/usr/bin/env python3
"""
Training script for Stage 1: The Profiler
Multi-task training for detection, identification, and localization.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_cosine_schedule_with_warmup
from transformers import AutoProcessor
from tqdm import tqdm
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dflip_models import DFLIPProfiler, create_profiler, MultiTaskLoss
from dflip_dataset import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFLIP Stage 1: Profiler")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dflip_config.yaml",
        help="Path to config file"
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: Dict):
    """Initialize Weights & Biases logging."""
    logging_config = config.get('logging', {})
    
    if logging_config.get('use_wandb', False):
        wandb.init(
            project=logging_config.get('wandb_project', 'dflip'),
            entity=logging_config.get('wandb_entity'),
            config=config,
            name=f"stage1_profiler_{wandb.util.generate_id()}"
        )


def create_optimizer(model: nn.Module, config: Dict):
    """Create optimizer with proper parameter grouping."""
    train_config = config['stage1_training']
    
    # Separate LoRA parameters from task heads
    lora_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora' in name.lower():
                lora_params.append(param)
            else:
                head_params.append(param)
    
    optimizer = AdamW([
        {'params': lora_params, 'lr': train_config['learning_rate']},
        {'params': head_params, 'lr': train_config['learning_rate'] * 2}  # Higher LR for heads
    ],
        weight_decay=train_config['weight_decay'],
        betas=(train_config['adam_beta1'], train_config['adam_beta2']),
        eps=train_config['adam_epsilon']
    )
    
    return optimizer


def create_scheduler(optimizer, num_training_steps: int, config: Dict):
    """Create learning rate scheduler."""
    train_config = config['stage1_training']
    
    num_warmup_steps = int(num_training_steps * train_config['warmup_ratio'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler


def train_epoch(
    model: DFLIPProfiler,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    loss_fn: MultiTaskLoss,
    epoch: int,
    config: Dict,
    debug: bool = False,
    max_steps: int = None
):
    """Train for one epoch."""
    model.train()
    
    train_config = config['stage1_training']
    gradient_accumulation_steps = train_config['gradient_accumulation_steps']
    
    total_loss = 0
    total_det_loss = 0
    total_iden_loss = 0
    total_loc_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        if max_steps and step >= max_steps:
            break
        
        # Move to device
        pixel_values = batch['pixel_values'].to(model.device)
        is_fake = batch['is_fake'].to(model.device)
        generator_ids = batch['generator_ids'].to(model.device)
        masks = batch['masks'].to(model.device)
        
        # Forward pass
        outputs = model(pixel_values)
        
        # Compute loss
        loss_dict = loss_fn(
            detection_logits=outputs['detection_logits'],
            identification_logits=outputs['identification_logits'],
            localization_pred=outputs['localization_mask'],
            is_fake_labels=is_fake,
            generator_labels=generator_ids,
            mask_labels=masks
        )
        
        loss = loss_dict['loss'] / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss_dict['loss']
        total_det_loss += loss_dict['detection_loss']
        total_iden_loss += loss_dict['identification_loss']
        total_loc_loss += loss_dict['localization_loss']
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['loss']:.4f}",
            'det': f"{loss_dict['detection_loss']:.4f}",
            'iden': f"{loss_dict['identification_loss']:.4f}",
            'loc': f"{loss_dict['localization_loss']:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log to wandb
        if wandb.run and (step + 1) % train_config['logging_steps'] == 0:
            wandb.log({
                'train/loss': loss_dict['loss'],
                'train/detection_loss': loss_dict['detection_loss'],
                'train/identification_loss': loss_dict['identification_loss'],
                'train/localization_loss': loss_dict['localization_loss'],
                'train/learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch,
                'step': step
            })
    
    # Return average losses
    num_steps = min(len(dataloader), max_steps) if max_steps else len(dataloader)
    return {
        'loss': total_loss / num_steps,
        'detection_loss': total_det_loss / num_steps,
        'identification_loss': total_iden_loss / num_steps,
        'localization_loss': total_loc_loss / num_steps
    }


@torch.no_grad()
def evaluate(model: DFLIPProfiler, dataloader: DataLoader, loss_fn: MultiTaskLoss):
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0
    total_det_loss = 0
    total_iden_loss = 0
    total_loc_loss = 0
    
    correct_det = 0
    correct_iden = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(model.device)
        is_fake = batch['is_fake'].to(model.device)
        generator_ids = batch['generator_ids'].to(model.device)
        masks = batch['masks'].to(model.device)
        
        # Forward pass
        outputs = model(pixel_values)
        predictions = model.predict(pixel_values)
        
        # Compute loss
        loss_dict = loss_fn(
            detection_logits=outputs['detection_logits'],
            identification_logits=outputs['identification_logits'],
            localization_pred=outputs['localization_mask'],
            is_fake_labels=is_fake,
            generator_labels=generator_ids,
            mask_labels=masks
        )
        
        # Accumulate metrics
        total_loss += loss_dict['loss']
        total_det_loss += loss_dict['detection_loss']
        total_iden_loss += loss_dict['identification_loss']
        total_loc_loss += loss_dict['localization_loss']
        
        # Compute accuracy
        correct_det += (predictions['is_fake'] == is_fake).sum().item()
        correct_iden += (predictions['generator_ids'] == generator_ids).sum().item()
        total_samples += len(is_fake)
    
    num_batches = len(dataloader)
    
    return {
        'loss': total_loss / num_batches,
        'detection_loss': total_det_loss / num_batches,
        'identification_loss': total_iden_loss / num_batches,
        'localization_loss': total_loc_loss / num_batches,
        'detection_accuracy': correct_det / total_samples,
        'identification_accuracy': correct_iden / total_samples
    }


def save_checkpoint(model: DFLIPProfiler, optimizer, scheduler, epoch: int, config: Dict, is_best: bool = False):
    """Save model checkpoint."""
    output_dir = Path(config['stage1_training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights
    model.save_lora_weights(str(output_dir / f'checkpoint-epoch-{epoch}'))
    
    # Save optimizer and scheduler
    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, output_dir / f'trainer_state-epoch-{epoch}.pt')
    
    if is_best:
        model.save_lora_weights(str(output_dir / 'best_model'))
        torch.save(checkpoint, output_dir / 'best_trainer_state.pt')
    
    print(f"Saved checkpoint to {output_dir}")


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override for debug mode
    if args.debug:
        config['stage1_training']['num_epochs'] = 1
        config['data']['num_workers'] = 0
        if args.steps:
            print(f"Debug mode: limiting to {args.steps} steps per epoch")
    
    # Set seed
    torch.manual_seed(config['hardware']['seed'])
    
    # Setup logging
    setup_wandb(config)
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        config['model']['base_model'],
        cache_dir=config['model'].get('cache_dir')
    )
    
    # Create datasets
    print("Creating datasets...")
    train_loader = create_dataloader(
        metadata_path=config['data']['metadata_path'],
        image_root=config['data']['image_root'],
        processor=processor,
        task_mode='profiling',
        batch_size=config['stage1_training']['batch_size'],
        split='train',
        config=config
    )
    
    val_loader = create_dataloader(
        metadata_path=config['data']['metadata_path'],
        image_root=config['data']['image_root'],
        processor=processor,
        task_mode='profiling',
        batch_size=config['stage1_training']['batch_size'],
        split='val',
        config=config
    )
    
    # Get number of generators from dataset
    config['num_generators'] = train_loader.dataset.num_generators
    
    # Create model
    print("Creating model...")
    model = create_profiler(config)
    
    # Create loss function
    loss_config = config['stage1_training']['loss_weights']
    loss_fn = MultiTaskLoss(
        detection_weight=loss_config['detection'],
        identification_weight=loss_config['identification'],
        localization_weight=loss_config['localization']
    )
    
    # Create optimizer and scheduler
    num_training_steps = (
        len(train_loader) // config['stage1_training']['gradient_accumulation_steps']
        * config['stage1_training']['num_epochs']
    )
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, num_training_steps, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        model.load_lora_weights(args.resume)
        # TODO: Load optimizer and scheduler state
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['stage1_training']['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['stage1_training']['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            epoch, config, args.debug, args.steps
        )
        
        print(f"\nTrain metrics: {train_metrics}")
        
        # Evaluate
        if not args.debug:
            val_metrics = evaluate(model, val_loader, loss_fn)
            print(f"Val metrics: {val_metrics}")
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    'val/loss': val_metrics['loss'],
                    'val/detection_loss': val_metrics['detection_loss'],
                    'val/identification_loss': val_metrics['identification_loss'],
                    'val/localization_loss': val_metrics['localization_loss'],
                    'val/detection_accuracy': val_metrics['detection_accuracy'],
                    'val/identification_accuracy': val_metrics['identification_accuracy'],
                    'epoch': epoch
                })
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                print(f"New best model! Val loss: {best_val_loss:.4f}")
            
            save_checkpoint(model, optimizer, scheduler, epoch, config, is_best)
        else:
            # In debug mode, just save each epoch
            save_checkpoint(model, optimizer, scheduler, epoch, config, False)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config['stage1_training']['output_dir']}")
    print("="*50)
    
    if wandb.run:
        wandb.finish()


if __name__ == '__main__':
    main()
