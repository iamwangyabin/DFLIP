#!/usr/bin/env python3
"""
Training script for Stage 2: The Interpreter
Supervised fine-tuning for prompt prediction with frozen Stage 1 weights.
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

from dflip_models import DFLIPInterpreter, create_interpreter
from dataset import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFLIP Stage 2: Interpreter")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dflip_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        required=True,
        help="Path to Stage 1 checkpoint"
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode with fewer steps")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps (for debug)")
    
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
            name=f"stage2_interpreter_{wandb.util.generate_id()}"
        )


def create_optimizer(model: nn.Module, config: Dict):
    """Create optimizer for LLM LoRA parameters."""
    train_config = config['stage2_training']
    
    # Only optimize trainable parameters (LLM LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = AdamW(
        trainable_params,
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        betas=(train_config['adam_beta1'], train_config['adam_beta2']),
        eps=train_config['adam_epsilon']
    )
    
    return optimizer


def create_scheduler(optimizer, num_training_steps: int, config: Dict):
    """Create learning rate scheduler."""
    train_config = config['stage2_training']
    
    num_warmup_steps = int(num_training_steps * train_config['warmup_ratio'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler


def train_epoch(
    model: DFLIPInterpreter,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    epoch: int,
    config: Dict,
    debug: bool = False,
    max_steps: int = None
):
    """Train for one epoch."""
    model.train()
    
    train_config = config['stage2_training']
    gradient_accumulation_steps = train_config['gradient_accumulation_steps']
    
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    # Determine correct device to use for tensors
    device = next(model.parameters()).device
    
    for step, batch in enumerate(progress_bar):
        if max_steps and step >= max_steps:
            break
        
        # Move to device
        pixel_values = batch.get('pixel_values').to(device)
        input_ids = batch.get('input_ids').to(device)
        attention_mask = batch.get('attention_mask').to(device)
        
        # Labels are the same as input_ids for causal LM
        # but we need to shift them and mask padding
        labels = input_ids.clone()
        labels[labels == model.module.processor.tokenizer.pad_token_id if hasattr(model, 'module') else model.processor.tokenizer.pad_token_id] = -100
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss'] / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += outputs['loss'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{outputs['loss'].item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log to wandb
        if wandb.run and (step + 1) % train_config['logging_steps'] == 0:
            wandb.log({
                'train/loss': outputs['loss'].item(),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch,
                'step': step
            })
    
    # Return average loss
    num_steps = min(len(dataloader), max_steps) if max_steps else len(dataloader)
    return {'loss': total_loss / num_steps}


@torch.no_grad()
def evaluate(model: DFLIPInterpreter, dataloader: DataLoader):
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0
    
    device = next(model.parameters()).device
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch.get('pixel_values').to(device)
        input_ids = batch.get('input_ids').to(device)
        attention_mask = batch.get('attention_mask').to(device)
        
        # Create labels
        labels = input_ids.clone()
        labels[labels == model.module.processor.tokenizer.pad_token_id if hasattr(model, 'module') else model.processor.tokenizer.pad_token_id] = -100
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        total_loss += outputs['loss'].item()
    
    num_batches = len(dataloader)
    return {'loss': total_loss / num_batches}


def save_checkpoint(
    model: DFLIPInterpreter,
    optimizer,
    scheduler,
    epoch: int,
    config: Dict,
    is_best: bool = False
):
    """Save model checkpoint."""
    output_dir = Path(config['stage2_training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights
    # If model is wrapped in DataParallel, get the underlying module
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_lora_weights(str(output_dir / f'checkpoint-epoch-{epoch}'))
    
    # Save optimizer and scheduler
    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, output_dir / f'trainer_state-epoch-{epoch}.pt')
    
    if is_best:
        model_to_save.save_lora_weights(str(output_dir / 'best_model'))
        torch.save(checkpoint, output_dir / 'best_trainer_state.pt')
    
    print(f"Saved checkpoint to {output_dir}")


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override for debug mode
    if args.debug:
        config['stage2_training']['num_epochs'] = 1
        config['data']['num_workers'] = 0
        if args.steps:
            print(f"Debug mode: limiting to {args.steps} steps per epoch")
    
    # Set seed
    torch.manual_seed(config['hardware']['seed'])
    
    # Setup logging
    setup_wandb(config)
    
    # Verify Stage 1 checkpoint exists
    if not Path(args.stage1_checkpoint).exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {args.stage1_checkpoint}")
    
    print(f"Using Stage 1 checkpoint: {args.stage1_checkpoint}")
    
    # Create datasets (interpreting mode)
    print("Creating datasets...")
    
    # We need processor first to create dataloader
    processor = AutoProcessor.from_pretrained(
        config['model']['base_model'],
        cache_dir=config['model'].get('cache_dir')
    )
    
    train_loader = create_dataloader(
        metadata_path=config['data']['metadata_path'],
        image_root=config['data']['image_root'],
        processor=processor,
        task_mode='interpreting',  # Stage 2 mode
        batch_size=config['stage2_training']['batch_size'],
        split='train',
        config=config
    )
    
    val_loader = create_dataloader(
        metadata_path=config['data']['metadata_path'],
        image_root=config['data']['image_root'],
        processor=processor,
        task_mode='interpreting',
        batch_size=config['stage2_training']['batch_size'],
        split='val',
        config=config
    )
    
    # Create model
    print("Creating model and loading Stage 1 weights...")
    model = create_interpreter(config, stage1_checkpoint=args.stage1_checkpoint)
    
    # Move model to device and wrap for multi-GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    
    # Create optimizer and scheduler
    num_training_steps = (
        len(train_loader) // config['stage2_training']['gradient_accumulation_steps']
        * config['stage2_training']['num_epochs']
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
    
    for epoch in range(start_epoch, config['stage2_training']['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['stage2_training']['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            epoch, config, args.debug, args.steps
        )
        
        print(f"\nTrain metrics: {train_metrics}")
        
        # Evaluate
        if not args.debug:
            val_metrics = evaluate(model, val_loader)
            print(f"Val metrics: {val_metrics}")
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    'val/loss': val_metrics['loss'],
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
    print(f"Checkpoints saved to: {config['stage2_training']['output_dir']}")
    print("="*50)
    
    if wandb.run:
        wandb.finish()


if __name__ == '__main__':
    main()
