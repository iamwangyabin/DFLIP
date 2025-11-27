#!/usr/bin/env python3
"""
End-to-end inference script for DFLIP.
Supports Stage 1 only, Stage 2 only, or both stages.
"""

import argparse
import yaml
from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dflip_models import DFLIPProfiler, DFLIPInterpreter, DFLIPFullPipeline
from dflip_models import create_profiler, create_interpreter
from dflip_dataset import parse_assistant_response


def parse_args():
    parser = argparse.ArgumentParser(description="DFLIP Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dflip_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=['profiling', 'interpreting', 'both'],
        default='both',
        help="Which stage(s) to run"
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        help="Path to Stage 1 checkpoint (required for profiling)"
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=str,
        help="Path to Stage 2 checkpoint (required for interpreting)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/inference",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save visualization images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path: str, processor):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    
    # Process image
    processed = processor(images=image, return_tensors='pt')
    
    return image, processed['pixel_values']


def visualize_stage1_results(
    image: Image.Image,
    results: dict,
    save_path: Path = None
):
    """
    Visualize Stage 1 profiling results.
    
    Creates a figure with:
    - Original image
    - Forgery heatmap overlay
    - Detection and identification results
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Heatmap overlay
    axes[1].imshow(image)
    
    # Get forgery mask
    forgery_mask = results['forgery_masks'][0, 0].cpu().numpy()
    
    # Resize mask to match image size if needed
    if forgery_mask.shape != (image.height, image.width):
        from scipy.ndimage import zoom
        scale_y = image.height / forgery_mask.shape[0]
        scale_x = image.width / forgery_mask.shape[1]
        forgery_mask = zoom(forgery_mask, (scale_y, scale_x))
    
    # Overlay heatmap
    axes[1].imshow(forgery_mask, cmap='hot', alpha=0.5)
    axes[1].set_title("Forgery Localization Heatmap")
    axes[1].axis('off')
    
    # Add text annotations
    is_fake = results['is_fake'][0].item()
    fake_prob = results['fake_probs'][0].item()
    generator_id = results['generator_ids'][0].item()
    
    status = "FAKE" if is_fake else "REAL"
    color = 'red' if is_fake else 'green'
    
    info_text = f"Status: {status} ({fake_prob:.2%})\nGenerator ID: {generator_id}"
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return fig


def save_results_text(results: dict, stage: str, save_path: Path):
    """Save results to text file."""
    with open(save_path, 'w') as f:
        f.write(f"DFLIP Analysis Results - Stage: {stage}\n")
        f.write("=" * 50 + "\n\n")
        
        if 'is_fake' in results:
            f.write("Stage 1: Profiling Results\n")
            f.write("-" * 30 + "\n")
            is_fake = results['is_fake'][0].item()
            fake_prob = results['fake_probs'][0].item()
            generator_id = results['generator_ids'][0].item()
            
            f.write(f"Detection: {'FAKE' if is_fake else 'REAL'}\n")
            f.write(f"Confidence: {fake_prob:.2%}\n")
            f.write(f"Generator ID: {generator_id}\n\n")
        
        if 'generated_prompts' in results:
            f.write("Stage 2: Interpretation Results\n")
            f.write("-" * 30 + "\n")
            f.write(results['generated_prompts'][0])
            f.write("\n")
    
    print(f"Saved text results to {save_path}")


def run_stage1_inference(args, config):
    """Run Stage 1 (Profiler) inference only."""
    print("Running Stage 1: Profiling...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config['model']['base_model'],
        cache_dir=config['model'].get('cache_dir')
    )
    
    # Create model
    model = create_profiler(config)
    
    # Load weights
    if args.stage1_checkpoint:
        model.load_lora_weights(args.stage1_checkpoint)
    else:
        raise ValueError("Stage 1 checkpoint required for profiling")
    
    model.to(args.device)
    model.eval()
    
    # Load image
    image, pixel_values = load_image(args.image, processor)
    pixel_values = pixel_values.to(args.device)
    
    # Inference
    with torch.no_grad():
        results = model.predict(pixel_values)
    
    # Print results
    print("\n" + "="*50)
    print("Stage 1 Results:")
    print("-"*50)
    print(f"Detection: {'FAKE' if results['is_fake'][0] else 'REAL'}")
    print(f"Confidence: {results['fake_probs'][0]:.2%}")
    print(f"Generator ID: {results['generator_ids'][0]}")
    print("="*50 + "\n")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_results_text(results, 'profiling', output_dir / 'results.txt')
    
    if args.save_visualization:
        visualize_stage1_results(image, results, output_dir / 'visualization.png')
    
    return results


def run_stage2_inference(args, config):
    """Run Stage 2 (Interpreter) inference only."""
    print("Running Stage 2: Interpretation...")
    
    # Create model
    model = create_interpreter(
        config,
        stage1_checkpoint=args.stage1_checkpoint
    )
    
    # Load Stage 2 weights
    if args.stage2_checkpoint:
        model.load_lora_weights(args.stage2_checkpoint)
    else:
        raise ValueError("Stage 2 checkpoint required for interpretation")
    
    model.to(args.device)
    model.eval()
    
    # Load image
    image, pixel_values = load_image(args.image, model.processor)
    pixel_values = pixel_values.to(args.device)
    
    # Generate
    generated_texts = model.generate(
        pixel_values=pixel_values,
        max_new_tokens=config['inference']['max_new_tokens'],
        temperature=config['inference']['temperature'],
        top_p=config['inference']['top_p'],
        do_sample=config['inference']['do_sample']
    )
    
    # Print results
    print("\n" + "="*50)
    print("Stage 2 Results:")
    print("-"*50)
    print(generated_texts[0])
    print("="*50 + "\n")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'generated_prompts': generated_texts}
    save_results_text(results, 'interpreting', output_dir / 'results.txt')
    
    return results


def run_full_pipeline(args, config):
    """Run both Stage 1 and Stage 2 in sequence."""
    print("Running Full DFLIP Pipeline...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config['model']['base_model'],
        cache_dir=config['model'].get('cache_dir')
    )
    
    # Create models
    profiler = create_profiler(config)
    profiler.load_lora_weights(args.stage1_checkpoint)
    profiler.to(args.device)
    
    interpreter = create_interpreter(config, args.stage1_checkpoint)
    interpreter.load_lora_weights(args.stage2_checkpoint)
    interpreter.to(args.device)
    
    # Create pipeline
    pipeline = DFLIPFullPipeline(profiler, interpreter)
    pipeline.to(args.device)
    pipeline.eval()
    
    # Load image
    image, pixel_values = load_image(args.image, processor)
    pixel_values = pixel_values.to(args.device)
    
    # Inference
    with torch.no_grad():
        results = pipeline.predict(
            pixel_values=pixel_values,
            max_new_tokens=config['inference']['max_new_tokens'],
            temperature=config['inference']['temperature'],
            top_p=config['inference']['top_p'],
            do_sample=config['inference']['do_sample']
        )
    
    # Print results
    print("\n" + "="*50)
    print("DFLIP Full Pipeline Results:")
    print("="*50)
    print("\nStage 1: Profiling")
    print("-"*50)
    print(f"Detection: {'FAKE' if results['is_fake'][0] else 'REAL'}")
    print(f"Confidence: {results['fake_probs'][0]:.2%}")
    print(f"Generator ID: {results['generator_ids'][0]}")
    
    print("\nStage 2: Interpretation")
    print("-"*50)
    print(results['generated_prompts'][0])
    print("="*50 + "\n")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_results_text(results, 'both', output_dir / 'results.txt')
    
    if args.save_visualization:
        visualize_stage1_results(image, results, output_dir / 'visualization.png')
    
    return results


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Verify image exists
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # Run appropriate pipeline
    if args.stage == 'profiling':
        results = run_stage1_inference(args, config)
    elif args.stage == 'interpreting':
        results = run_stage2_inference(args, config)
    else:  # both
        if not args.stage1_checkpoint or not args.stage2_checkpoint:
            raise ValueError("Both stage1-checkpoint and stage2-checkpoint required for full pipeline")
        results = run_full_pipeline(args, config)
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
