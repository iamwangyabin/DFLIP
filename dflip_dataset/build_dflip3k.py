#!/usr/bin/env python3
"""
Data preprocessing script for DFLIP-3K dataset.
Converts raw dataset into standardized JSON format.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Build DFLIP-3K metadata")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of DFLIP-3K dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/dflip3k_meta.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Validate that all images can be loaded"
    )
    
    return parser.parse_args()


def scan_images(data_root: Path) -> List[Dict]:
    """
    Scan directory structure and build metadata.
    
    Expected structure:
    data_root/
        real/
            images/
        fake/
            stable-diffusion-v1.5/
                images/
                prompts.txt (optional)
                masks/ (optional)
            midjourney-v5/
                images/
                prompts.txt
                masks/
            ...
    
    Returns:
        List of metadata dictionaries
    """
    metadata = []
    uid_counter = 1
    
    data_root = Path(data_root)
    
    # Process real images
    print("Processing real images...")
    real_dir = data_root / 'real' / 'images'
    if real_dir.exists():
        for img_path in tqdm(list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))):
            metadata.append({
                'uid': f"DFLIP_{uid_counter:06d}",
                'file_path': str(img_path.relative_to(data_root)),
                'label': 'Real',
                'generator': None,
                'generator_id': 0,
                'gt_prompt': None,
                'mask_path': None
            })
            uid_counter += 1
    
    # Process fake images
    print("Processing fake images...")
    fake_dir = data_root / 'fake'
    
    if fake_dir.exists():
        generator_id = 1
        
        for gen_dir in sorted(fake_dir.iterdir()):
            if not gen_dir.is_dir():
                continue
            
            generator_name = gen_dir.name
            print(f"  Processing generator: {generator_name}")
            
            images_dir = gen_dir / 'images'
            prompts_file = gen_dir / 'prompts.txt'
            masks_dir = gen_dir / 'masks'
            
            if not images_dir.exists():
                print(f"    Warning: No images directory found for {generator_name}")
                continue
            
            # Load prompts if available
            prompts = {}
            if prompts_file.exists():
                with open(prompts_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Format: image_name.jpg: prompt text here
                        if ':' in line:
                            img_name, prompt = line.split(':', 1)
                            prompts[img_name.strip()] = prompt.strip()
            
            # Process images
            for img_path in tqdm(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))):
                img_name = img_path.name
                
                # Find corresponding mask
                mask_path = None
                if masks_dir.exists():
                    mask_file = masks_dir / img_name
                    if mask_file.exists():
                        mask_path = str(mask_file.relative_to(data_root))
                
                # Find corresponding prompt
                gt_prompt = prompts.get(img_name, None)
                
                metadata.append({
                    'uid': f"DFLIP_{uid_counter:06d}",
                    'file_path': str(img_path.relative_to(data_root)),
                    'label': 'Fake',
                    'generator': generator_name,
                    'generator_id': generator_id,
                    'gt_prompt': gt_prompt,
                    'mask_path': mask_path
                })
                uid_counter += 1
            
            generator_id += 1
    
    return metadata


def validate_images(metadata: List[Dict], data_root: Path):
    """Validate that all images can be loaded."""
    print("Validating images...")
    
    invalid_images = []
    
    for item in tqdm(metadata):
        img_path = data_root / item['file_path']
        
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            invalid_images.append(item['uid'])
    
    if invalid_images:
        print(f"Found {len(invalid_images)} invalid images:")
        for uid in invalid_images[:10]:  # Show first 10
            print(f"  {uid}")
        if len(invalid_images) > 10:
            print(f"  ... and {len(invalid_images) - 10} more")
        
        # Remove invalid images
        metadata = [item for item in metadata if item['uid'] not in invalid_images]
    
    return metadata


def print_statistics(metadata: List[Dict]):
    """Print dataset statistics."""
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    
    total = len(metadata)
    real_count = sum(1 for item in metadata if item['label'] == 'Real')
    fake_count = total - real_count
    
    print(f"Total images: {total}")
    print(f"Real images: {real_count} ({real_count/total*100:.1f}%)")
    print(f"Fake images: {fake_count} ({fake_count/total*100:.1f}%)")
    
    # Count by generator
    generators = {}
    for item in metadata:
        if item['label'] == 'Fake':
            gen = item['generator']
            generators[gen] = generators.get(gen, 0) + 1
    
    print(f"\nGenerator breakdown:")
    for gen, count in sorted(generators.items()):
        print(f"  {gen}: {count}")
    
    # Count with prompts
    with_prompts = sum(1 for item in metadata if item['gt_prompt'] is not None)
    print(f"\nImages with prompts: {with_prompts} ({with_prompts/total*100:.1f}%)")
    
    # Count with masks
    with_masks = sum(1 for item in metadata if item['mask_path'] is not None)
    print(f"Images with masks: {with_masks} ({with_masks/total*100:.1f}%)")
    
    print("="*50 + "\n")


def main():
    args = parse_args()
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    # Scan and build metadata
    metadata = scan_images(data_root)
    
    if not metadata:
        raise ValueError("No images found! Check your data directory structure.")
    
    # Validate images if requested
    if args.validate_images:
        metadata = validate_images(metadata, data_root)
    
    # Print statistics
    print_statistics(metadata)
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to: {output_path}")
    print(f"Total samples: {len(metadata)}")


if __name__ == '__main__':
    main()
