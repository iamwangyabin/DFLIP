#!/usr/bin/env python3
"""
Simple script to verify that all image files referenced in metadata actually exist.

This script loads the metadata JSON file and checks if each image file exists
on the filesystem by combining image_root with the image_path from each record.

Usage:
    python tools/verify_image_files.py --metadata_path ./assets/dflip3k_meta.json --image_root /media/DATASET/person_data/dflip3k
    
    # Or use config file:
    python tools/verify_image_files.py --config configs/dinov2_small_train_config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_metadata(metadata_path: str) -> List[Dict]:
    """Load metadata from JSON file."""
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} records from metadata")
    return metadata


def verify_image_files(metadata: List[Dict], image_root: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Verify that all image files exist.
    
    Args:
        metadata: List of metadata records
        image_root: Root directory for images
        
    Returns:
        Tuple of (existing_files, missing_files, split_stats)
    """
    print(f"Checking image files in root directory: {image_root}")
    
    existing_files = []
    missing_files = []
    split_stats = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    
    total_files = len(metadata)
    
    for i, record in enumerate(metadata):
        # Show progress every 1000 files
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"Progress: {i + 1}/{total_files} ({(i + 1)/total_files*100:.1f}%)")
        
        # Get image path from record
        image_path = record.get("image_path")
        if not image_path:
            print(f"Warning: Record {i} has no 'image_path' field: {record}")
            continue
            
        # Construct full path
        full_path = os.path.join(image_root, image_path)
        
        # Check if file exists
        if os.path.exists(full_path):
            existing_files.append(image_path)
        else:
            missing_files.append(image_path)
            
        # Count by split
        split = record.get("split", "unknown")
        if split in split_stats:
            split_stats[split] += 1
        else:
            split_stats["unknown"] += 1
    
    print(f"Verification complete: {len(existing_files)} existing, {len(missing_files)} missing")
    return existing_files, missing_files, split_stats


def print_statistics(existing_files: List[str], missing_files: List[str], split_stats: Dict[str, int]):
    """Print verification statistics."""
    total_files = len(existing_files) + len(missing_files)
    
    print("\n" + "=" * 60)
    print("IMAGE FILE VERIFICATION RESULTS")
    print("=" * 60)
    
    print(f"Total files in metadata: {total_files}")
    print(f"Existing files: {len(existing_files)} ({len(existing_files)/total_files*100:.2f}%)")
    print(f"Missing files: {len(missing_files)} ({len(missing_files)/total_files*100:.2f}%)")
    
    print(f"\nSplit distribution:")
    for split, count in split_stats.items():
        if count > 0:
            print(f"  {split}: {count}")
    
    if missing_files:
        print(f"\n⚠️  WARNING: {len(missing_files)} files are missing!")
        print("First 10 missing files:")
        for i, missing_file in enumerate(missing_files[:10]):
            print(f"  {i+1}. {missing_file}")
        
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
            
        # Save missing files to a text file
        missing_file_path = "missing_image_files.txt"
        with open(missing_file_path, "w") as f:
            for missing_file in missing_files:
                f.write(f"{missing_file}\n")
        print(f"\nFull list of missing files saved to: {missing_file_path}")
    else:
        print(f"\n✅ All image files exist!")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Verify that all image files in metadata exist")
    
    # Option 1: Specify metadata and image root directly
    parser.add_argument("--metadata_path", type=str, help="Path to metadata JSON file")
    parser.add_argument("--image_root", type=str, help="Root directory for images")
    
    # Option 2: Use config file
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    
    args = parser.parse_args()
    
    # Determine metadata_path and image_root
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        metadata_path = config["data"]["metadata_path"]
        image_root = config["data"]["image_root"]
    elif args.metadata_path and args.image_root:
        metadata_path = args.metadata_path
        image_root = args.image_root
    else:
        print("Error: Either provide --config or both --metadata_path and --image_root")
        sys.exit(1)
    
    # Verify paths exist
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)
        
    if not os.path.exists(image_root):
        print(f"Error: Image root directory not found: {image_root}")
        sys.exit(1)
    
    # Load metadata and verify files
    try:
        metadata = load_metadata(metadata_path)
        existing_files, missing_files, split_stats = verify_image_files(metadata, image_root)
        print_statistics(existing_files, missing_files, split_stats)
        
        # Exit with error code if files are missing
        if missing_files:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()