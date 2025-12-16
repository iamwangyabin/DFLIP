#!/usr/bin/env python3
"""
Clean metadata by removing entries for missing image files.

This script loads the metadata JSON file, checks which image files are missing,
and creates a new cleaned metadata file with only existing image files.

Usage:
    python tools/clean_metadata.py --metadata_path ./assets/dflip3k_meta.json --image_root /media/DATASET/person_data/dflip3k --output cleaned_dflip3k_meta.json
    
    # Or use config file:
    python tools/clean_metadata.py --config configs/dinov2_small_train_config.yaml --output cleaned_dflip3k_meta.json
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


def clean_metadata(metadata: List[Dict], image_root: str) -> Tuple[List[Dict], List[str], Dict[str, int]]:
    """
    Clean metadata by removing entries for missing image files.
    
    Args:
        metadata: List of metadata records
        image_root: Root directory for images
        
    Returns:
        Tuple of (cleaned_metadata, missing_files, stats)
    """
    print(f"Checking image files in root directory: {image_root}")
    
    cleaned_metadata = []
    missing_files = []
    stats = {
        "total_original": len(metadata),
        "existing": 0,
        "missing": 0,
        "splits": {"train": 0, "val": 0, "test": 0, "unknown": 0}
    }
    
    total_files = len(metadata)
    
    for i, record in enumerate(metadata):
        # Show progress every 1000 files
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"Progress: {i + 1}/{total_files} ({(i + 1)/total_files*100:.1f}%)")
        
        # Get image path from record
        image_path = record.get("image_path")
        if not image_path:
            print(f"Warning: Record {i} has no 'image_path' field, skipping: {record}")
            stats["missing"] += 1
            continue
            
        # Construct full path
        full_path = os.path.join(image_root, image_path)
        
        # Check if file exists
        if os.path.exists(full_path):
            cleaned_metadata.append(record)
            stats["existing"] += 1
            
            # Count by split for existing files
            split = record.get("split", "unknown")
            if split in stats["splits"]:
                stats["splits"][split] += 1
            else:
                stats["splits"]["unknown"] += 1
        else:
            missing_files.append(image_path)
            stats["missing"] += 1
    
    print(f"Cleaning complete: {stats['existing']} kept, {stats['missing']} removed")
    return cleaned_metadata, missing_files, stats


def save_cleaned_metadata(cleaned_metadata: List[Dict], output_path: str):
    """Save cleaned metadata to JSON file."""
    print(f"Saving cleaned metadata to: {output_path}")
    
    # Create backup of original if output path is the same as input
    if os.path.exists(output_path):
        backup_path = output_path + ".backup"
        print(f"Creating backup of existing file: {backup_path}")
        os.rename(output_path, backup_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Cleaned metadata saved successfully")


def print_statistics(stats: Dict, missing_files: List[str]):
    """Print cleaning statistics."""
    print("\n" + "=" * 60)
    print("METADATA CLEANING RESULTS")
    print("=" * 60)
    
    print(f"Original records: {stats['total_original']}")
    print(f"Records kept: {stats['existing']} ({stats['existing']/stats['total_original']*100:.2f}%)")
    print(f"Records removed: {stats['missing']} ({stats['missing']/stats['total_original']*100:.2f}%)")
    
    print(f"\nFinal split distribution:")
    for split, count in stats["splits"].items():
        if count > 0:
            print(f"  {split}: {count}")
    
    if missing_files:
        print(f"\n📝 Removed {len(missing_files)} records for missing files:")
        print("First 10 missing files:")
        for i, missing_file in enumerate(missing_files[:10]):
            print(f"  {i+1}. {missing_file}")
        
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
            
        # Save missing files to a text file
        missing_file_path = "removed_missing_files.txt"
        with open(missing_file_path, "w") as f:
            for missing_file in missing_files:
                f.write(f"{missing_file}\n")
        print(f"\nFull list of removed files saved to: {missing_file_path}")
    else:
        print(f"\n✅ No missing files found - no cleaning needed!")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Clean metadata by removing entries for missing image files")
    
    # Option 1: Specify metadata and image root directly
    parser.add_argument("--metadata_path", type=str, help="Path to metadata JSON file")
    parser.add_argument("--image_root", type=str, help="Root directory for images")
    
    # Option 2: Use config file
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    
    # Output file
    parser.add_argument("--output", type=str, required=True, help="Output path for cleaned metadata JSON file")
    
    # Option to overwrite original
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the original metadata file (creates backup)")
    
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
    
    # Handle overwrite option
    if args.overwrite:
        output_path = metadata_path
    else:
        output_path = args.output
    
    # Verify paths exist
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)
        
    if not os.path.exists(image_root):
        print(f"Error: Image root directory not found: {image_root}")
        sys.exit(1)
    
    # Load metadata and clean it
    try:
        metadata = load_metadata(metadata_path)
        cleaned_metadata, missing_files, stats = clean_metadata(metadata, image_root)
        
        # Save cleaned metadata
        save_cleaned_metadata(cleaned_metadata, output_path)
        
        # Print statistics
        print_statistics(stats, missing_files)
        
        if missing_files:
            print(f"\n✅ Successfully cleaned metadata: removed {len(missing_files)} entries for missing files")
        else:
            print(f"\n✅ No cleaning needed: all image files exist")
            
    except Exception as e:
        print(f"Error during cleaning: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()