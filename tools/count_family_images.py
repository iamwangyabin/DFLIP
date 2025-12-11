#!/usr/bin/env python3
"""
Count the number of images for each model family in the DFLIP3K dataset.
"""
import json
import argparse
from collections import Counter
from pathlib import Path


def count_family_images(meta_file: str):
    """
    Count images per family_id from metadata JSON file.
    
    Args:
        meta_file: Path to the metadata JSON file
    """
    print(f"Reading metadata from: {meta_file}")
    
    with open(meta_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    # Count images per family_id
    family_counter = Counter()
    real_images = 0
    fake_images = 0
    
    for entry in data:
        family_id = entry.get('family_id')
        is_fake = entry.get('is_fake', 0)
        
        if is_fake == 1:
            fake_images += 1
            if family_id is not None:
                family_counter[family_id] += 1
        else:
            real_images += 1
    
    print(f"\nReal images: {real_images}")
    print(f"Fake images: {fake_images}")
    print(f"\nNumber of unique families: {len(family_counter)}")
    
    # Sort by family_id
    print("\n" + "="*60)
    print(f"{'Family ID':<15} {'Image Count':<15}")
    print("="*60)
    
    for family_id in sorted(family_counter.keys()):
        count = family_counter[family_id]
        print(f"{family_id:<15} {count:<15}")
    
    print("="*60)
    
    # Additional statistics
    if family_counter:
        max_count = max(family_counter.values())
        min_count = min(family_counter.values())
        avg_count = sum(family_counter.values()) / len(family_counter)
        
        print(f"\nStatistics:")
        print(f"  Max images per family: {max_count}")
        print(f"  Min images per family: {min_count}")
        print(f"  Average images per family: {avg_count:.2f}")
        
        # Find families with max/min counts
        max_families = [fid for fid, cnt in family_counter.items() if cnt == max_count]
        min_families = [fid for fid, cnt in family_counter.items() if cnt == min_count]
        
        print(f"  Families with max count: {max_families}")
        print(f"  Families with min count: {min_families}")


def main():
    parser = argparse.ArgumentParser(
        description='Count images per model family in DFLIP3K metadata'
    )
    parser.add_argument(
        'meta_file',
        type=str,
        help='Path to the metadata JSON file (e.g., dflip3k_meta.json)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    meta_path = Path(args.meta_file)
    if not meta_path.exists():
        print(f"Error: File not found: {args.meta_file}")
        return 1
    
    count_family_images(args.meta_file)
    return 0


if __name__ == '__main__':
    exit(main())