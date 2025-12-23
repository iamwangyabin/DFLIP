#!/usr/bin/env python3
"""
Fix family_id and version_id for real images in dflip3k metadata JSON.

This script reads an existing metadata JSON file, analyzes the fake images to learn
the (family, submodel) -> (family_id, version_id) mapping, then applies this mapping
to real images that currently have null family_id and version_id values.

Usage:
    python tools/fix_real_image_ids.py \
        --input /Users/wangyabin/Downloads/dflip3k_meta_processed.json \
        --output /Users/wangyabin/Downloads/dflip3k_meta_fixed.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix family_id and version_id for real images in metadata JSON"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input metadata JSON file path"
    )
    parser.add_argument(
        "--output", 
        type=str,
        required=True,
        help="Output fixed metadata JSON file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually modifying the file"
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Skip interactive confirmation and proceed automatically"
    )
    return parser.parse_args()


def parse_image_path(image_path: str, is_fake: bool) -> Optional[Tuple[str, str]]:
    """
    Extract family and submodel from image path.
    
    Expected formats:
    - Fake: train/fake/<family>/<submodel>/<file>
    - Real: train/real/<family>/<submodel>/<file>
    
    Returns:
        Tuple of (family, submodel) or None if path doesn't match expected format
    """
    parts = image_path.split('/')
    
    if len(parts) < 4:
        return None
        
    # Check if it's the expected fake/real structure
    if is_fake and parts[1] != 'fake':
        return None
    if not is_fake and parts[1] != 'real':
        return None
        
    family = parts[2]
    submodel = parts[3]
    
    return (family, submodel)


def build_id_mappings(records: List[Dict]) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    Build (family, submodel) -> (family_id, version_id) mapping from fake images.
    
    Args:
        records: List of metadata records
        
    Returns:
        Dictionary mapping (family, submodel) tuples to (family_id, version_id) tuples
    """
    mappings = {}
    
    for record in records:
        if record['is_fake'] == 1 and record['family_id'] is not None:
            path_info = parse_image_path(record['image_path'], is_fake=True)
            if path_info:
                family, submodel = path_info
                family_id = record['family_id']
                version_id = record['version_id']
                
                key = (family, submodel)
                value = (family_id, version_id)
                
                # Check for consistency
                if key in mappings and mappings[key] != value:
                    print(f"[Warning] Inconsistent mapping for {key}: "
                          f"existing {mappings[key]} vs new {value}")
                else:
                    mappings[key] = value
    
    return mappings


def fix_real_image_ids(records: List[Dict], id_mappings: Dict[Tuple[str, str], Tuple[int, int]], dry_run: bool = False) -> int:
    """
    Update family_id and version_id for real images with null values.
    
    Args:
        records: List of metadata records to modify
        id_mappings: Mapping from (family, submodel) to (family_id, version_id)
        dry_run: If True, only show what would be changed without modifying
        
    Returns:
        Number of records that were (or would be) updated
    """
    updated_count = 0
    skipped_count = 0
    
    for record in records:
        # Only process real images with null family_id
        if record['is_fake'] == 0 and record['family_id'] is None:
            path_info = parse_image_path(record['image_path'], is_fake=False)
            
            if path_info:
                family, submodel = path_info
                key = (family, submodel)
                
                if key in id_mappings:
                    family_id, version_id = id_mappings[key]
                    
                    if dry_run:
                        print(f"[DRY RUN] Would update {record['image_path']}: "
                              f"family_id={family_id}, version_id={version_id}")
                    else:
                        record['family_id'] = family_id
                        record['version_id'] = version_id
                        print(f"Updated {record['image_path']}: "
                              f"family_id={family_id}, version_id={version_id}")
                    
                    updated_count += 1
                else:
                    print(f"[Warning] No mapping found for real image: {record['image_path']} "
                          f"(family={family}, submodel={submodel})")
                    skipped_count += 1
            else:
                print(f"[Warning] Could not parse path for real image: {record['image_path']}")
                skipped_count += 1
    
    return updated_count


def get_user_confirmation(id_mappings: Dict[Tuple[str, str], Tuple[int, int]],
                         records: List[Dict], auto_confirm: bool = False) -> bool:
    """
    Show the user the mappings and examples of changes, then ask for confirmation.
    
    Args:
        id_mappings: The mappings that will be applied
        records: All records to show examples
        auto_confirm: If True, skip confirmation and return True
        
    Returns:
        True if user confirms, False otherwise
    """
    if auto_confirm:
        return True
    
    print("\n" + "="*80)
    print("CORRESPONDENCE RELATIONSHIPS FOUND:")
    print("="*80)
    
    print(f"\nFound {len(id_mappings)} unique (family, submodel) → (family_id, version_id) mappings:")
    print("-" * 60)
    for (family, submodel), (family_id, version_id) in sorted(id_mappings.items()):
        print(f"  {family:15} / {submodel:20} → family_id={family_id:2}, version_id={version_id:2}")
    
    # Show some examples of real images that would be affected
    print(f"\n" + "="*80)
    print("EXAMPLES OF REAL IMAGES THAT WILL BE UPDATED:")
    print("="*80)
    
    example_count = 0
    max_examples = 10
    
    for record in records:
        if record['is_fake'] == 0 and record['family_id'] is None and example_count < max_examples:
            path_info = parse_image_path(record['image_path'], is_fake=False)
            if path_info:
                family, submodel = path_info
                key = (family, submodel)
                if key in id_mappings:
                    family_id, version_id = id_mappings[key]
                    print(f"  {record['image_path']}")
                    print(f"    → Will set: family_id={family_id}, version_id={version_id}")
                    print()
                    example_count += 1
    
    if example_count == 0:
        print("  No real images found that match the discovered mappings.")
    elif example_count == max_examples:
        remaining = sum(1 for r in records
                       if r['is_fake'] == 0 and r['family_id'] is None
                       and parse_image_path(r['image_path'], is_fake=False)
                       and parse_image_path(r['image_path'], is_fake=False) in id_mappings)
        if remaining > max_examples:
            print(f"  ... and {remaining - max_examples} more real images")
    
    print("\n" + "="*80)
    
    while True:
        response = input("\nDo these mappings look correct? Do you want to proceed? (y/n/q): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        elif response in ['q', 'quit']:
            print("Exiting...")
            sys.exit(0)
        else:
            print("Please enter 'y' (yes), 'n' (no), or 'q' (quit)")


def main():
    args = parse_args()
    
    # Load the input JSON file
    print(f"Loading metadata from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    print(f"Loaded {len(records)} records")
    
    # Count current state
    fake_count = sum(1 for r in records if r['is_fake'] == 1)
    real_count = sum(1 for r in records if r['is_fake'] == 0)
    real_null_count = sum(1 for r in records if r['is_fake'] == 0 and r['family_id'] is None)
    
    print(f"  - Fake images: {fake_count}")
    print(f"  - Real images: {real_count}")
    print(f"  - Real images with null family_id: {real_null_count}")
    
    # Build mappings from fake images
    print("\nBuilding family/submodel mappings from fake images...")
    id_mappings = build_id_mappings(records)
    
    if not id_mappings:
        print("No mappings found! Cannot proceed.")
        return
    
    # Show mappings and get user confirmation
    if not get_user_confirmation(id_mappings, records, args.auto_confirm):
        print("Operation cancelled by user.")
        return
    
    # Fix real images
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Fixing real image IDs...")
    updated_count = fix_real_image_ids(records, id_mappings, dry_run=args.dry_run)
    
    print(f"\n{'Would update' if args.dry_run else 'Updated'} {updated_count} real images")
    
    # Save the result (unless dry run)
    if not args.dry_run:
        print(f"\nSaving fixed metadata to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print("Done!")
        
        # Verify the fix
        real_null_count_after = sum(1 for r in records if r['is_fake'] == 0 and r['family_id'] is None)
        print(f"\nVerification:")
        print(f"  - Real images with null family_id before: {real_null_count}")
        print(f"  - Real images with null family_id after: {real_null_count_after}")
        print(f"  - Successfully fixed: {real_null_count - real_null_count_after}")
    else:
        print(f"\nDry run completed. Use --output to actually save the changes.")


if __name__ == "__main__":
    main()