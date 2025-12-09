#!/usr/bin/env python3
"""
Generate dflip3k_meta.json from organized image folders.

Expected directory structure:
    data/images/
        real/*.png
        family0/version0/*.png
        family0/version1/*.png
        family1/version7/*.png
        ...
        family2/version14/*.png
        ...

This script will create a JSON metadata file with all required fields for BHEP training.
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dflip3k_meta.json from organized folders")
    parser.add_argument(
        "--image-root",
        type=str,
        default="./data/images",
        help="Root directory containing organized image folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./assets/dflip3k_meta.json",
        help="Output path for metadata JSON"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val/test split ratio (train_ratio, val_ratio=0.1, test_ratio=0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def scan_images(image_root: Path) -> List[Dict]:
    """
    Scan organized image folders and generate metadata records.
    
    Expected structure:
    - real/*.png -> is_fake=0
    - family{fid}/version{vid}/*.png -> is_fake=1, family_id=fid, version_id=vid
    """
    records = []
    image_root = Path(image_root)
    
    print(f"Scanning images from {image_root}...")
    
    # 1. Real images
    real_dir = image_root / "real"
    if real_dir.exists():
        real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg"))
        print(f"Found {len(real_images)} real images")
        
        for img_path in real_images:
            rel_path = img_path.relative_to(image_root).as_posix()
            records.append({
                "image_path": rel_path,
                "is_fake": 0,
                "family_id": None,
                "version_id": None,
                "split": "train"
            })
    else:
        print(f"Warning: No 'real' folder found at {real_dir}")
    
    # 2. Fake images organized by family and version
    for family_dir in sorted(image_root.glob("family*")):
        if not family_dir.is_dir():
            continue
        
        try:
            fam_id = int(family_dir.name.replace("family", ""))
        except ValueError:
            print(f"Skipping {family_dir.name}: cannot parse family ID")
            continue
        
        fam_images = 0
        
        for version_dir in sorted(family_dir.glob("version*")):
            if not version_dir.is_dir():
                continue
            
            try:
                ver_id = int(version_dir.name.replace("version", ""))
            except ValueError:
                print(f"Skipping {version_dir.name}: cannot parse version ID")
                continue
            
            version_images = (
                list(version_dir.glob("*.png")) +
                list(version_dir.glob("*.jpg")) +
                list(version_dir.glob("*.jpeg"))
            )
            
            for img_path in version_images:
                rel_path = img_path.relative_to(image_root).as_posix()
                records.append({
                    "image_path": rel_path,
                    "is_fake": 1,
                    "family_id": fam_id,
                    "version_id": ver_id,
                    "split": "train"
                })
                fam_images += 1
        
        print(f"Found {fam_images} fake images in family {fam_id}")
    
    total_images = len(records)
    print(f"Total images found: {total_images}")
    
    return records


def split_dataset(records: List[Dict], train_ratio: float = 0.8, seed: int = 42) -> List[Dict]:
    """
    Split records into train/val/test based on ratios.
    Default: 80% train, 10% val, 10% test
    """
    random.seed(seed)
    random.shuffle(records)
    
    n = len(records)
    train_end = int(train_ratio * n)
    val_end = int(train_ratio * n + 0.1 * n)
    
    for i, record in enumerate(records):
        if i < train_end:
            record["split"] = "train"
        elif i < val_end:
            record["split"] = "val"
        else:
            record["split"] = "test"
    
    train_count = sum(1 for r in records if r["split"] == "train")
    val_count = sum(1 for r in records if r["split"] == "val")
    test_count = sum(1 for r in records if r["split"] == "test")
    
    train_pct = 100 * train_count / n
    val_pct = 100 * val_count / n
    test_pct = 100 * test_count / n
    
    print(f"\nDataset split:")
    print(f"  Train: {train_count} ({train_pct:.1f}%)")
    print(f"  Val:   {val_count} ({val_pct:.1f}%)")
    print(f"  Test:  {test_count} ({test_pct:.1f}%)")
    
    return records


def validate_hierarchy(records: List[Dict], num_families: int = 3, num_versions: int = 20) -> bool:
    """
    Validate that all family_id and version_id values are within expected ranges.
    """
    fake_records = [r for r in records if r["is_fake"] == 1]
    
    issues = []
    
    for record in fake_records:
        fam_id = record.get("family_id")
        ver_id = record.get("version_id")
        
        if fam_id is not None and (fam_id < 0 or fam_id >= num_families):
            issues.append(f"Invalid family_id {fam_id} in {record['image_path']}")
        
        if ver_id is not None and (ver_id < 0 or ver_id >= num_versions):
            issues.append(f"Invalid version_id {ver_id} in {record['image_path']}")
    
    if issues:
        num_issues = len(issues)
        print(f"\nValidation warnings ({num_issues} issues):")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if num_issues > 10:
            remaining = num_issues - 10
            print(f"  ... and {remaining} more issues")
        return False
    
    print("\nValidation passed!")
    return True


def main():
    args = parse_args()
    
    image_root = Path(args.image_root)
    if not image_root.exists():
        print(f"Error: Image root directory not found at {image_root}")
        return
    
    records = scan_images(image_root)
    
    if not records:
        print("Error: No images found!")
        return
    
    records = split_dataset(records, train_ratio=args.train_ratio, seed=args.seed)
    
    validate_hierarchy(records)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"\nMetadata saved to {output_path}")


if __name__ == '__main__':
    main()
