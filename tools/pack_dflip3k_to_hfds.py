#!/usr/bin/env python3
"""Pack DFLIP image dataset into a HuggingFace Datasets format with path indexing.

This script directly scans 'fake' and 'real' folders, packs all images into a
HuggingFace Dataset, and creates a path index for fast image retrieval.

Usage example:

    python tools/pack_dflip3k_to_hfds.py \
        --data-root /home/data/yabin/DFLIP3K \
        --output /home/data/yabin/datasets/dflip3k_hfds

The data-root should contain 'fake' and 'real' subdirectories.

After packing, you can quickly access images by path:

    from datasets import load_from_disk
    import json
    
    # Load dataset
    ds = load_from_disk('/path/to/output')
    
    # Load path index
    with open('/path/to/output/path_index.json') as f:
        path_to_idx = json.load(f)
    
    # Get image by path
    idx = path_to_idx['fake/family_001/v1/img001.png']
    image = ds['all'][idx]['image']
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict, Features, Image, Value
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack DFLIP dataset (fake/real folders) to HF Datasets format with path indexing"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing 'fake' and 'real' subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for HF dataset (will be created)",
    )
    parser.add_argument(
        "--image-extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Image file extensions to include (default: .jpg .jpeg .png .bmp .webp)",
    )
    return parser.parse_args()


def scan_images(
    data_root: Path, image_extensions: List[str]
) -> List[Dict]:
    """Scan fake and real folders for all images.
    
    Returns:
        List of samples with image path, relative path, and label
    """
    fake_dir = data_root / "fake"
    real_dir = data_root / "real"
    
    if not fake_dir.exists():
        raise NotADirectoryError(f"Fake directory not found: {fake_dir}")
    if not real_dir.exists():
        raise NotADirectoryError(f"Real directory not found: {real_dir}")
    
    samples = []
    
    # Scan fake folder
    print(f"\nScanning fake images in {fake_dir}...")
    fake_images = [
        img for img in fake_dir.rglob("*")
        if img.is_file() and img.suffix.lower() in image_extensions
    ]
    for img_path in tqdm(fake_images, desc="Processing fake images"):
        rel_path = img_path.relative_to(data_root)
        samples.append({
            "abs_path": str(img_path),
            "rel_path": str(rel_path).replace("\\", "/"),  # Use forward slash
            "is_fake": 1,
        })
    
    # Scan real folder
    print(f"\nScanning real images in {real_dir}...")
    real_images = [
        img for img in real_dir.rglob("*")
        if img.is_file() and img.suffix.lower() in image_extensions
    ]
    for img_path in tqdm(real_images, desc="Processing real images"):
        rel_path = img_path.relative_to(data_root)
        samples.append({
            "abs_path": str(img_path),
            "rel_path": str(rel_path).replace("\\", "/"),  # Use forward slash
            "is_fake": 0,
        })
    
    print(f"\nFound {len(fake_images)} fake images")
    print(f"Found {len(real_images)} real images")
    print(f"Total: {len(samples)} images")
    
    return samples


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output).resolve()

    print("=" * 70)
    print("Pack DFLIP dataset to HF Datasets with Path Indexing")
    print("=" * 70)
    print(f"Data root:     {data_root}")
    print(f"Output:        {output_path}")
    print(f"Extensions:    {args.image_extensions}")
    print("=" * 70)

    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root not found: {data_root}")

    # Scan all images from fake and real folders
    samples = scan_images(data_root, args.image_extensions)
    
    if not samples:
        raise RuntimeError("No images found in fake or real folders!")

    # Prepare dataset samples
    print(f"\nPreparing {len(samples)} samples for HF Dataset...")
    dataset_samples = []
    for sample in samples:
        dataset_samples.append({
            "image": sample["abs_path"],
            "image_path": sample["rel_path"],
            "is_fake": sample["is_fake"],
        })

    # Define HF features
    features = Features({
        "image": Image(),
        "image_path": Value("string"),  # Relative path for indexing
        "is_fake": Value("int8"),
    })

    # Build HF dataset (single 'all' split, no train/val division)
    print(f"\nBuilding HF Dataset with {len(dataset_samples)} samples...")
    hf_dataset = Dataset.from_list(dataset_samples, features=features)
    
    # Wrap in DatasetDict
    hf_ds = DatasetDict({"all": hf_dataset})

    # Create path index: image_path -> index
    print("\nCreating path index...")
    path_to_idx = {sample["image_path"]: idx for idx, sample in enumerate(dataset_samples)}

    # Save dataset to disk
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving HF DatasetDict to {output_path}...")
    hf_ds.save_to_disk(str(output_path))
    
    # Save path index to JSON
    index_file = output_path / "path_index.json"
    print(f"Saving path index to {index_file}...")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(path_to_idx, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ Done!")
    print("=" * 70)
    print(f"\nDataset saved to: {output_path}")
    print(f"Path index saved to: {index_file}")
    print(f"\nDataset info:")
    print(f"  - Total images: {len(dataset_samples)}")
    print(f"  - Fake images:  {sum(1 for s in dataset_samples if s['is_fake'])}")
    print(f"  - Real images:  {sum(1 for s in dataset_samples if not s['is_fake'])}")
    print(f"\nUsage example:")
    print(f"  from datasets import load_from_disk")
    print(f"  import json")
    print(f"  ")
    print(f"  # Load dataset")
    print(f"  ds = load_from_disk('{output_path}')")
    print(f"  ")
    print(f"  # Load path index")
    print(f"  with open('{index_file}') as f:")
    print(f"      path_to_idx = json.load(f)")
    print(f"  ")
    print(f"  # Get image by path")
    print(f"  image_path = 'fake/family_001/v1/img001.png'")
    print(f"  idx = path_to_idx[image_path]")
    print(f"  image = ds['all'][idx]['image']")
    print("=" * 70)


if __name__ == "__main__":
    main()
