#!/usr/bin/env python3
"""Pack DFLIP-style image dataset into a HuggingFace Datasets Arrow dataset.

This script reads an existing metadata JSON (e.g. dflip3k_meta.json) and the
corresponding image directory, and packs them into a HuggingFace DatasetDict
with Arrow backing files. This greatly reduces random small-file I/O during
training, similar to typical HuggingFace image datasets.

Usage example:

    python tools/pack_dflip3k_to_hfds.py \
        --metadata ./assets/dflip3k_meta.json \
        --image-root /home/data/yabin/DFLIP3K \
        --output /home/data/yabin/datasets/dflip3k_hfds

After running this script, you can load the dataset in two ways:

1) Directly with HuggingFace Datasets:

    from datasets import load_from_disk
    ds_dict = load_from_disk('/home/data/yabin/datasets/dflip3k_hfds')
    train_ds = ds_dict['train']

2) Via this repo's HFProfilingDataset wrapper (see dataset/profiling_dataset.py)

Note: This script expects the metadata format produced by
`tools/generate_dataset_meta.py` or `tools/generate_dataset_meta_dflip3k.py`.
"""

import argparse
import json
import os
from typing import Dict, List

from datasets import Dataset, DatasetDict, Features, Image, Value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack DFLIP dataset to HF Datasets format")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata JSON file (e.g. ./assets/dflip3k_meta.json)",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory of images used in metadata (config.data.image_root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for HF dataset (will be created, e.g. ./assets/dflip3k_hfds)",
    )
    parser.add_argument(
        "--split-names",
        type=str,
        default="train,val,test",
        help=(
            "Comma-separated list of valid split names. "
            "Any record whose split is not in this list will be mapped to 'train'. "
            "Default: 'train,val,test'."
        ),
    )
    return parser.parse_args()


def load_metadata(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise RuntimeError(f"Metadata file {path} does not contain a list of records.")
    print(f"Loaded {len(records)} records from {path}")
    return records


def main() -> None:
    args = parse_args()

    metadata_path = os.path.abspath(args.metadata)
    image_root = os.path.abspath(args.image_root)
    output_path = os.path.abspath(args.output)

    print("=== Pack DFLIP dataset to HF Datasets ===")
    print(f"metadata:   {metadata_path}")
    print(f"image_root: {image_root}")
    print(f"output:     {output_path}")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.isdir(image_root):
        raise NotADirectoryError(f"Image root not found or not a directory: {image_root}")

    records = load_metadata(metadata_path)

    # Prepare split containers
    valid_splits = [s.strip() for s in args.split_names.split(",") if s.strip()]
    if not valid_splits:
        valid_splits = ["train", "val", "test"]
    print(f"Valid splits: {valid_splits}")

    splits: Dict[str, List[Dict]] = {s: [] for s in valid_splits}

    # Fill split data
    missing_count = 0
    for r in records:
        split = r.get("split", "train")
        if split not in valid_splits:
            # Map unknown split to 'train' (defensive handling)
            missing_count += 1
            split = valid_splits[0]

        img_rel = r["image_path"]
        img_abs = os.path.join(image_root, img_rel)

        sample = {
            "image": img_abs,  # datasets.Image will load from this path and store bytes
            "is_fake": int(r["is_fake"]),
            # Use -1 for real images (None in metadata)
            "family_id": -1 if r.get("family_id") is None else int(r["family_id"]),
            "version_id": -1 if r.get("version_id") is None else int(r["version_id"]),
            "split": split,
        }
        splits[split].append(sample)

    total_after = sum(len(v) for v in splits.values())
    print("Samples per split (before HF construction):")
    for s in valid_splits:
        print(f"  {s}: {len(splits[s])}")
    if missing_count > 0:
        print(f"[Info] {missing_count} samples had unknown split; mapped to '{valid_splits[0]}'")

    # Define HF features. We keep labels as numeric for direct training use.
    features = Features(
        {
            "image": Image(),
            "is_fake": Value("int8"),
            "family_id": Value("int32"),
            "version_id": Value("int32"),
            "split": Value("string"),
        }
    )

    hf_splits: Dict[str, Dataset] = {}
    for split_name, rows in splits.items():
        if not rows:
            print(f"[Warning] split '{split_name}' has 0 samples; skipping.")
            continue
        print(f"Building HF Dataset for split '{split_name}' with {len(rows)} samples...")
        hf_splits[split_name] = Dataset.from_list(rows, features=features)

    if not hf_splits:
        raise RuntimeError("No non-empty splits to build HF Dataset from. Check your metadata.")

    hf_ds = DatasetDict(hf_splits)

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving HF DatasetDict to {output_path} ...")
    hf_ds.save_to_disk(output_path)
    print("Done. You can now load it via datasets.load_from_disk(output_path).")


if __name__ == "__main__":
    main()
