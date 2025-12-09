#!/usr/bin/env python3
"""Generate dflip3k_meta.json from /home/data/yabin/DFLIP3K/{fake, real_pair}.

This script adapts the original generate_dataset_meta.py to your real
DFLIP3K layout:

Fake images (with family/version labels):
    /home/data/yabin/DFLIP3K/fake/
        aura_flow/...
        flux2-dev/...
        illustrious/T-Illunai/*.jpeg
        ...

Real images (paired, but pairing is not required by this script):
    /home/data/yabin/DFLIP3K/real_pair/
        illustrious/T-Illunai/*.jpeg
        ...

The script builds hierarchical labels for fake images:
- family_id: index of the top-level directory under fake/ (sorted by name)
- version_id: global index of (family_name, submodel_name) pairs

Real images are labeled as:
- is_fake = 0
- family_id = null
- version_id = null

The output JSON is fully compatible with dataset/profiling_dataset.py.

Example usage:
    python tools/generate_dataset_meta_dflip3k.py \
        --fake-root /home/data/yabin/DFLIP3K/fake \
        --real-root /home/data/yabin/DFLIP3K/real_pair \
        --image-root /home/data/yabin/DFLIP3K \
        --output ./assets/dflip3k_meta.json \
        --train-ratio 0.8 \
        --seed 42

After running, this script prints suggested BHEP config values:
- num_families
- num_versions
- hierarchy (list of version_ids for each family)

You can copy these into configs/dflip_config.yaml.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dflip3k_meta.json for DFLIP3K-style dataset"
    )
    parser.add_argument(
        "--fake-root",
        type=str,
        required=True,
        help="Root of fake images, e.g. /home/data/yabin/DFLIP3K/fake",
    )
    parser.add_argument(
        "--real-root",
        type=str,
        required=True,
        help="Root of real images, e.g. /home/data/yabin/DFLIP3K/real_pair",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help=(
            "Common image_root used in training, "
            "must be a parent of both fake_root and real_root"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./assets/dflip3k_meta.json",
        help="Output path for metadata JSON",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help=(
            "Train/test split ratio. "
            "This script only produces 'train' and 'test' splits; "
            "validation data should be sampled from the train split in training code."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    """Return True if the path is a file with a supported image extension."""

    return path.is_file() and path.suffix.lower() in IMG_EXTS


def collect_image_files(root: Path) -> List[Path]:
    """Recursively collect image files under a root directory."""

    files: List[Path] = []
    if not root.exists():
        print(f"[Warning] root not found: {root}")
        return files

    for p in root.rglob("*"):
        if is_image_file(p):
            files.append(p)
    return files


def build_label_maps(
    fake_root: Path,
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int], List[List[int]]]:
    """Build family_id / version_id mappings and hierarchy from fake_root.

    family_id: index of the top-level directory under fake_root (sorted by name).
    version_id: global index assigned to each (family_name, submodel_name) pair.
    hierarchy: list of [version_id] for each family (same order as family_ids).
    """

    family_dirs = [p for p in fake_root.iterdir() if p.is_dir()]
    family_names = sorted([p.name for p in family_dirs])
    if not family_names:
        raise RuntimeError(f"No family directories found under {fake_root}")

    family_to_id: Dict[str, int] = {name: i for i, name in enumerate(family_names)}

    version_id_map: Dict[Tuple[str, str], int] = {}
    hierarchy: List[List[int]] = []
    current_vid = 0

    print("Families (sorted):")
    for fam_name in family_names:
        print(f"  - {fam_name}")

    for fam_name in family_names:
        fam_dir = fake_root / fam_name
        sub_dirs = [p for p in fam_dir.iterdir() if p.is_dir()]
        sub_names = sorted([p.name for p in sub_dirs])
        fam_vids: List[int] = []
        if not sub_names:
            print(f"[Warning] family {fam_name} has no submodel directories")

        for sub_name in sub_names:
            version_id_map[(fam_name, sub_name)] = current_vid
            fam_vids.append(current_vid)
            current_vid += 1
        hierarchy.append(fam_vids)

    print(f"\nTotal families: {len(family_names)}")
    print(f"Total versions: {current_vid}")

    return family_to_id, version_id_map, hierarchy


def build_metadata(args):
    """Scan fake/real roots and build metadata records and hierarchy info."""

    fake_root = Path(args.fake_root).resolve()
    real_root = Path(args.real_root).resolve()
    image_root = Path(args.image_root).resolve()

    if not fake_root.exists():
        raise RuntimeError(f"Fake root not found: {fake_root}")
    if not real_root.exists():
        print(f"[Warning] real root not found: {real_root} (will have only fake samples)")

    # Build label maps from fake_root
    family_to_id, version_id_map, hierarchy = build_label_maps(fake_root)
    num_families = len(family_to_id)
    num_versions = sum(len(v) for v in hierarchy)

    records: List[Dict] = []

    # Fake images
    fake_files = collect_image_files(fake_root)
    print(f"\nFound {len(fake_files)} fake images")

    for img_path in fake_files:
        # Expected structure: fake_root/<family>/<submodel>/<file>
        rel_to_fake = img_path.relative_to(fake_root)
        if len(rel_to_fake.parts) < 3:
            # Skip files not under family/submodel/
            print(f"[Warning] skip fake image with unexpected depth: {img_path}")
            continue

        fam_name = rel_to_fake.parts[0]
        sub_name = rel_to_fake.parts[1]

        if fam_name not in family_to_id:
            print(f"[Warning] unknown family {fam_name} for {img_path}, skip")
            continue

        key = (fam_name, sub_name)
        if key not in version_id_map:
            print(f"[Warning] unknown (family, submodel) {key} for {img_path}, skip")
            continue

        fam_id = family_to_id[fam_name]
        ver_id = version_id_map[key]

        try:
            rel_to_image_root = img_path.relative_to(image_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"Image {img_path} is not under image_root={image_root}. "
                f"Please ensure image_root is a parent of fake_root/real_root."
            ) from exc

        records.append(
            {
                "image_path": rel_to_image_root,
                "is_fake": 1,
                "family_id": fam_id,
                "version_id": ver_id,
                "split": "train",  # placeholder, will be set in split_dataset
            }
        )

    # Real images
    real_files = collect_image_files(real_root) if real_root.exists() else []
    print(f"Found {len(real_files)} real images")

    for img_path in real_files:
        try:
            rel_to_image_root = img_path.relative_to(image_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"Image {img_path} is not under image_root={image_root}. "
                f"Please ensure image_root is a parent of fake_root/real_root."
            ) from exc

        records.append(
            {
                "image_path": rel_to_image_root,
                "is_fake": 0,
                "family_id": None,
                "version_id": None,
                "split": "train",  # placeholder
            }
        )

    print(f"Total images (real + fake): {len(records)}")

    return records, hierarchy, num_families, num_versions


def split_dataset(records: List[Dict], train_ratio: float = 0.8, seed: int = 42) -> List[Dict]:
    """Split into train/test according to train_ratio.

    This function no longer creates a separate "val" split. If you need a
    validation set, you should sample it from the "train" split externally
    (e.g. in the training/dataloader code).
    """

    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    if n == 0:
        raise RuntimeError("No records to split.")

    train_end = int(train_ratio * n)

    for i, r in enumerate(records):
        if i < train_end:
            r["split"] = "train"
        else:
            r["split"] = "test"

    # Print statistics
    cnt = {"train": 0, "test": 0}
    for r in records:
        cnt[r["split"]] += 1

    total = float(n)
    print("\nDataset split:")
    for k in ["train", "test"]:
        v = cnt[k]
        print(f"  {k}: {v} ({100.0 * v / total:.1f}%)")

    return records


def main():
    args = parse_args()
    records, hierarchy, num_families, num_versions = build_metadata(args)
    records = split_dataset(records, train_ratio=args.train_ratio, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nMetadata saved to {out_path}")

    print("\n=== BHEP config suggestion ===")
    print(f"num_families: {num_families}")
    print(f"num_versions: {num_versions}")
    print("hierarchy (family_idx -> version_ids):")
    for fam_idx, fam_vids in enumerate(hierarchy):
        print(f"  family {fam_idx}: {fam_vids}")


if __name__ == "__main__":
    main()
