import json
import os
from pathlib import Path
from typing import Dict, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk


class ProfilingDataset(Dataset):
    """Dataset for BHEP Profiler training with hierarchical labels.

    This is the original JSON + image-folder based dataset.
    """

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        config: Dict = None,
    ):
        """\
        Args:
            metadata_path: Path to dflip3k_meta.json
            image_root: Root directory for images
            transform: Torchvision-style transform to apply to PIL images
            split: "train", "val", or "test"
            config: Configuration dict
        """

        self.metadata_path = metadata_path
        self.image_root = image_root
        self.transform = transform
        self.split = split
        self.config = config or {}

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.all_records = json.load(f)

        # Filter by split
        self.records = [r for r in self.all_records if r.get("split", "train") == split]

        print(f"Loaded {len(self.records)} samples for split '{split}' from {metadata_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Load image
        image_path = os.path.join(self.image_root, record["image_path"])
        image = Image.open(image_path).convert("RGB")

        # Process image using torchvision transforms
        if self.transform is not None:
            pixel_values = self.transform(image)  # (3, H, W)
        else:
            # Fallback: basic ToTensor without normalization
            pixel_values = transforms.ToTensor()(image)

        # Get labels
        is_fake = torch.tensor(record["is_fake"], dtype=torch.long)

        # Get family and version labels (with -1 as placeholder for real images)
        family_id = record.get("family_id", -1)
        if family_id is None:
            family_id = -1
        family_id = torch.tensor(family_id, dtype=torch.long)

        version_id = record.get("version_id", -1)
        if version_id is None:
            version_id = -1
        version_id = torch.tensor(version_id, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "is_fake": is_fake,
            "family_ids": family_id,
            "version_ids": version_id,
        }


class HFProfilingDataset(Dataset):
    """Dataset wrapper on top of a HuggingFace `datasets.Dataset`.

    Expected HF dataset schema (per sample):
      - image: PIL.Image or image path (from `datasets.Image()`)
      - is_fake: int (0 or 1)
      - family_id: int (>=0 for fake, -1 for real)
      - version_id: int (>=0 for fake, -1 for real)
    """

    def __init__(
        self,
        hf_dataset,
        transform: Optional[Callable] = None,
        config: Dict = None,
    ):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.config = config or {}

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        record = self.hf_dataset[int(idx)]

        # HF `Image` feature usually returns a PIL.Image.
        image = record["image"]
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            # Defensive: if it's a path for some reason
            img = Image.open(image).convert("RGB")

        if self.transform is not None:
            pixel_values = self.transform(img)
        else:
            pixel_values = transforms.ToTensor()(img)

        is_fake = torch.tensor(int(record.get("is_fake", 0)), dtype=torch.long)

        family_id_val = record.get("family_id", -1)
        if family_id_val is None:
            family_id_val = -1
        family_id = torch.tensor(int(family_id_val), dtype=torch.long)

        version_id_val = record.get("version_id", -1)
        if version_id_val is None:
            version_id_val = -1
        version_id = torch.tensor(int(version_id_val), dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "is_fake": is_fake,
            "family_ids": family_id,
            "version_ids": version_id,
        }


def create_dataloader(
    metadata_path: str,
    image_root: str,
    transform=None,
    task_mode: str = "profiling",
    batch_size: int = 8,
    split: str = "train",
    config: Dict = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """\
    Create a DataLoader for the specified task and split.

    Args:
        metadata_path: Path to metadata JSON file
        image_root: Root directory for images
        transform: Image transform
        task_mode: Task mode ("profiling" for BHEP training)
        batch_size: Batch size
        split: "train", "val", or "test"
        config: Configuration dict
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """

    if task_mode != "profiling":
        raise ValueError(f"Unknown task_mode: {task_mode}")

    config = config or {}
    data_cfg = config.get("data", {})
    use_hf = bool(data_cfg.get("use_hf_dataset", False))

    if use_hf:
        hf_root = data_cfg.get("hf_dataset_path")
        if hf_root is None:
            raise ValueError("data.use_hf_dataset=True but data.hf_dataset_path is not set in config.")

        ds_dict = load_from_disk(hf_root)
        if split not in ds_dict:
            raise ValueError(f"Split '{split}' not found in HF dataset at {hf_root}.")
        hf_split = ds_dict[split]

        dataset: Dataset = HFProfilingDataset(
            hf_dataset=hf_split,
            transform=transform,
            config=config,
        )
    else:
        dataset = ProfilingDataset(
            metadata_path=metadata_path,
            image_root=image_root,
            transform=transform,
            split=split,
            config=config,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )

    return dataloader


def create_profiling_dataloaders(config: Dict, train_transform, val_transform):
    """Create train and val dataloaders for single-GPU training.

    This will respect `config["data"]["use_hf_dataset"]` to choose between
    JSON+folder and HuggingFace Datasets backends.
    """

    train_loader = create_dataloader(
        metadata_path=config["data"]["metadata_path"],
        image_root=config["data"]["image_root"],
        transform=train_transform,
        task_mode="profiling",
        batch_size=config["training"]["batch_size"],
        split="train",
        config=config,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )

    val_loader = create_dataloader(
        metadata_path=config["data"]["metadata_path"],
        image_root=config["data"]["image_root"],
        transform=val_transform,
        task_mode="profiling",
        batch_size=config["training"]["batch_size"],
        split="val",
        config=config,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )

    return train_loader, val_loader


def create_profiling_dataloaders_ddp(
    config: Dict,
    train_transform,
    val_transform,
    *,
    world_size: int,
    rank: int,
):
    """Create train and val dataloaders with samplers for DDP training.

    This will respect `config["data"]["use_hf_dataset"]` similarly to
    `create_profiling_dataloaders`.
    """

    data_cfg = config.get("data", {})
    use_hf = bool(data_cfg.get("use_hf_dataset", False))

    if use_hf:
        hf_root = data_cfg.get("hf_dataset_path")
        if hf_root is None:
            raise ValueError("data.use_hf_dataset=True but data.hf_dataset_path is not set in config.")

        ds_dict = load_from_disk(hf_root)
        if "train" not in ds_dict or "val" not in ds_dict:
            raise ValueError(f"HF dataset at {hf_root} must contain 'train' and 'val' splits for DDP.")

        train_dataset = HFProfilingDataset(ds_dict["train"], transform=train_transform, config=config)
        val_dataset = HFProfilingDataset(ds_dict["val"], transform=val_transform, config=config)
    else:
        train_dataset = ProfilingDataset(
            metadata_path=config["data"]["metadata_path"],
            image_root=config["data"]["image_root"],
            transform=train_transform,
            split="train",
            config=config,
        )
        val_dataset = ProfilingDataset(
            metadata_path=config["data"]["metadata_path"],
            image_root=config["data"]["image_root"],
            transform=val_transform,
            split="val",
            config=config,
        )

    # Create DistributedSamplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    per_device_batch_size = config["training"]["batch_size"]

    # Create DataLoaders with samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        sampler=train_sampler,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        sampler=val_sampler,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )

    return train_loader, val_loader, train_sampler, val_sampler


__all__ = [
    "ProfilingDataset",
    "HFProfilingDataset",
    "create_dataloader",
    "create_profiling_dataloaders",
    "create_profiling_dataloaders_ddp",
]
