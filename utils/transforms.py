"""Transforms / data augmentation utilities for DFLIP.

This module centralizes image transforms so they can be reused across
training scripts or notebooks and kept consistent.
"""
from typing import Dict, Tuple

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_val_transforms(config: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train & validation transforms from config.

    Expected config structure::

        config["data"]["image_size"]: int (optional, default 224)
        config["data"]["augmentation"]: {
            "horizontal_flip": bool,
            "rotation_range": int,
            "brightness_range": float,
            "contrast_range": float,
        }

    Returns
    -------
    train_transform : torchvision.transforms.Compose
        Transform pipeline with data augmentation for training.
    val_transform : torchvision.transforms.Compose
        Deterministic transform pipeline for validation.
    """
    data_cfg = config["data"]
    image_size = data_cfg.get("image_size", 224)
    aug_cfg = data_cfg.get("augmentation", {})

    horizontal_flip = aug_cfg.get("horizontal_flip", False)
    rotation_range = aug_cfg.get("rotation_range", 0)
    brightness_range = aug_cfg.get("brightness_range", 0.0)
    contrast_range = aug_cfg.get("contrast_range", 0.0)

    # Train transforms with augmentation
    train_tfms = [transforms.RandomResizedCrop(image_size)]

    if horizontal_flip:
        train_tfms.append(transforms.RandomHorizontalFlip())

    if rotation_range and rotation_range > 0:
        train_tfms.append(transforms.RandomRotation(rotation_range))

    if (brightness_range and brightness_range > 0.0) or (contrast_range and contrast_range > 0.0):
        train_tfms.append(
            transforms.ColorJitter(
                brightness=brightness_range if brightness_range > 0.0 else 0,
                contrast=contrast_range if contrast_range > 0.0 else 0,
            )
        )

    train_tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_transform = transforms.Compose(train_tfms)

    # Validation transforms: deterministic, no heavy augmentation
    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, val_transform
