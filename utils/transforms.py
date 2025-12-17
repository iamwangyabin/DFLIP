"""Transforms / data augmentation utilities for DFLIP.

This module centralizes image transforms so they can be reused across
training scripts or notebooks and kept consistent.
"""
import importlib
import random
from typing import Dict, Tuple, List, Any, Union
import cv2
import numpy as np
from PIL import Image, ImageFilter

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def import_class(class_path: str):
    """Dynamically import a class from a string path.
    
    Args:
        class_path: Full path to class, e.g., 'torchvision.transforms.Resize'
        
    Returns:
        The imported class
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class DataAugment:
    """Custom data augmentation for deepfake detection.
    
    Implements blur and JPEG compression augmentations commonly used
    in deepfake detection research.
    """
    
    def __init__(
        self,
        blur_prob: float = 0.1,
        blur_sig: List[float] = [0.0, 3.0],
        jpg_prob: float = 0.1,
        jpg_method: List[str] = ['cv2', 'pil'],
        jpg_qual: List[int] = [30, 100],
    ):
        """Initialize DataAugment.
        
        Args:
            blur_prob: Probability of applying blur augmentation
            blur_sig: Range of blur sigma values [min, max]
            jpg_prob: Probability of applying JPEG compression
            jpg_method: List of JPEG compression methods to choose from
            jpg_qual: Range of JPEG quality values [min, max]
        """
        self.blur_prob = blur_prob
        self.blur_sig = blur_sig
        self.jpg_prob = jpg_prob
        self.jpg_method = jpg_method
        self.jpg_qual = jpg_qual
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply augmentations to PIL Image.
        
        Args:
            image: PIL Image to augment
            
        Returns:
            Augmented PIL Image
        """
        # Apply blur augmentation
        if random.random() < self.blur_prob:
            image = self._apply_blur(image)
        
        # Apply JPEG compression augmentation
        if random.random() < self.jpg_prob:
            image = self._apply_jpeg_compression(image)
        
        return image
    
    def _apply_blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur to image.
        
        Args:
            image: PIL Image to blur
            
        Returns:
            Blurred PIL Image
        """
        sigma = random.uniform(self.blur_sig[0], self.blur_sig[1])
        if sigma > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image
    
    def _apply_jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Apply JPEG compression to image.
        
        Args:
            image: PIL Image to compress
            
        Returns:
            Compressed PIL Image
        """
        quality = random.randint(self.jpg_qual[0], self.jpg_qual[1])
        method = random.choice(self.jpg_method)
        
        if method == 'cv2':
            return self._jpeg_compress_cv2(image, quality)
        else:  # pil
            return self._jpeg_compress_pil(image, quality)
    
    def _jpeg_compress_cv2(self, image: Image.Image, quality: int) -> Image.Image:
        """Apply JPEG compression using OpenCV.
        
        Args:
            image: PIL Image to compress
            quality: JPEG quality (0-100)
            
        Returns:
            Compressed PIL Image
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', img_bgr, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # Convert back to PIL
        img_rgb = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    def _jpeg_compress_pil(self, image: Image.Image, quality: int) -> Image.Image:
        """Apply JPEG compression using PIL.
        
        Args:
            image: PIL Image to compress
            quality: JPEG quality (0-100)
            
        Returns:
            Compressed PIL Image
        """
        import io
        
        # Save to bytes buffer with JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load compressed image
        return Image.open(buffer)


def build_transforms_from_config(config: Dict, split: str = "train") -> transforms.Compose:
    """Build transform pipeline from configuration.
    
    Args:
        config: Configuration dictionary containing transform specifications
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        Composed transform pipeline
        
    Example config format:
        config["data"]["transforms"][split] = [
            {"_target_": "torchvision.transforms.Resize", "size": 256},
            {"_target_": "torchvision.transforms.RandomResizedCrop", "size": 224},
            {"_target_": "utils.transforms.DataAugment", "blur_prob": 0.1},
            {"_target_": "torchvision.transforms.ToTensor"},
            {"_target_": "torchvision.transforms.Normalize",
             "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        ]
    """
    data_cfg = config.get("data", {})
    
    # Check if new transform format is available
    if "transforms" in data_cfg and split in data_cfg["transforms"]:
        transform_configs = data_cfg["transforms"][split]
        transforms_list = []
        
        for transform_config in transform_configs:
            # Make a copy to avoid modifying original config
            config_copy = transform_config.copy()
            target = config_copy.pop("_target_")
            
            # Handle special case for utils.transforms classes
            if target.startswith("utils.transforms."):
                class_name = target.split(".")[-1]
                if class_name == "DataAugment":
                    transform_class = DataAugment
                else:
                    raise ValueError(f"Unknown utils.transforms class: {class_name}")
            else:
                # Import class dynamically
                transform_class = import_class(target)
            
            # Create transform instance
            transform = transform_class(**config_copy)
            transforms_list.append(transform)
        
        return transforms.Compose(transforms_list)
    
    # Fallback to legacy format
    else:
        return build_train_val_transforms(config)[0 if split == "train" else 1]


def build_legacy_transforms_from_config(config: Dict, split: str = "train") -> transforms.Compose:
    """Build transforms using legacy augmentation config format.
    
    This is a wrapper around the original build_train_val_transforms function
    for backward compatibility.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        Composed transform pipeline
    """
    train_transform, val_transform = build_train_val_transforms(config)
    
    if split == "train":
        return train_transform
    else:
        return val_transform


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
