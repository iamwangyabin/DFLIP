# OpenSDI Dataset for DFLIP

This document explains how to use the OpenSDI dataset reader with the DFLIP project.

## Dataset Information

- **HuggingFace Path**: `nebula/OpenSDI_train`
- **Format**: Image classification for spotting diffusion-generated images
- **Split**: `sd15` (200,824 examples, ~34.79 GB)
- **Fields**:
  - `key`: String identifier
  - `image`: PIL Image
  - `mask`: PIL Image (manipulation mask)
  - `label`: Binary label (0=Real, 1=Fake)

## Quick Start

### 1. Basic Usage - Profiling Mode (Stage 1)

```python
from transformers import AutoProcessor
from dflip_dataset import OpenSDIDataset, create_opensdi_dataloader

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Create dataset
dataset = OpenSDIDataset(
    dataset_path="nebula/OpenSDI_train",
    split_name="sd15",
    task_mode='profiling',
    processor=processor,
    output_size=(1024, 1024)
)

# Get a sample
sample = dataset[0]
print(f"Image: {sample['image'].shape}")
print(f"Mask: {sample['mask'].shape}")
print(f"Label: {sample['label']}")
print(f"Generator ID: {sample['generator_id']}")
```

### 2. DataLoader for Training

```python
# Create dataloader
dataloader = create_opensdi_dataloader(
    dataset_path="nebula/OpenSDI_train",
    split_name="sd15",
    task_mode='profiling',
    processor=processor,
    batch_size=8,
    num_workers=4
)

# Iterate over batches
for batch in dataloader:
    images = batch['pixel_values']  # (B, C, H, W)
    masks = batch['masks']          # (B, 1, H, W)
    labels = batch['is_fake']       # (B,)
    generator_ids = batch['generator_ids']  # (B,)
```

### 3. Interpreting Mode (Stage 2)

```python
# Create dataset for Stage 2
dataset = OpenSDIDataset(
    dataset_path="nebula/OpenSDI_train",
    split_name="sd15",
    task_mode='interpreting',
    processor=processor
)

# Get a sample
sample = dataset[0]
print(f"Image: {type(sample['image'])}")
print(f"Messages: {sample['messages']}")
```

## Advanced Options

### Filter Pixel-Level Manipulations Only

```python
dataset = OpenSDIDataset(
    dataset_path="nebula/OpenSDI_train",
    split_name="sd15",
    task_mode='profiling',
    processor=processor,
    pixel_only=True  # Filter for pixel-level manipulations
)
```

### Add Data Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Common transforms (applied to both image and mask)
common_transforms = A.Compose([
    A.RandomResizedCrop(height=1024, width=1024, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
])

# Post transforms (applied after common transforms)
post_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

dataset = OpenSDIDataset(
    dataset_path="nebula/OpenSDI_train",
    split_name="sd15",
    task_mode='profiling',
    processor=processor,
    common_transforms=common_transforms,
    post_transform=post_transform
)
```

### Edge Mask Generation

```python
dataset = OpenSDIDataset(
    dataset_path="nebula/OpenSDI_train",
    split_name="sd15",
    task_mode='profiling',
    processor=processor,
    edge_width=5  # Generate edge masks with 5-pixel width
)

sample = dataset[0]
edge_mask = sample['edge_mask']  # Additional edge mask
```

## Integration with DFLIP Training

### Modify `train_profiler.py`

Replace the dataset creation with OpenSDI:

```python
from dflip_dataset import create_opensdi_dataloader

# Create training dataloader
train_loader = create_opensdi_dataloader(
    dataset_path="nebula/OpenSDI_train",
    split_name="sd15",
    task_mode='profiling',
    processor=processor,
    batch_size=config['stage1_training']['batch_size'],
    num_workers=config['data']['num_workers'],
    config=config
)
```

## Output Format

### Profiling Mode Sample
```python
{
    'image': Tensor[C, H, W],           # Image tensor
    'mask': Tensor[1, H, W],            # Binary mask
    'label': int,                        # 0 or 1
    'generator_id': int,                 # Generator model ID
    'shape': Tensor[2],                  # Original shape (H, W)
    'name': str,                         # Sample identifier
    'edge_mask': Tensor[1, H, W]        # Optional edge mask
}
```

### Interpreting Mode Sample
```python
{
    'image': PIL.Image,                  # PIL image
    'messages': List[Dict],              # Conversation format
    'uid': str                           # Sample identifier
}
```

## Requirements

```bash
pip install datasets transformers torch pillow numpy
# Optional for transforms
pip install albumentations scipy
```

## Example Script

Run the example script to test the dataset:

```bash
python examples/opensdi_dataset_example.py
```

## Notes

- The dataset automatically builds a generator ID mapping from the data
- Real images are assigned generator ID 0
- Fake images are assigned IDs 1-N based on their generator model
- The `key` field is used to extract generator information if explicit generator field is not available
- Masks are automatically binarized using a threshold of 127.5
