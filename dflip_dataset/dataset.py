"""
DFLIP Dataset Module
Implements the core DFLIPDataset class with dual task modes for two-stage training.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Literal
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoProcessor

from .formatting import format_stage1_output, format_stage2_conversation


class DFLIPDataset(Dataset):
    """
    DFLIP Dataset for Linguistic Profiling of Deepfakes.
    
    Supports two task modes:
    - 'profiling': Returns image + labels for Stage 1 (Detection, Identification, Localization)
    - 'interpreting': Returns image + formatted prompts for Stage 2 (Prompt Prediction)
    
    Args:
        metadata_path: Path to dflip3k_meta.json
        image_root: Root directory for images
        mask_root: Root directory for segmentation masks
        task_mode: 'profiling' or 'interpreting'
        processor: Qwen VL processor for image preprocessing
        split: 'train', 'val', or 'test'
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        mask_root: Optional[str] = None,
        task_mode: Literal['profiling', 'interpreting'] = 'profiling',
        processor: Optional[AutoProcessor] = None,
        split: str = 'train',
        config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.metadata_path = Path(metadata_path)
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root) if mask_root else None
        self.task_mode = task_mode
        self.processor = processor
        self.split = split
        self.config = config or {}
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Build generator ID mapping
        self.generator_to_id = self._build_generator_mapping()
        self.num_generators = len(self.generator_to_id)
        
        print(f"Loaded DFLIP dataset: {len(self.metadata)} samples")
        print(f"Task mode: {task_mode}, Split: {split}")
        print(f"Number of generators: {self.num_generators}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load and filter metadata based on split."""
        with open(self.metadata_path, 'r') as f:
            data = json.load(f)
        
        # TODO: Implement proper train/val/test splitting based on config
        # For now, return all data
        return data
    
    def _build_generator_mapping(self) -> Dict[str, int]:
        """Build mapping from generator name to ID."""
        generators = set()
        for item in self.metadata:
            if item.get('generator'):
                generators.add(item['generator'])
        
        # Add 'Real' as ID 0
        mapping = {'Real': 0}
        for idx, gen in enumerate(sorted(generators), start=1):
            mapping[gen] = idx
        
        return mapping
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns data based on task_mode:
        
        Profiling mode (Stage 1):
            {
                'image': PIL.Image or Tensor,
                'is_fake': 0 or 1,
                'generator_id': int (0 for real, 1-N for fake models),
                'mask': np.array or None (forgery localization)
            }
        
        Interpreting mode (Stage 2):
            {
                'image': PIL.Image or Tensor,
                'messages': List (conversation format for LLM),
                'text': str (formatted prompt for processor)
            }
        """
        item = self.metadata[idx]
        
        # Load image
        image_path = self.image_root / item['file_path']
        image = Image.open(image_path).convert('RGB')
        
        if self.task_mode == 'profiling':
            return self._get_profiling_sample(item, image)
        else:  # interpreting
            return self._get_interpreting_sample(item, image)
    
    def _get_profiling_sample(self, item: Dict, image: Image.Image) -> Dict:
        """Prepare sample for Stage 1 (Profiler) training."""
        # Binary label: 0=Real, 1=Fake
        is_fake = 1 if item['label'] == 'Fake' else 0
        
        # Generator ID
        if item['label'] == 'Real':
            generator_id = 0
        else:
            generator_name = item['generator']
            generator_id = self.generator_to_id.get(generator_name, 0)
        
        # Load mask if available
        mask = None
        if item.get('mask_path') and self.mask_root:
            mask_path = self.mask_root / item['mask_path']
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert('L'))
                mask = (mask > 128).astype(np.float32)  # Binarize
        
        sample = {
            'image': image,
            'is_fake': is_fake,
            'generator_id': generator_id,
            'mask': mask,
            'uid': item['uid']
        }
        
        return sample
    
    def _get_interpreting_sample(self, item: Dict, image: Image.Image) -> Dict:
        """Prepare sample for Stage 2 (Interpreter) training."""
        # Get Stage 1 outputs (ground truth for conditioning)
        is_fake = item['label'] == 'Fake'
        generator = item.get('generator', 'Unknown')
        gt_prompt = item.get('gt_prompt', '')
        
        # Format conversation for SFT
        messages = format_stage2_conversation(
            is_fake=is_fake,
            generator=generator,
            gt_prompt=gt_prompt,
            include_assistant=True  # Include for training
        )
        
        sample = {
            'image': image,
            'messages': messages,
            'uid': item['uid']
        }
        
        return sample
    
    def collate_fn_profiling(self, batch: List[Dict]) -> Dict:
        """Custom collate function for profiling mode."""
        images = [item['image'] for item in batch]
        is_fake = torch.tensor([item['is_fake'] for item in batch], dtype=torch.long)
        generator_ids = torch.tensor([item['generator_id'] for item in batch], dtype=torch.long)
        
        # Process images with Qwen processor
        if self.processor:
            processed = self.processor(images=images, return_tensors='pt')
            pixel_values = processed['pixel_values']
        else:
            # Fallback: simple tensor conversion
            pixel_values = torch.stack([torch.from_numpy(np.array(img)) for img in images])
            pixel_values = pixel_values.permute(0, 3, 1, 2).float() / 255.0
        
        # Handle masks
        masks = []
        for item in batch:
            if item['mask'] is not None:
                masks.append(torch.from_numpy(item['mask']))
            else:
                # Create dummy mask
                masks.append(torch.zeros(pixel_values.shape[-2:]))
        
        masks = torch.stack(masks).unsqueeze(1)  # (B, 1, H, W)
        
        return {
            'pixel_values': pixel_values,
            'is_fake': is_fake,
            'generator_ids': generator_ids,
            'masks': masks,
            'uids': [item['uid'] for item in batch]
        }
    
    def collate_fn_interpreting(self, batch: List[Dict]) -> Dict:
        """Custom collate function for interpreting mode (Stage 2)."""
        images = [item['image'] for item in batch]
        messages_batch = [item['messages'] for item in batch]
        
        # Process with Qwen VL processor
        # The processor will handle both image and text
        texts = []
        for messages in messages_batch:
            # Convert messages to Qwen VL format
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Process images and texts together
        processed = self.processor(
            images=images,
            text=texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        return {
            **processed,
            'uids': [item['uid'] for item in batch]
        }
    
    def get_collate_fn(self):
        """Return appropriate collate function based on task mode."""
        if self.task_mode == 'profiling':
            return self.collate_fn_profiling
        else:
            return self.collate_fn_interpreting


def create_dataloader(
    metadata_path: str,
    image_root: str,
    processor: AutoProcessor,
    task_mode: Literal['profiling', 'interpreting'],
    batch_size: int,
    split: str = 'train',
    config: Optional[Dict] = None,
    **kwargs
):
    """
    Convenience function to create DataLoader with appropriate settings.
    
    Args:
        metadata_path: Path to metadata JSON
        image_root: Root directory for images
        processor: Qwen VL processor
        task_mode: 'profiling' or 'interpreting'
        batch_size: Batch size
        split: Data split
        config: Configuration dictionary
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader
    
    config = config or {}
    mask_root = config.get('data', {}).get('mask_root')
    
    dataset = DFLIPDataset(
        metadata_path=metadata_path,
        image_root=image_root,
        mask_root=mask_root,
        task_mode=task_mode,
        processor=processor,
        split=split,
        config=config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=dataset.get_collate_fn(),
        num_workers=config.get('data', {}).get('num_workers', 4),
        pin_memory=config.get('data', {}).get('pin_memory', True),
        **kwargs
    )
    
    return dataloader
