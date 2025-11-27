"""
OpenSDI Dataset Module for DFLIP
Adapted from HuggingFace dataset format for the DFLIP project.
Dataset: nebula/OpenSDI_train
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Literal

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoProcessor


class OpenSDIDataset(Dataset):
    """
    OpenSDI Dataset for DFLIP project.
    Loads data from HuggingFace dataset: nebula/OpenSDI_train
    
    Supports two task modes:
    - 'profiling': Returns image + labels for Stage 1 (Detection, Identification, Localization)
    - 'interpreting': Returns image + formatted prompts for Stage 2 (Prompt Prediction)
    
    Args:
        dataset_path: HuggingFace dataset path (default: "nebula/OpenSDI_train")
        split_name: Dataset split name (e.g., 'train', 'validation', 'test')
        task_mode: 'profiling' or 'interpreting'
        processor: Qwen VL processor for image preprocessing
        pixel_only: If True, filter to keep only pixel-level manipulated images
        output_size: Target output size for images (H, W)
        common_transforms: Albumentations transforms to apply to both image and mask
        post_transform: Post-processing transforms for images
        edge_width: Width for edge mask generation (None to disable)
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        dataset_path: str = "nebula/OpenSDI_train",
        split_name: str = "train",
        task_mode: Literal['profiling', 'interpreting'] = 'profiling',
        processor: Optional[AutoProcessor] = None,
        pixel_only: bool = False,
        output_size: tuple = (1024, 1024),
        common_transforms=None,
        post_transform=None,
        edge_width: Optional[int] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.split_name = split_name
        self.task_mode = task_mode
        self.processor = processor
        self.output_size = output_size
        self.common_transforms = common_transforms
        self.post_transform = post_transform
        self.edge_width = edge_width
        self.config = config or {}
        
        # Load dataset from HuggingFace
        print(f"Loading dataset from HuggingFace: {dataset_path}...")
        self.dataset = load_dataset(dataset_path)[split_name]
        
        # Filter for pixel-level manipulations if requested
        if pixel_only:
            print("Filtering for pixel-level manipulations...")
            filtered_indices = [
                i for i, sample in enumerate(self.dataset)
                if self._is_pixel_manipulation(sample)
            ]
            self.dataset = self.dataset.select(filtered_indices)
            print(f"Filtered to {len(filtered_indices)} pixel-level samples")
        
        # Build generator mapping
        self.generator_to_id = self._build_generator_mapping()
        self.num_generators = len(self.generator_to_id)
        
        print(f"Loaded OpenSDI dataset: {len(self.dataset)} samples")
        print(f"Task mode: {task_mode}, Split: {split_name}")
        print(f"Number of generators: {self.num_generators}")
    
    def _is_pixel_manipulation(self, sample: Dict) -> bool:
        """
        Determine if a sample contains pixel-level manipulation.
        Adjust this logic based on the actual OpenSDI dataset structure.
        """
        # Check if the sample has a key field
        if 'key' in sample:
            key = sample['key']
            # Check for partial/fake indicators in the key
            if ('partial' in key and 'fake' in key) or 'fake' in key:
                return True
        
        # Check if label is fake
        if 'label' in sample and sample['label'] == 1:
            return True
        
        return False
    
    def _build_generator_mapping(self) -> Dict[str, int]:
        """Build mapping from generator name to ID."""
        generators = set()
        
        for sample in self.dataset:
            # Assuming the dataset has a 'generator' or similar field
            # Adjust based on actual OpenSDI dataset structure
            if 'generator' in sample and sample['generator'] is not None:
                generators.add(sample['generator'])
            elif 'key' in sample:
                # Try to extract generator from key
                key = sample['key']
                # This is a placeholder - adjust based on actual key format
                # Example: if key is like "stable-diffusion/image_001.jpg"
                parts = key.split('/')
                if len(parts) > 1:
                    generators.add(parts[0])
        
        # Add 'Real' as ID 0
        mapping = {'Real': 0}
        for idx, gen in enumerate(sorted(generators), start=1):
            mapping[gen] = idx
        
        return mapping
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _prepare_gt_img(self, tp_img: np.ndarray, mask, label: int) -> np.ndarray:
        """
        Prepare ground truth mask image.
        
        Args:
            tp_img: Input image as numpy array
            mask: Mask from dataset (PIL Image or None)
            label: Binary label (0=Real, 1=Fake)
        
        Returns:
            Ground truth mask as numpy array (H, W, 3)
        """
        if label == 0:
            # Real image: all zeros
            return np.zeros((*tp_img.shape[:2], 3), dtype=np.uint8)
        elif label == 1 and mask is None:
            # Fake but no mask: all ones
            return np.full((*tp_img.shape[:2], 3), 255, dtype=np.uint8)
        else:
            # Have mask: convert to numpy
            return np.array(mask.convert('RGB'))
    
    def _process_masks(self, gt_img: np.ndarray) -> List[np.ndarray]:
        """
        Process ground truth mask into binary format.
        
        Args:
            gt_img: Ground truth mask (H, W, 3)
        
        Returns:
            List of masks [binary_mask, edge_mask (optional)]
        """
        # Convert to binary: average across channels and threshold
        gt_img = (np.mean(gt_img, axis=2, keepdims=True) > 127.5) * 1.0
        gt_img = gt_img.transpose(2, 0, 1)[0]  # (H, W)
        
        masks_list = [gt_img]
        
        # Generate edge mask if requested
        if self.edge_width is not None:
            # Simple edge detection using dilation
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(gt_img, iterations=self.edge_width)
            edge_mask = dilated.astype(np.float32) - gt_img
            masks_list.append(edge_mask)
        
        return masks_list
    
    def __getitem__(self, index: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            For profiling mode:
            {
                'image': Tensor,
                'mask': Tensor,
                'label': int (0 or 1),
                'generator_id': int,
                'shape': Tensor,
                'name': str,
                'edge_mask': Tensor (optional)
            }
            
            For interpreting mode:
            {
                'image': Tensor,
                'messages': List,
                'uid': str
            }
        """
        sample = self.dataset[index]
        
        # Load image
        tp_img = np.array(sample['image'].convert('RGB'))
        
        # Get label (assuming 'label' field: 0=Real, 1=Fake)
        label = sample.get('label', 0)
        
        # Load mask if available
        mask = sample.get('mask', None)
        
        # Prepare ground truth mask
        gt_img = self._prepare_gt_img(tp_img, mask, label)
        
        # Apply common transforms if available
        if self.common_transforms:
            res_dict = self.common_transforms(image=tp_img, mask=gt_img)
            tp_img, gt_img = res_dict['image'], res_dict['mask']
            # Recompute label based on transformed mask
            label = 1 if np.sum(gt_img > 0) > 0 else 0
        
        # Task mode specific processing
        if self.task_mode == 'profiling':
            return self._get_profiling_sample(sample, tp_img, gt_img, label)
        else:  # interpreting
            return self._get_interpreting_sample(sample, tp_img, label)
    
    def _get_profiling_sample(self, sample: Dict, tp_img: np.ndarray, 
                             gt_img: np.ndarray, label: int) -> Dict:
        """Prepare sample for Stage 1 (Profiler) training."""
        
        # Get generator ID
        generator_id = 0  # Default to Real
        if label == 1:
            # Extract generator from sample
            generator_name = sample.get('generator', 'Unknown')
            if generator_name == 'Unknown' and 'key' in sample:
                # Try to extract from key
                parts = sample['key'].split('/')
                if len(parts) > 1:
                    generator_name = parts[0]
            
            generator_id = self.generator_to_id.get(generator_name, 0)
        
        # Process masks
        masks_list = self._process_masks(gt_img)
        
        # Apply post-transform if available
        if self.post_transform:
            res_dict = self.post_transform(image=tp_img, masks=masks_list)
            tp_img = res_dict['image']
            masks_list = res_dict['masks']
        else:
            # Default: convert to tensor
            tp_img = torch.from_numpy(tp_img).permute(2, 0, 1).float() / 255.0
            masks_list = [torch.from_numpy(m).float() for m in masks_list]
        
        # Build data dict
        data_dict = {
            'image': tp_img,
            'mask': masks_list[0].unsqueeze(0) if isinstance(masks_list[0], torch.Tensor) else torch.from_numpy(masks_list[0]).unsqueeze(0),
            'label': label,
            'generator_id': generator_id,
            'shape': torch.tensor(self.output_size),
            'name': sample.get('key', f'sample_{index}')
        }
        
        # Add edge mask if generated
        if len(masks_list) > 1:
            edge_mask = masks_list[1]
            if isinstance(edge_mask, torch.Tensor):
                data_dict['edge_mask'] = edge_mask.unsqueeze(0)
            else:
                data_dict['edge_mask'] = torch.from_numpy(edge_mask).unsqueeze(0)
        
        return data_dict
    
    def _get_interpreting_sample(self, sample: Dict, tp_img: np.ndarray, label: int) -> Dict:
        """Prepare sample for Stage 2 (Interpreter) training."""
        
        # Get generator name
        generator_name = sample.get('generator', 'Unknown')
        if generator_name == 'Unknown' and 'key' in sample:
            parts = sample['key'].split('/')
            if len(parts) > 1:
                generator_name = parts[0]
        
        # Get prompt if available
        gt_prompt = sample.get('prompt', sample.get('gt_prompt', ''))
        
        # Convert image to PIL for processor
        image = Image.fromarray(tp_img)
        
        # Format conversation for Stage 2
        from dflip_dataset.formatting import format_stage2_conversation
        
        messages = format_stage2_conversation(
            is_fake=(label == 1),
            generator=generator_name,
            gt_prompt=gt_prompt,
            include_assistant=True
        )
        
        return {
            'image': image,
            'messages': messages,
            'uid': sample.get('key', f'sample_{index}')
        }
    
    def collate_fn_profiling(self, batch: List[Dict]) -> Dict:
        """Custom collate function for profiling mode."""
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        generator_ids = torch.tensor([item['generator_id'] for item in batch], dtype=torch.long)
        shapes = torch.stack([item['shape'] for item in batch])
        
        result = {
            'pixel_values': images,
            'masks': masks,
            'is_fake': labels,
            'generator_ids': generator_ids,
            'shapes': shapes,
            'uids': [item['name'] for item in batch]
        }
        
        # Add edge masks if present
        if 'edge_mask' in batch[0]:
            edge_masks = torch.stack([item['edge_mask'] for item in batch])
            result['edge_masks'] = edge_masks
        
        return result
    
    def collate_fn_interpreting(self, batch: List[Dict]) -> Dict:
        """Custom collate function for interpreting mode (Stage 2)."""
        images = [item['image'] for item in batch]
        messages_batch = [item['messages'] for item in batch]
        
        # Process with Qwen VL processor
        texts = []
        for messages in messages_batch:
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


def create_opensdi_dataloader(
    dataset_path: str = "nebula/OpenSDI_train",
    split_name: str = "train",
    task_mode: Literal['profiling', 'interpreting'] = 'profiling',
    processor: Optional[AutoProcessor] = None,
    batch_size: int = 8,
    pixel_only: bool = False,
    output_size: tuple = (1024, 1024),
    common_transforms=None,
    post_transform=None,
    edge_width: Optional[int] = None,
    num_workers: int = 4,
    config: Optional[Dict] = None,
    **kwargs
):
    """
    Convenience function to create DataLoader for OpenSDI dataset.
    
    Args:
        dataset_path: HuggingFace dataset path
        split_name: Dataset split name
        task_mode: 'profiling' or 'interpreting'
        processor: Qwen VL processor
        batch_size: Batch size
        pixel_only: Filter for pixel-level manipulations only
        output_size: Target output size (H, W)
        common_transforms: Albumentations transforms
        post_transform: Post-processing transforms
        edge_width: Width for edge mask generation
        num_workers: Number of data loading workers
        config: Configuration dictionary
        **kwargs: Additional DataLoader arguments
    
    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset = OpenSDIDataset(
        dataset_path=dataset_path,
        split_name=split_name,
        task_mode=task_mode,
        processor=processor,
        pixel_only=pixel_only,
        output_size=output_size,
        common_transforms=common_transforms,
        post_transform=post_transform,
        edge_width=edge_width,
        config=config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split_name == 'train'),
        collate_fn=dataset.get_collate_fn(),
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    return dataloader
