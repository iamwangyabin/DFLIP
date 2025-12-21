#!/usr/bin/env python3
"""Create frequency domain spectrum figures for different generative models.

This script analyzes the frequency domain characteristics of images from different
generative models in the DFLIP3K dataset and creates publication-ready spectrum
visualizations arranged in a grid layout.

Usage:
    python tools/create_frequency_spectrum_figure.py \
        --metadata ./tools/dflip3k_meta_processed.json \
        --image-root /path/to/DFLIP3K \
        --output frequency_spectrum_figure.png \
        --grid-size "3x9" \
        --samples-per-model 200 \
        --seed 42
"""

import argparse
import json
import random
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# For image processing and figure generation
try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    import matplotlib.font_manager as fm
    from scipy import fft
    import cv2

    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False
    print("Error: Required packages missing. Install with: pip install Pillow matplotlib scipy opencv-python")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create frequency domain spectrum figures for different generative models"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to the metadata JSON file (e.g., ./tools/dflip3k_meta_processed.json)",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory containing the images (e.g., /path/to/DFLIP3K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="frequency_spectrum_figure.png",
        help="Output PNG file path",
    )
    parser.add_argument(
        "--grid-size",
        type=str,
        default="3x9",
        help="Grid layout (rows x cols), e.g., '3x9' for 3 rows, 9 columns",
    )
    parser.add_argument(
        "--samples-per-model",
        type=int,
        default=200,
        help="Number of images to sample per model for spectrum analysis",
    )
    parser.add_argument(
        "--target-models",
        type=int,
        default=27,
        help="Number of models to include in the figure",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Which data split to sample from",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output image",
    )
    parser.add_argument(
        "--spectrum-size",
        type=int,
        default=256,
        help="Size of the frequency spectrum visualization",
    )
    return parser.parse_args()


def load_metadata(metadata_path: str) -> List[Dict]:
    """Load metadata from JSON file."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_models(records: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    """Analyze available models and their statistics."""
    family_stats = defaultdict(lambda: {
        'name': None,
        'count': 0,
        'versions': set()
    })

    version_stats = defaultdict(lambda: {
        'family_id': None,
        'family_name': None,
        'version_name': None,
        'count': 0,
        'sample_paths': []
    })

    family_to_versions = defaultdict(set)

    # Only process fake images (real images have family_id=None)
    fake_records = [r for r in records if r['is_fake'] == 1]
    print(f"Debug: Found {len(fake_records)} fake records")

    processed_count = 0
    for record in fake_records:
        family_id = record['family_id']
        version_id = record['version_id']
        image_path = record['image_path']

        # Skip records with None IDs
        if family_id is None or version_id is None:
            continue

        # Extract family and version names from path
        # Expected format: {split}/fake/family_name/version_name/image.jpg
        path_parts = Path(image_path).parts
        if len(path_parts) >= 4 and path_parts[1] == 'fake':
            family_name = path_parts[2]
            version_name = path_parts[3]

            # Update family stats
            family_stats[family_id]['name'] = family_name
            family_stats[family_id]['count'] += 1
            family_stats[family_id]['versions'].add(version_id)

            # Update version stats
            version_stats[version_id]['family_id'] = family_id
            version_stats[version_id]['family_name'] = family_name
            version_stats[version_id]['version_name'] = version_name
            version_stats[version_id]['count'] += 1
            version_stats[version_id]['sample_paths'].append(image_path)

            # Update family to versions mapping
            family_to_versions[family_id].add(version_id)

            processed_count += 1

    print(f"Debug: Processed {processed_count} records with valid family/version IDs")

    # Convert sets to lists
    for fid in family_stats:
        family_stats[fid]['versions'] = list(family_stats[fid]['versions'])

    for fid in family_to_versions:
        family_to_versions[fid] = list(family_to_versions[fid])

    return dict(family_stats), dict(version_stats), dict(family_to_versions)


def select_representative_models(
        family_stats: Dict,
        version_stats: Dict,
        family_to_versions: Dict,
        target_models: int = 27,
        seed: int = 42
) -> List[int]:
    """Select representative models ensuring diversity across families."""
    random.seed(seed)

    # Sort families by number of versions (descending)
    families_by_importance = sorted(
        family_stats.items(),
        key=lambda x: len(x[1]['versions']),
        reverse=True
    )

    selected_versions = []
    versions_per_family = {}

    # First pass: select at least one version from each family
    for family_id, family_info in families_by_importance:
        if len(selected_versions) >= target_models:
            break

        # Sort versions in this family by image count (descending)
        family_versions = family_to_versions[family_id]
        family_versions_sorted = sorted(
            family_versions,
            key=lambda vid: version_stats[vid]['count'],
            reverse=True
        )

        # Select the version with most images from this family
        if family_versions_sorted:
            selected_version = family_versions_sorted[0]
            selected_versions.append(selected_version)
            versions_per_family[family_id] = [selected_version]

    # Second pass: add more versions from families with many versions
    remaining_slots = target_models - len(selected_versions)

    for _ in range(remaining_slots):
        best_family = None
        best_version = None
        best_score = -1

        for family_id, family_info in families_by_importance:
            already_selected = versions_per_family.get(family_id, [])
            available_versions = [
                vid for vid in family_to_versions[family_id]
                if vid not in already_selected
            ]

            if not available_versions:
                continue

            # Score = (total family versions) / (already selected from family + 1)
            score = len(family_info['versions']) / (len(already_selected) + 1)

            if score > best_score:
                # Select version with most images from available ones
                best_version_candidate = max(
                    available_versions,
                    key=lambda vid: version_stats[vid]['count']
                )
                best_family = family_id
                best_version = best_version_candidate
                best_score = score

        if best_version is not None:
            selected_versions.append(best_version)
            if best_family not in versions_per_family:
                versions_per_family[best_family] = []
            versions_per_family[best_family].append(best_version)

    return selected_versions[:target_models]


def sample_images_for_spectrum_analysis(
        records: List[Dict],
        selected_versions: List[int],
        samples_per_model: int = 200,
        split_filter: str = "train",
        seed: int = 42
) -> Dict[int, List[str]]:
    """Sample multiple images from each selected model for spectrum analysis."""
    random.seed(seed)

    # Group records by version_id
    version_to_records = defaultdict(list)
    for record in records:
        if record['is_fake'] == 1:  # Only fake images
            if split_filter == "all" or record['split'] == split_filter:
                version_to_records[record['version_id']].append(record)

    sampled_images = {}

    for version_id in selected_versions:
        available_records = version_to_records.get(version_id, [])

        if not available_records:
            print(f"Warning: No images found for version_id {version_id}")
            continue

        # Sample up to samples_per_model images
        sample_count = min(samples_per_model, len(available_records))
        selected_records = random.sample(available_records, sample_count)
        sampled_images[version_id] = [record['image_path'] for record in selected_records]

        print(f"Sampled {len(sampled_images[version_id])} images for version_id {version_id}")

    return sampled_images


def compute_frequency_spectrum(image_path: str, spectrum_size: int = 256) -> Optional[np.ndarray]:
    """Compute the 2D frequency spectrum of an image."""
    try:
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Resize to standard size for consistent analysis
        img = cv2.resize(img, (spectrum_size, spectrum_size))
        
        # Apply 2D FFT
        f_transform = fft.fft2(img)
        f_shift = fft.fftshift(f_transform)
        
        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(f_shift)
        
        # Apply log transform for better visualization
        magnitude_spectrum = np.log(magnitude_spectrum + 1)
        
        return magnitude_spectrum
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def compute_average_spectrum(
        image_paths: List[str], 
        image_root: str, 
        spectrum_size: int = 256
) -> Optional[np.ndarray]:
    """Compute the average frequency spectrum from multiple images."""
    spectrums = []
    image_root_path = Path(image_root)
    
    for image_path in image_paths:
        full_path = image_root_path / image_path
        spectrum = compute_frequency_spectrum(str(full_path), spectrum_size)
        if spectrum is not None:
            spectrums.append(spectrum)
    
    if not spectrums:
        return None
    
    # Compute average spectrum
    avg_spectrum = np.mean(spectrums, axis=0)
    return avg_spectrum


def get_model_name_mapping() -> Dict[str, str]:
    """Create mapping from original family names to display names."""
    return {
        # Actual model names found in the dataset
        'sd_1.5': 'Stable Diffusion 1.5',
        'sdxl_1.0': 'Stable Diffusion XL',
        'pony': 'Pony Diffusion',
        'illustrious': 'Illustrious',
        'sd_2.1': 'Stable Diffusion 2.1',
        'flux.1_d': 'FLUX.1 [dev]',
        'qwen': 'Qwen-Image',
        'noobai': 'NoobAI',
        'flux.1_s': 'FLUX.1 [schnell]',
        'sd_3.5_large': 'Stable Diffusion 3.5 Large',
        'gpt-image-1': 'GPT-Image 1',
        'pixart': 'PixArt',
        'imagen4': 'Imagen 4',
        'kolors': 'Kolors',
        'sd_3.5_medium': 'Stable Diffusion 3.5 Medium',
        'playground_v2': 'Playground v2',
        'seedream': 'Seedream',
        'nano_banana_pro': 'Nano-Banana Pro',
        'z-image': 'Z-Image',
        'hidream': 'HiDream-I1',
        'chroma': 'Chroma',
        'nano_banana': 'Nano-Banana',
        'flux2-dev': 'FLUX.2 [dev]',
        'pony_v7': 'Pony Diffusion v7',
        'stable_cascade': 'Stable Cascade',
        'aura_flow': 'AuraFlow',
        'hunyuan': 'Hunyuan-DiT',

        # Additional alternative naming patterns for compatibility
        'auraflow': 'AuraFlow',
        'flux_1_dev': 'FLUX.1 [dev]',
        'flux_1_schnell': 'FLUX.1 [schnell]',
        'flux_2_dev': 'FLUX.2 [dev]',
        'gpt_image_1': 'GPT-Image 1',
        'hidream_i1': 'HiDream-I1',
        'hunyuan_dit': 'Hunyuan-DiT',
        'imagen_4': 'Imagen 4',
        'pony_diffusion_v6': 'Pony Diffusion v6',
        'pony_diffusion_v7': 'Pony Diffusion v7',
        'qwen_image': 'Qwen-Image',
        'stable_diffusion_1': 'Stable Diffusion 1',
        'stable_diffusion_2': 'Stable Diffusion 2',
        'stable_diffusion_3_5_large': 'Stable Diffusion 3.5 Large',
        'stable_diffusion_3_5_medium': 'Stable Diffusion 3.5 Medium',
        'stable_diffusion_xl': 'Stable Diffusion XL',
        'z_image_turbo': 'Z-Image Turbo',
        'z_image': 'Z-Image',
    }


def create_frequency_spectrum_figure(
        sampled_images: Dict[int, List[str]],
        version_stats: Dict,
        image_root: str,
        output_path: str,
        grid_size: str = "3x9",
        spectrum_size: int = 256,
        dpi: int = 300
) -> str:
    """Create a frequency spectrum figure with multiple models."""
    
    # Parse grid size
    rows, cols = map(int, grid_size.split('x'))
    total_slots = rows * cols

    # Prepare model data
    model_data = []
    for version_id, image_paths in sampled_images.items():
        version_info = version_stats[version_id]
        model_data.append({
            'version_id': version_id,
            'family_name': version_info['family_name'],
            'version_name': version_info['version_name'],
            'image_paths': image_paths
        })

    # Sort by family name for consistent ordering
    model_data.sort(key=lambda x: (x['family_name'], x['version_name']))

    # Limit to available slots
    model_data = model_data[:total_slots]

    # Layout parameters
    TITLE_FONT_SIZE = 16
    TITLE_PAD = 10

    fig_width = cols * 3.0
    fig_height = rows * 3.5

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Adjust GridSpec spacing
    gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.1,
                  left=0.02, right=0.98, top=0.95, bottom=0.05)

    # Set font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Times']
    times_font = fm.FontProperties(family='serif')

    print(f"Computing frequency spectrums for {len(model_data)} models...")

    for i, model_info in enumerate(model_data):
        row = i // cols
        col = i % cols

        ax = fig.add_subplot(gs[row, col])

        print(f"Processing model {i+1}/{len(model_data)}: {model_info['family_name']}")

        # Compute average frequency spectrum
        avg_spectrum = compute_average_spectrum(
            model_info['image_paths'], 
            image_root, 
            spectrum_size
        )

        if avg_spectrum is not None:
            # Display spectrum
            im = ax.imshow(avg_spectrum, cmap='hot', origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar for reference (optional, can be removed for cleaner look)
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Show placeholder if spectrum computation failed
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        # Create clean title
        original_family_name = model_info['family_name'].lower()
        name_mapping = get_model_name_mapping()

        if original_family_name in name_mapping:
            display_name = name_mapping[original_family_name]
        else:
            display_name = model_info['family_name'].replace('_', ' ').title()

        ax.set_title(display_name, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD, 
                     weight='bold', fontproperties=times_font, wrap=False,
                     horizontalalignment='center')

    # Fill empty slots
    for i in range(len(model_data), total_slots):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    return output_path


def main():
    args = parse_args()

    print("Loading metadata...")
    records = load_metadata(args.metadata)

    print("\nAnalyzing models...")
    family_stats, version_stats, family_to_versions = analyze_models(records)

    print(f"\nSelecting {args.target_models} representative models...")
    selected_versions = select_representative_models(
        family_stats, version_stats, family_to_versions,
        target_models=args.target_models, seed=args.seed
    )

    # Print selected models with display names
    print("\nSelected models:")
    name_mapping = get_model_name_mapping()
    for i, vid in enumerate(selected_versions, 1):
        version_info = version_stats[vid]
        original_name = version_info['family_name']
        display_name = name_mapping.get(original_name.lower(), original_name.replace('_', ' ').title())
        print(f"{i:2d}. {original_name} -> {display_name}")

    print(f"\nSampling {args.samples_per_model} images per model from '{args.split}' split...")
    sampled_images = sample_images_for_spectrum_analysis(
        records, selected_versions, samples_per_model=args.samples_per_model,
        split_filter=args.split, seed=args.seed
    )

    print(f"\nCreating frequency spectrum figure ({args.grid_size})...")
    output_path = create_frequency_spectrum_figure(
        sampled_images, version_stats, args.image_root, args.output,
        grid_size=args.grid_size, spectrum_size=args.spectrum_size, dpi=args.dpi
    )

    print(f"\nFrequency spectrum figure created successfully: {output_path}")


if __name__ == "__main__":
    main()