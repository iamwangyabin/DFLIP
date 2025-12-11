#!/usr/bin/env python3
import argparse
import json
import os
import random
import io
import time
import psutil
from pathlib import Path
from typing import Dict, List, Generator

from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Image, Value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack DFLIP dataset to HF Datasets format (Memory Efficient Version)"
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--image-extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Save images in original format without resizing to JPEG (faster processing)",
    )
    return parser.parse_args()


def resize_to_jpeg(img_path: Path) -> bytes:
    """
    将图片resize为最短边512，并转换为JPEG格式（使用随机压缩质量和插值算法）。
    返回字节数据。
    """
    # 随机选择插值算法
    interpolation_methods = [
        PILImage.NEAREST,    # 最近邻
        PILImage.BILINEAR,   # 双线性
        PILImage.BICUBIC,    # 双三次
        PILImage.LANCZOS,    # Lanczos
    ]
    interp = random.choice(interpolation_methods)
    
    # 随机选择JPEG压缩质量 (75-95)
    quality = random.randint(75, 95)
    
    # 打开图片
    with PILImage.open(img_path) as img:
        # 转换为RGB（JPEG不支持透明通道）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 计算新尺寸（最短边为512）
        width, height = img.size
        if width < height:
            new_width = 512
            new_height = int(height * (512 / width))
        else:
            new_height = 512
            new_width = int(width * (512 / height))
        
        # Resize图片
        img_resized = img.resize((new_width, new_height), interp)
        
        # 转换为JPEG字节
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG', quality=quality, optimize=True)
        return buffer.getvalue()


def read_original_image(img_path: Path) -> bytes:
    """
    直接读取原始图片文件，不做任何处理。
    返回字节数据。
    """
    with open(img_path, 'rb') as f:
        return f.read()


def image_generator(data_root: Path, image_extensions: set, no_resize: bool = False) -> Generator[Dict, None, None]:
    """
    使用生成器(yield)逐个返回样本，避免一次性把所有文件加载到内存列表。
    处理图片：
    - 如果no_resize=True：直接保存原始图片
    - 如果no_resize=False：resize最短边为512，转换为JPEG格式（随机压缩质量和插值算法）
    """
    data_root = Path(data_root)
    fake_dir = data_root / "fake"
    real_dir = data_root / "real"

    if not fake_dir.exists() or not real_dir.exists():
        raise NotADirectoryError("Fake or Real directory not found")

    # 遍历 Fake (流式)
    print(f"Scanning {fake_dir}...")
    # rglob 返回的是迭代器，不要用 list() 包裹它
    total_bytes = 0
    count = 0
    start_time = time.time()
    last_log_time = start_time
    process = psutil.Process(os.getpid())
    for img_path in fake_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            rel_path = img_path.relative_to(data_root)
            try:
                # 根据no_resize标志选择处理方式
                if no_resize:
                    img_bytes = read_original_image(img_path)
                else:
                    img_bytes = resize_to_jpeg(img_path)
                
                # DEBUG: Track cumulative size and performance
                total_bytes += len(img_bytes)
                count += 1
                if count % 1000 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    interval = current_time - last_log_time
                    mem_info = process.memory_info()
                    
                    print(f"[DEBUG] Processed {count} images:")
                    print(f"  - Cumulative size: {total_bytes / (1024**3):.2f} GB, avg: {total_bytes / count / (1024**2):.2f} MB")
                    print(f"  - Speed: {1000/interval:.1f} img/s (last 1000), {count/elapsed:.1f} img/s (overall)")
                    print(f"  - Memory: RSS={mem_info.rss / (1024**3):.2f} GB, VMS={mem_info.vms / (1024**3):.2f} GB")
                    last_log_time = current_time
                
                yield {
                    "image": {"bytes": img_bytes},  # 直接提供字节数据
                    "image_path": str(rel_path),
                    "is_fake": 1,
                }
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    # 遍历 Real (流式)
    print(f"Scanning {real_dir}...")
    print(f"[DEBUG] Fake images total: {count}, total size: {total_bytes / (1024**3):.2f} GB")
    for img_path in real_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            rel_path = img_path.relative_to(data_root)
            try:
                # 根据no_resize标志选择处理方式
                if no_resize:
                    img_bytes = read_original_image(img_path)
                else:
                    img_bytes = resize_to_jpeg(img_path)
                
                # DEBUG: Track cumulative size and performance
                total_bytes += len(img_bytes)
                count += 1
                if count % 1000 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    interval = current_time - last_log_time
                    mem_info = process.memory_info()
                    
                    print(f"[DEBUG] Processed {count} images:")
                    print(f"  - Cumulative size: {total_bytes / (1024**3):.2f} GB, avg: {total_bytes / count / (1024**2):.2f} MB")
                    print(f"  - Speed: {1000/interval:.1f} img/s (last 1000), {count/elapsed:.1f} img/s (overall)")
                    print(f"  - Memory: RSS={mem_info.rss / (1024**3):.2f} GB, VMS={mem_info.vms / (1024**3):.2f} GB")
                    last_log_time = current_time
                
                yield {
                    "image": {"bytes": img_bytes},  # 直接提供字节数据
                    "image_path": str(rel_path),
                    "is_fake": 0,
                }
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output).resolve()
    extensions = set(args.image_extensions)

    processing_mode = "original format (no resize)" if args.no_resize else "resize to JPEG (512px shortest side)"
    print(f"Packing dataset from {data_root} to {output_path}")
    print(f"Processing mode: {processing_mode}")

    # 1. 定义 Features
    features = Features({
        "image": Image(),
        "image_path": Value("string"),
        "is_fake": Value("int8"),
    })

    # 2. 使用 from_generator 构建 Dataset
    # 这样 HuggingFace 会流式地读取图片并写入磁盘，不会撑爆内存
    # Calculate optimal writer batch size based on expected image size
    # Target: keep batch size under 1.5GB to safely avoid 2GB PyArrow limit
    # For original images (~2.74MB avg), use smaller batch size
    if args.no_resize:
        # Conservative batch size for large original images
        # Assuming 3MB avg per image: 1.5GB / 3MB = ~500 images
        writer_batch_size = 500
        print(f"[DEBUG] Using writer_batch_size={writer_batch_size} for original format images")
    else:
        # Resized images are smaller (~100-200KB), can use larger batches
        writer_batch_size = 1000
        print(f"[DEBUG] Using writer_batch_size={writer_batch_size} for resized images")
    
    hf_dataset = Dataset.from_generator(
        generator=image_generator,
        gen_kwargs={
            "data_root": data_root,
            "image_extensions": extensions,
            "no_resize": args.no_resize,
        },
        features=features,
        writer_batch_size=writer_batch_size,
    )

    hf_ds = DatasetDict({"all": hf_dataset})

    # 3. 保存 Dataset 到磁盘
    # num_proc 可以开启多进程加速图片编码，但要注意内存占用会随进程数增加
    print("Saving dataset to disk (this involves reading and compressing images)...")
    save_start = time.time()
    
    # [DIAGNOSTIC] Monitor memory during save
    parent_process = psutil.Process(os.getpid())
    print(f"[DEBUG] Pre-save memory: RSS={parent_process.memory_info().rss / (1024**3):.2f} GB")
    
    hf_ds.save_to_disk(str(output_path), num_proc=4)
    
    save_elapsed = time.time() - save_start
    print(f"[DEBUG] Save completed in {save_elapsed:.1f}s")
    print(f"[DEBUG] Post-save memory: RSS={parent_process.memory_info().rss / (1024**3):.2f} GB")

    # 4. 创建索引 (Path Index)
    # 由于我们之前是用 generator 流式写入的，现在需要重新遍历一下生成的 dataset 来建立索引
    # 虽然多了一步遍历，但这是内存安全的做法。
    # 如果不想重新遍历，可以在 generator 里写个副作用记录，但因为多进程处理，那样做很复杂。
    print("Creating path index from saved dataset...")

    # 重新加载（Lazy loading，不占内存）
    saved_ds = hf_ds["all"]

    # 我们只需要 'image_path' 列，不需要加载图片数据，所以速度很快
    print("[DEBUG] Creating path index...")
    index_start = time.time()
    
    # [DIAGNOSTIC] Monitor memory during index creation
    mem_before = parent_process.memory_info().rss / (1024**3)
    path_list = saved_ds["image_path"]
    mem_after_load = parent_process.memory_info().rss / (1024**3)
    print(f"[DEBUG] Loaded {len(path_list)} paths, memory increased by {mem_after_load - mem_before:.2f} GB")
    
    path_to_idx = {path: idx for idx, path in enumerate(path_list)}

    index_file = output_path / "path_index.json"
    print(f"Saving path index ({len(path_to_idx)} items) to {index_file}...")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(path_to_idx, f, ensure_ascii=False, indent=2)
    
    index_elapsed = time.time() - index_start
    print(f"[DEBUG] Index creation completed in {index_elapsed:.1f}s")

    print("Done!")


if __name__ == "__main__":
    main()