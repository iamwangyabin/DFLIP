#!/usr/bin/env python3
"""
从test数据集中随机复制50张fake图片的简单脚本
"""
import json
import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict


def load_metadata(metadata_path: str) -> List[Dict]:
    """加载metadata文件"""
    print(f"正在读取metadata文件: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"总共加载了 {len(data)} 条记录")
    return data


def filter_test_fake_images(data: List[Dict]) -> List[Dict]:
    """筛选出test split中的fake图片"""
    test_fake_images = []
    
    for record in data:
        if record.get('split') == 'test' and record.get('is_fake') == 1:
            test_fake_images.append(record)
    
    print(f"找到 {len(test_fake_images)} 张test split中的fake图片")
    return test_fake_images


def copy_random_images(fake_images: List[Dict], image_root: str, output_dir: str, count: int = 50):
    """随机选择并复制指定数量的fake图片"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path.absolute()}")
    
    # 随机选择图片
    if len(fake_images) < count:
        print(f"警告: 只有 {len(fake_images)} 张fake图片，少于请求的 {count} ��")
        selected_images = fake_images
    else:
        selected_images = random.sample(fake_images, count)
    
    print(f"准备复制 {len(selected_images)} 张图片...")
    
    # 复制图片
    copied_count = 0
    failed_count = 0
    
    for i, record in enumerate(selected_images, 1):
        image_path = record['image_path']
        source_path = Path(image_root) / image_path
        
        # 生成目标文件名
        file_extension = source_path.suffix
        target_filename = f"fake_{i:03d}{file_extension}"
        target_path = output_path / target_filename
        
        try:
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                copied_count += 1
                print(f"[{i:2d}/{len(selected_images)}] 复制成功: {image_path} -> {target_filename}")
            else:
                print(f"[{i:2d}/{len(selected_images)}] 文件不存在: {source_path}")
                failed_count += 1
        except Exception as e:
            print(f"[{i:2d}/{len(selected_images)}] 复制失败: {image_path} - 错误: {e}")
            failed_count += 1
    
    print(f"\n复制完成!")
    print(f"成功复制: {copied_count} 张")
    print(f"失败: {failed_count} 张")
    print(f"输出目录: {output_path.absolute()}")


def find_image_root():
    """尝试找到图片根目录"""
    possible_paths = [
        "/media/dataset/person_dataset/DFLIP3K_processed_traintest",
        "/media/DATASET/person_data/DFLIP3K_processed_traintest",
    ]
    
    for path in possible_paths:
        expanded_path = Path(path).expanduser()
        if expanded_path.exists():
            return str(expanded_path)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='从test数据集中随机复制fake图片')
    parser.add_argument(
        '--metadata', 
        type=str, 
        default='./assets/dflip3k_meta_processed.json',
        help='metadata文件路径'
    )
    parser.add_argument(
        '--image-root', 
        type=str,
        help='图片根目录路径'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./copied_fake_images',
        help='输出目录'
    )
    parser.add_argument(
        '--count', 
        type=int, 
        default=50,
        help='要复制的图片数量'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 检查metadata文件
    if not Path(args.metadata).exists():
        print(f"错误: metadata文件不存在: {args.metadata}")
        return 1
    
    # 确定图片根目录
    if args.image_root:
        image_root = args.image_root
    else:
        print("正在尝试自动查找图片根目录...")
        image_root = find_image_root()
        if not image_root:
            print("错误: 无法找到图片根目录，请使用 --image-root 参数指定")
            return 1
    
    if not Path(image_root).exists():
        print(f"错误: 图片根目录不存在: {image_root}")
        return 1
    
    print(f"使用图片根目录: {image_root}")
    
    try:
        # 加载数据
        data = load_metadata(args.metadata)
        
        # 筛选test fake图片
        fake_images = filter_test_fake_images(data)
        
        if not fake_images:
            print("错误: 没有找到test split中的fake图片")
            return 1
        
        # 复制图片
        copy_random_images(fake_images, image_root, args.output_dir, args.count)
        
        return 0
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
