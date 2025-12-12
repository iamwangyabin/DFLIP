#!/usr/bin/env python3
"""分析 dflip3k_meta.json 中的统计信息。

该脚本读取由 generate_dataset_meta_dflip3k.py 生成的 metadata JSON 文件，
并输出以下统计信息：
- 所有生成模型（家族和子模型）
- 每个模型下的图片数量
- 真实图片和假图片的总数

使用方法:
    python tools/analyze_metadata_stats.py --meta-file ./assets/dflip3k_meta.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 DFLIP3K metadata 统计信息"
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        default="./assets/dflip3k_meta.json",
        help="Metadata JSON 文件路径",
    )
    return parser.parse_args()


def analyze_metadata(meta_file: Path) -> None:
    """分析 metadata 并打印统计信息。"""
    
    if not meta_file.exists():
        print(f"错误: 文件不存在: {meta_file}")
        return
    
    print(f"正在读取文件: {meta_file}")
    with meta_file.open("r", encoding="utf-8") as f:
        records = json.load(f)
    
    print(f"总记录数: {len(records)}\n")
    
    # 统计真实和假图片
    real_count = sum(1 for r in records if r["is_fake"] == 0)
    fake_count = sum(1 for r in records if r["is_fake"] == 1)
    
    print("=" * 60)
    print("整体统计:")
    print("=" * 60)
    print(f"真实图片: {real_count}")
    print(f"假图片: {fake_count}")
    print(f"总计: {len(records)}\n")
    
    # 按 family 和 version 统计
    # 首先需要从路径中提取 family 和 submodel 信息
    family_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    family_total: Dict[str, int] = defaultdict(int)
    
    for record in records:
        if record["is_fake"] == 1:
            # 从路径中提取 family 和 submodel
            # 路径格式: fake/<family>/<submodel>/<filename>
            path = record["image_path"]
            parts = path.split("/")
            
            if len(parts) >= 3 and parts[0] == "fake":
                family = parts[1]
                submodel = parts[2]
                
                family_stats[family][submodel] += 1
                family_total[family] += 1
    
    # 打印按 family 分组的统计
    print("=" * 60)
    print("按生成模型家族统计:")
    print("=" * 60)
    
    sorted_families = sorted(family_stats.keys())
    for i, family in enumerate(sorted_families, 1):
        print(f"\n{i}. {family} (总计: {family_total[family]} 张图片)")
        print("-" * 60)
        
        submodels = family_stats[family]
        sorted_submodels = sorted(submodels.items(), key=lambda x: x[1], reverse=True)
        
        for j, (submodel, count) in enumerate(sorted_submodels, 1):
            percentage = (count / family_total[family]) * 100
            print(f"   {j}. {submodel}: {count} 张 ({percentage:.1f}%)")
    
    # 打印所有子模型的总结表
    print("\n" + "=" * 60)
    print("所有子模型汇总表 (按图片数量排序):")
    print("=" * 60)
    
    all_submodels: List[tuple] = []
    for family, submodels in family_stats.items():
        for submodel, count in submodels.items():
            all_submodels.append((family, submodel, count))
    
    all_submodels.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n{'序号':<6} {'家族':<20} {'子模型':<25} {'图片数':<10}")
    print("-" * 65)
    for i, (family, submodel, count) in enumerate(all_submodels, 1):
        print(f"{i:<6} {family:<20} {submodel:<25} {count:<10}")
    
    # 按数据集划分统计
    print("\n" + "=" * 60)
    print("按数据集划分统计:")
    print("=" * 60)
    
    split_stats = defaultdict(lambda: {"real": 0, "fake": 0, "total": 0})
    for record in records:
        split = record.get("split", "unknown")
        if record["is_fake"] == 0:
            split_stats[split]["real"] += 1
        else:
            split_stats[split]["fake"] += 1
        split_stats[split]["total"] += 1
    
    for split in ["train", "test"]:
        if split in split_stats:
            stats = split_stats[split]
            print(f"\n{split.upper()}:")
            print(f"  真实图片: {stats['real']}")
            print(f"  假图片: {stats['fake']}")
            print(f"  总计: {stats['total']}")
    
    print("\n" + "=" * 60)
    print(f"总生成模型家族数: {len(family_stats)}")
    print(f"总子模型数: {len(all_submodels)}")
    print("=" * 60)


def main():
    args = parse_args()
    meta_file = Path(args.meta_file)
    analyze_metadata(meta_file)


if __name__ == "__main__":
    main()