#!/usr/bin/env python3
"""将DFLIP3K fake目录中的所有JSON文件整合成CSV和/或大JSON文件。

用法:
    # 生成CSV和JSON
    python tools/consolidate_json_files.py --fake-root /home/data/yabin/DFLIP3K/fake --output output
    
    # 只生成CSV
    python tools/consolidate_json_files.py --fake-root /home/data/yabin/DFLIP3K/fake --output output.csv --format csv
    
    # 只生成JSON
    python tools/consolidate_json_files.py --fake-root /home/data/yabin/DFLIP3K/fake --output output.json --format json
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="整合所有JSON文件到CSV或大JSON文件"
    )
    parser.add_argument(
        "--fake-root",
        type=str,
        required=True,
        help="Fake图片的根目录，例如: /home/data/yabin/DFLIP3K/fake",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径（不带扩展名会自动生成CSV和JSON，带扩展名只生成该格式）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "both"],
        default="both",
        help="输出格式: csv, json, 或 both (默认: both)",
    )
    parser.add_argument(
        "--max-sample",
        type=int,
        default=None,
        help="最大采样数量（用于测试，默认处理所有文件）",
    )
    return parser.parse_args()


def iter_json_files(root: Path, max_sample: int = None):
    """生成器：边遍历边yield JSON文件，避免重复IO"""
    if not root.exists():
        print(f"[错误] 目录不存在: {root}")
        return
    
    print("📂 开始流式处理JSON文件...")
    
    # 如果有max_sample限制，需要先收集再采样
    if max_sample:
        all_files = list(root.rglob("*.json"))
        print(f"[采样] 从{len(all_files)}个文件中采样{max_sample}个")
        import random
        random.seed(42)
        sampled = random.sample(all_files, min(max_sample, len(all_files)))
        for p in sampled:
            if p.is_file():
                yield p
    else:
        # 流式处理，边遍历边yield
        for p in root.rglob("*.json"):
            if p.is_file():
                yield p


def load_json_file(json_path: Path) -> Dict[str, Any]:
    """加载单个JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[警告] 无法读取 {json_path}: {e}")
        return {}


def analyze_json_structure(root: Path, sample_size: int = 100) -> Set[str]:
    """快速采样分析JSON结构，获取所有可能的字段"""
    all_keys = set()
    count = 0
    
    print(f"\n🔍 快速采样分析JSON结构 (最多{sample_size}个文件)...")
    
    # 使用生成器，采样指定数量的文件
    pbar = tqdm(desc="🔍 采样分析", unit="个", ncols=100, colour='blue')
    for json_path in root.rglob("*.json"):
        if count >= sample_size:
            break
        if json_path.is_file():
            data = load_json_file(json_path)
            if data:
                all_keys.update(flatten_dict(data).keys())
            count += 1
            pbar.update(1)
    pbar.close()
    
    print(f"   从{count}个文件中发现字段")
    return all_keys


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """扁平化嵌套字典"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # 将列表转换为字符串
            items.append((new_key, json.dumps(v, ensure_ascii=False)))
        else:
            items.append((new_key, v))
    return dict(items)


def consolidate_to_json(fake_root: Path, output_path: Path, max_sample: int = None):
    """流式整合所有JSON到一个大JSON文件"""
    print(f"\n📦 流式整合JSON文件到大JSON...")
    
    consolidated = []
    count = 0
    
    # 使用生成器流式处理
    pbar = tqdm(desc="📦 整合JSON", unit="个", ncols=100, colour='cyan')
    for json_path in iter_json_files(fake_root, max_sample):
        data = load_json_file(json_path)
        if data:
            # 添加元数据
            try:
                rel_path = json_path.relative_to(fake_root)
                data['_meta_json_path'] = str(rel_path)
                data['_meta_filename'] = json_path.name
                
                # 提取family和submodel
                if len(rel_path.parts) >= 2:
                    data['_meta_family'] = rel_path.parts[0]
                    data['_meta_submodel'] = rel_path.parts[1]
            except ValueError:
                data['_meta_json_path'] = str(json_path)
            
            consolidated.append(data)
            count += 1
            pbar.update(1)
    pbar.close()
    
    # 保存为JSON
    print(f"💾 保存到 {output_path}...")
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 成功保存 {count} 条记录到 {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def consolidate_to_csv(fake_root: Path, output_path: Path, max_sample: int = None):
    """流式整合所有JSON到CSV文件"""
    print(f"\n📊 流式整合JSON文件到CSV...")
    
    # 第一步：快速采样分析字段结构
    all_keys = analyze_json_structure(fake_root, sample_size=100)
    
    # 添加元数据字段
    meta_fields = ['_meta_json_path', '_meta_filename', '_meta_family', '_meta_submodel']
    all_keys.update(meta_fields)
    
    # 按字母排序字段，但元数据字段放在前面
    sorted_keys = meta_fields + sorted([k for k in all_keys if k not in meta_fields])
    
    print(f"📋 发现 {len(sorted_keys)} 个字段")
    
    # 第二步：流式写入CSV
    print(f"💾 流式写入到 {output_path}...")
    count = 0
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys, extrasaction='ignore')
        writer.writeheader()
        
        # 使用生成器流式处理
        pbar = tqdm(desc="📊 写入CSV", unit="个", ncols=100, colour='magenta')
        for json_path in iter_json_files(fake_root, max_sample):
            data = load_json_file(json_path)
            if data:
                # 扁平化数据
                flat_data = flatten_dict(data)
                
                # 添加元数据
                try:
                    rel_path = json_path.relative_to(fake_root)
                    flat_data['_meta_json_path'] = str(rel_path)
                    flat_data['_meta_filename'] = json_path.name
                    
                    if len(rel_path.parts) >= 2:
                        flat_data['_meta_family'] = rel_path.parts[0]
                        flat_data['_meta_submodel'] = rel_path.parts[1]
                except ValueError:
                    flat_data['_meta_json_path'] = str(json_path)
                
                writer.writerow(flat_data)
                count += 1
                pbar.update(1)
        pbar.close()
    
    print(f"✅ 成功保存 {count} 条记录到 {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    args = parse_args()
    fake_root = Path(args.fake_root).resolve()
    
    if not fake_root.exists():
        print(f"[错误] fake_root不存在: {fake_root}")
        return
    
    print(f"📂 处理目录: {fake_root}\n")
    print("="*70)
    
    # 确定输出格式
    output_path = Path(args.output)
    
    if args.format == "both" or (args.format == "both" and not output_path.suffix):
        # 生成两种格式（流式处理，避免重复IO）
        base_path = output_path.with_suffix('')
        csv_path = base_path.with_suffix('.csv')
        json_path = base_path.with_suffix('.json')
        
        consolidate_to_csv(fake_root, csv_path, args.max_sample)
        consolidate_to_json(fake_root, json_path, args.max_sample)
        
        print("\n" + "="*70)
        print("🎉 整合完成！")
        print("="*70)
        print(f"📊 CSV文件:  {csv_path}")
        print(f"📦 JSON文件: {json_path}")
        
    elif args.format == "csv" or (output_path.suffix.lower() == '.csv'):
        # 只生成CSV（流式处理）
        csv_path = output_path.with_suffix('.csv')
        consolidate_to_csv(fake_root, csv_path, args.max_sample)
        
        print("\n" + "="*70)
        print("🎉 整合完成！")
        print("="*70)
        print(f"📊 CSV文件: {csv_path}")
        
    elif args.format == "json" or (output_path.suffix.lower() == '.json'):
        # 只生成JSON（流式处理）
        json_path = output_path.with_suffix('.json')
        consolidate_to_json(fake_root, json_path, args.max_sample)
        
        print("\n" + "="*70)
        print("🎉 整合完成！")
        print("="*70)
        print(f"📦 JSON文件: {json_path}")
    
    # 提供使用建议
    print("\n💡 使用建议:")
    print("  - CSV格式: 适合在Excel/Numbers中打开，方便数据分析和筛选")
    print("  - JSON格式: 保留完整的嵌套结构，适合程序化处理")
    print("  - CSV中的嵌套数据已被扁平化（用.分隔），列表会转为JSON字符串")


if __name__ == "__main__":
    main()