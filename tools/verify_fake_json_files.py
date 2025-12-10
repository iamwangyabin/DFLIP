#!/usr/bin/env python3
"""验证DFLIP3K fake目录中的图片是否都有对应的JSON文件。

用法:
    python tools/verify_fake_json_files.py --fake-root /home/data/yabin/DFLIP3K/fake
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证fake图片是否都有对应的JSON文件"
    )
    parser.add_argument(
        "--fake-root",
        type=str,
        required=True,
        help="Fake图片的根目录，例如: /home/data/yabin/DFLIP3K/fake",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    """判断是否为图片文件"""
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def collect_image_files(root: Path) -> List[Path]:
    """递归收集所有图片文件"""
    files: List[Path] = []
    if not root.exists():
        print(f"[错误] 目录不存在: {root}")
        return files
    
    print("正在扫描图片文件...")
    for p in tqdm(list(root.rglob("*")), desc="扫描文件", unit="文件"):
        if is_image_file(p):
            files.append(p)
    return files


def check_json_files(fake_root: Path) -> Tuple[int, int, List[Path]]:
    """检查图片的JSON文件
    
    返回:
        (有JSON的图片数, 没有JSON的图片数, 没有JSON的图片列表)
    """
    image_files = collect_image_files(fake_root)
    
    if not image_files:
        print(f"[警告] 在 {fake_root} 中没有找到图片文件")
        return 0, 0, []
    
    print(f"找到 {len(image_files)} 个fake图片")
    
    has_json = 0
    no_json = 0
    missing_json_files: List[Path] = []
    
    print("\n正在检查JSON文件...")
    for img_path in tqdm(image_files, desc="检查JSON", unit="图片"):
        # 构造对应的JSON文件路径（去掉图片扩展名，加上.json）
        json_path = img_path.with_suffix('.json')
        
        if json_path.exists():
            has_json += 1
        else:
            no_json += 1
            missing_json_files.append(img_path)
    
    return has_json, no_json, missing_json_files


def main():
    args = parse_args()
    fake_root = Path(args.fake_root).resolve()
    
    if not fake_root.exists():
        print(f"[错误] fake_root不存在: {fake_root}")
        return
    
    print(f"检查目录: {fake_root}\n")
    
    has_json, no_json, missing_json_files = check_json_files(fake_root)
    total = has_json + no_json
    
    if total == 0:
        print("没有找到任何图片文件")
        return
    
    print("\n" + "="*70)
    print("统计结果:")
    print("="*70)
    print(f"总图片数: {total}")
    print(f"有JSON文件: {has_json} ({100.0 * has_json / total:.2f}%)")
    print(f"无JSON文件: {no_json} ({100.0 * no_json / total:.2f}%)")
    print("="*70)
    
    if missing_json_files:
        print(f"\n缺少JSON文件的图片 (共{len(missing_json_files)}个):")
        print("-"*70)
        
        # 按family/submodel分组显示
        from collections import defaultdict
        by_path = defaultdict(list)
        
        for img_path in missing_json_files:
            try:
                rel_path = img_path.relative_to(fake_root)
                if len(rel_path.parts) >= 2:
                    family_sub = f"{rel_path.parts[0]}/{rel_path.parts[1]}"
                else:
                    family_sub = str(rel_path.parent)
                by_path[family_sub].append(img_path.name)
            except ValueError:
                by_path["other"].append(str(img_path))
        
        for path_key in sorted(by_path.keys()):
            files = by_path[path_key]
            print(f"\n{path_key}/ ({len(files)}个文件):")
            for fname in sorted(files)[:10]:  # 每个目录最多显示10个
                print(f"  - {fname}")
            if len(files) > 10:
                print(f"  ... 还有 {len(files) - 10} 个文件")
    else:
        print("\n✓ 所有fake图片都有对应的JSON文件！")


if __name__ == "__main__":
    main()