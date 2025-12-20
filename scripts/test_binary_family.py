#!/usr/bin/env python3
"""
二分类测试脚本，测试各个方法在每个family上的表现

注意：此脚本需要使用修复过的metadata文件，其中real图片也有family_id。
每个family的测试包含该family的fake图片和对应的real图片。

用法:
    # 首先修复metadata文件（如果还没有）
    python tools/fix_real_image_ids.py \
        --input ./tools/dflip3k_meta_processed.json \
        --output ./tools/dflip3k_meta_fixed.json
    
    # 然后运行测试
    python scripts/test_binary_family.py \
        --method npr \
        --config configs/npr_train_config.yaml \
        --checkpoint ./checkpoints/npr_binary/best_model.pt \
        --metadata ./tools/dflip3k_meta_fixed.json
"""

import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm
import sys
import os

# Add project root to Python path to enable imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 使用现有的imports
from dataset import create_profiling_dataloaders
from utils.transforms import build_transforms_from_config
from models import create_npr, create_rine, create_ojha, create_aide, create_cnndet


def load_model(method, config_path, checkpoint_path, device='cuda'):
    """加载模型"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 根据方法创建模型
    if method == 'npr':
        model = create_npr(config, verbose=False)
    elif method == 'rine':
        model = create_rine(config, verbose=False)
    elif method == 'ojha':
        model = create_ojha(config, verbose=False)
    elif method == 'aide':
        model = create_aide(config, verbose=False)
    elif method == 'cnndet':
        model = create_cnndet(config, verbose=False)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def test_model(model, dataloader, device='cuda'):
    """测试模型，返回预测、标签和family_ids"""
    all_preds = []
    all_labels = []
    all_scores = []
    all_family_ids = []
    
    for batch in tqdm(dataloader, desc="Testing"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['is_fake'].cpu().numpy()
        family_ids = batch['family_ids'].cpu().numpy()
        
        outputs = model(pixel_values)
        
        # 处理不同的输出格式
        if 'logits' in outputs:
            logits = outputs['logits']
        elif 'detection_logits' in outputs:
            logits = outputs['detection_logits']
        else:
            raise ValueError("Unknown output format")
        
        # 转换为概率
        if logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]  # fake的概率
        else:
            probs = torch.sigmoid(logits.squeeze())
        
        scores = probs.cpu().numpy()
        preds = (scores > 0.5).astype(int)
        
        all_preds.append(preds)
        all_labels.append(labels)
        all_scores.append(scores)
        all_family_ids.append(family_ids)
    
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_scores),
            np.concatenate(all_family_ids))


def group_predictions_by_family(predictions, labels, scores, family_ids):
    """按family分组预测结果"""
    family_preds = defaultdict(list)
    family_labels = defaultdict(list)
    family_scores = defaultdict(list)
    
    for i in range(len(predictions)):
        fid = int(family_ids[i])  # Convert NumPy int64 to Python int
        if fid >= 0:  # 只处理有效的family_id (>=0)
            family_preds[fid].append(predictions[i])
            family_labels[fid].append(labels[i])
            family_scores[fid].append(scores[i])
        else:
            # 跳过没有family_id的图片 (family_id = -1)
            print(f"[Warning] Skipping prediction for image without valid family_id: {fid}")
    
    return family_preds, family_labels, family_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['npr', 'rine', 'ojha', 'aide', 'cnndet'])
    parser.add_argument('--config', required=True, help='Model config file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--metadata', default='./tools/dflip3k_meta_fixed.json',
                       help='Metadata file (should use fixed version with real image family_ids)')
    parser.add_argument('--image-root', default='/media/dataset/person_dataset/DFLIP3K_processed_traintest')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    print(f"Testing method: {args.method}")
    print(f"Using metadata: {args.metadata}")
    
    # 检查metadata文件是否存在
    if not Path(args.metadata).exists():
        print(f"[Error] Metadata file not found: {args.metadata}")
        if 'dflip3k_meta_processed.json' in args.metadata:
            print("Hint: You may need to use the fixed metadata file with real image family_ids.")
            print("Try using: --metadata ./tools/dflip3k_meta_fixed.json")
            print("Or run: python tools/fix_real_image_ids.py --input ./tools/dflip3k_meta_processed.json --output ./tools/dflip3k_meta_fixed.json")
        return
    
    # 加载模型
    model = load_model(args.method, args.config, args.checkpoint, args.device)
    
    # 加载配置用于transforms
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 确保config中有必要的数据路径信息
    if "data" not in config:
        config["data"] = {}
    config["data"]["metadata_path"] = args.metadata
    config["data"]["image_root"] = args.image_root
    
    # 创建transforms
    test_transform = build_transforms_from_config(config, 'test')
    
    # 创建dataloader (修正函数调用)
    _, _, test_loader = create_profiling_dataloaders(
        config,
        train_transform=None,
        val_transform=test_transform,
        test_transform=test_transform
    )
    
    # 测试模型
    print("Running inference...")
    predictions, labels, scores, family_ids = test_model(model, test_loader, args.device)
    
    # 按family分组结果
    print("Grouping results by family...")
    family_preds, family_labels, family_scores = group_predictions_by_family(
        predictions, labels, scores, family_ids
    )
    
    # 初始化结果字典
    family_results = {}
    
    # 计算指标 - 只计算有数据的family
    for fid in sorted(family_preds.keys()):
        if len(family_preds[fid]) > 0:
            preds = np.array(family_preds[fid])
            lbls = np.array(family_labels[fid])
            scrs = np.array(family_scores[fid])
            
            # 检查是否有足够的正负样本
            unique_labels = np.unique(lbls)
            if len(unique_labels) < 2:
                print(f"[Warning] Family {fid} has only one class (labels: {unique_labels}), skipping AP calculation")
                acc = accuracy_score(lbls, preds) * 100
                ap = 0.0  # 无法计算AP
            else:
                acc = accuracy_score(lbls, preds) * 100
                ap = average_precision_score(lbls, scrs) * 100
            
            # Ensure fid is Python int for JSON serialization
            fid_key = int(fid) if hasattr(fid, 'item') else fid
            family_results[fid_key] = {'accuracy': acc, 'ap': ap}
            
            # 显示每个family的样本统计
            fake_count = np.sum(lbls == 1)
            real_count = np.sum(lbls == 0)
            print(f"Family {fid}: Acc={acc:.1f}%, AP={ap:.1f}% (fake={fake_count}, real={real_count})")
    
    # 检查是否有结果
    if not family_results:
        print("[Error] No valid family results found. Please check your data and metadata.")
        return
    
    # 计算平均值
    avg_acc = np.mean([r['accuracy'] for r in family_results.values()])
    avg_ap = np.mean([r['ap'] for r in family_results.values()])
    
    print(f"\nAverage: Acc={avg_acc:.1f}%, AP={avg_ap:.1f}%")
    print(f"Total families tested: {len(family_results)}")
    
    # 保存结果
    output_file = f"{args.method}_family_results.json"
    with open(output_file, 'w') as f:
        json.dump(family_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()