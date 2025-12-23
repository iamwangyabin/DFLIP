#!/usr/bin/env python3
"""
多分类测试脚本，测试家族分类器在27个AI生成图像家族上的表现

此脚本专门用于测试家族分类器的多分类性能，提供详细的评估指标和可视化结果。
只对伪造图像进行分类测试（真实图像会被过滤掉）。

用法:
    python scripts/test_family_classifier.py \
        --config configs/family_classifier_clip_base.yaml \
        --checkpoint ./checkpoints/family_classifier_clip_base/best_model.pt \
        --metadata ./tools/dflip3k_meta_fixed.json
"""

import argparse
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, top_k_accuracy_score
)
from tqdm import tqdm
import sys
import os

# Add project root to Python path to enable imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from dataset import create_profiling_dataloaders
from utils.transforms import build_transforms_from_config
from models.family_classifier import create_family_classifier

# Family ID to Name mapping
FAMILY_ID_TO_NAME = {
    0: "AuraFlow",
    1: "Chroma",
    2: "FLUX.1-d",
    3: "FLUX.1-s",
    4: "FLUX.2-d",
    5: "GPT-Img 1",
    6: "HiDream",
    7: "Hunyuan",
    8: "Illustrious",
    9: "Imagen 4",
    10: "Kolors",
    11: "NaBan.",
    12: "NaBan. Pro",
    13: "NoobAI",
    14: "PixArt",
    15: "Playground",
    16: "Pony v6",
    17: "Pony v7",
    18: "Qwen-Image",
    19: "SD 1.5",
    20: "SD 2.1",
    21: "SD 3.5-L",
    22: "SD 3.5-M",
    23: "SD XL",
    24: "SeeDream",
    25: "Sta. Cas.",
    26: "Z-Image"
}



def load_family_classifier(config_path, checkpoint_path, device='cuda'):
    """加载家族分类器模型"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建家族分类器模型
    model = create_family_classifier(config, verbose=False)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def test_family_classifier(model, dataloader, device='cuda'):
    """测试家族分类器，返回预测、标签和相关信息"""
    all_preds = []
    all_labels = []
    all_probs = []
    all_family_ids = []
    all_is_fake = []
    
    for batch in tqdm(dataloader, desc="Testing Family Classifier"):
        pixel_values = batch['pixel_values'].to(device)
        family_ids = batch['family_ids'].cpu().numpy()
        is_fake = batch['is_fake'].cpu().numpy()
        
        outputs = model(pixel_values)
        family_logits = outputs['family_logits']
        
        # 转换为概率
        probs = torch.softmax(family_logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        all_preds.append(preds)
        all_labels.append(family_ids)
        all_probs.append(probs)
        all_family_ids.append(family_ids)
        all_is_fake.append(is_fake)
    
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs),
            np.concatenate(all_family_ids),
            np.concatenate(all_is_fake))


def filter_valid_samples(predictions, labels, probabilities, family_ids, is_fake):
    """过滤有效样本：只保留伪造图像且有有效family_id的样本"""
    # 只保留伪造图像且family_id >= 0的样本
    valid_mask = (is_fake == 1) & (family_ids >= 0)
    
    if valid_mask.sum() == 0:
        print("[Warning] No valid samples found!")
        return None, None, None, None
    
    valid_preds = predictions[valid_mask]
    valid_labels = family_ids[valid_mask]  # 使用family_ids作为标签
    valid_probs = probabilities[valid_mask]
    valid_family_ids = family_ids[valid_mask]
    
    print(f"Filtered {valid_mask.sum()} valid samples from {len(predictions)} total samples")
    return valid_preds, valid_labels, valid_probs, valid_family_ids


def compute_multiclass_metrics(y_true, y_pred, y_probs, num_classes=27):
    """计算多分类评估指标"""
    # 基本准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # Top-k准确率
    top3_accuracy = top_k_accuracy_score(y_true, y_probs, k=3)
    top5_accuracy = top_k_accuracy_score(y_true, y_probs, k=5)
    
    # 每个类别的精确率、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )
    
    # 计算每个类别的accuracy (正确分类的样本数 / 该类别的总样本数)
    # 对于多分类，每个类别的accuracy就是recall
    per_class_accuracy = recall.copy()
    
    # 宏平均和微平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'per_class': {
            'accuracy': per_class_accuracy.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'micro_avg': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'confusion_matrix': cm.tolist()
    }


def analyze_confusion_matrix(cm, num_classes=27):
    """分析混淆矩阵，找出最容易混淆的类别对"""
    confused_pairs = []
    
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            confusion_count = cm[i][j] + cm[j][i]
            if confusion_count > 0:
                confused_pairs.append({
                    'families': [int(i), int(j)],
                    'confusion_count': int(confusion_count),
                    'i_to_j': int(cm[i][j]),
                    'j_to_i': int(cm[j][i])
                })
    
    # 按混淆次数排序
    confused_pairs.sort(key=lambda x: x['confusion_count'], reverse=True)
    
    return confused_pairs


def create_confusion_matrix_plot(cm, output_path, num_classes=27):
    """创建混淆矩阵热力图"""
    plt.figure(figsize=(12, 10))
    
    # 计算百分比矩阵用于显示
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建家族标签（只显示名称，不显示ID）
    family_labels = []
    for i in range(num_classes):
        family_name = FAMILY_ID_TO_NAME.get(i, f"unk_{i}")
        # 使用简短标签以适应图表
        short_name = family_name[:8] if len(family_name) > 8 else family_name
        family_labels.append(short_name)
    
    # 创建热力图
    sns.heatmap(cm_percent, annot=False, fmt='.1f', cmap='Blues',
                xticklabels=family_labels, yticklabels=family_labels)
    
    # 不显示标题
    # plt.title('Family Classification Confusion Matrix (%)', fontsize=16)
    plt.xlabel('Predicted Family', fontsize=12)
    plt.ylabel('True Family', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def create_performance_plot(metrics, output_path, num_classes=27):
    """创建每个家族性能对比图"""
    families = list(range(num_classes))
    accuracy = metrics['per_class']['accuracy']
    precision = metrics['per_class']['precision']
    recall = metrics['per_class']['recall']
    f1 = metrics['per_class']['f1']
    support = metrics['per_class']['support']
    
    # 只显示有样本的家族
    valid_families = [i for i, s in enumerate(support) if s > 0]
    
    if not valid_families:
        print("[Warning] No families with samples found for performance plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 性能指标图
    x = np.arange(len(valid_families))
    width = 0.2
    
    ax1.bar(x - 1.5*width, [accuracy[i] for i in valid_families], width, label='Accuracy', alpha=0.8)
    ax1.bar(x - 0.5*width, [precision[i] for i in valid_families], width, label='Precision', alpha=0.8)
    ax1.bar(x + 0.5*width, [recall[i] for i in valid_families], width, label='Recall', alpha=0.8)
    ax1.bar(x + 1.5*width, [f1[i] for i in valid_families], width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Family ID')
    ax1.set_ylabel('Score')
    ax1.set_title('Per-Family Performance Metrics')
    ax1.set_xticks(x)
    # 只显示家族名称，不显示ID
    family_labels = [FAMILY_ID_TO_NAME.get(i, f'unk_{i}') for i in valid_families]
    ax1.set_xticklabels(family_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 样本数量图
    ax2.bar(valid_families, [support[i] for i in valid_families], alpha=0.7, color='green')
    ax2.set_xlabel('Family ID')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Distribution per Family')
    ax2.set_xticks(valid_families)
    ax2.set_xticklabels([FAMILY_ID_TO_NAME.get(i, f'unk_{i}') for i in valid_families],
                        rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to: {output_path}")


def print_detailed_results(metrics, confused_pairs, num_classes=27):
    """打印详细的测试结果"""
    print("\n" + "=" * 60)
    print("FAMILY CLASSIFIER TEST RESULTS")
    print("=" * 60)
    
    # 整体性能
    print(f"\n=== OVERALL PERFORMANCE ===")
    print(f"Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.1%}")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.1%}")
    
    print(f"\nMacro Average:")
    print(f"  Precision: {metrics['macro_avg']['precision']:.1%}")
    print(f"  Recall: {metrics['macro_avg']['recall']:.1%}")
    print(f"  F1-Score: {metrics['macro_avg']['f1']:.1%}")
    
    print(f"\nWeighted Average:")
    print(f"  Precision: {metrics['weighted_avg']['precision']:.1%}")
    print(f"  Recall: {metrics['weighted_avg']['recall']:.1%}")
    print(f"  F1-Score: {metrics['weighted_avg']['f1']:.1%}")
    
    # 每个家族的性能（只显示有样本的家族）
    print(f"\n=== PER-FAMILY PERFORMANCE ===")
    support = metrics['per_class']['support']
    accuracy = metrics['per_class']['accuracy']
    precision = metrics['per_class']['precision']
    recall = metrics['per_class']['recall']
    f1 = metrics['per_class']['f1']
    
    families_with_samples = [(i, support[i]) for i in range(num_classes) if support[i] > 0]
    families_with_samples.sort(key=lambda x: x[1], reverse=True)  # 按样本数排序
    
    # 显示所有有样本的家族，不截断
    for family_id, sample_count in families_with_samples:
        family_name = FAMILY_ID_TO_NAME.get(family_id, f"unknown_{family_id}")
        print(f"Family {family_id:2d} ({family_name}): Accuracy={accuracy[family_id]:.1%}, "
              f"Precision={precision[family_id]:.1%}, Recall={recall[family_id]:.1%}, "
              f"F1={f1[family_id]:.1%} ({sample_count} samples)")
    
    # 最容易混淆的家族对 - 显示所有混淆对
    print(f"\n=== ALL CONFUSED FAMILY PAIRS ===")
    for i, pair in enumerate(confused_pairs):  # 显示所有混淆对
        fam1, fam2 = pair['families']
        fam1_name = FAMILY_ID_TO_NAME.get(fam1, f"unknown_{fam1}")
        fam2_name = FAMILY_ID_TO_NAME.get(fam2, f"unknown_{fam2}")
        count = pair['confusion_count']
        i_to_j = pair['i_to_j']
        j_to_i = pair['j_to_i']
        print(f"{i+1:2d}. Family {fam1:2d} ({fam1_name}) ↔ Family {fam2:2d} ({fam2_name}): "
              f"{count} total confusions ({fam1}→{fam2}: {i_to_j}, {fam2}→{fam1}: {j_to_i})")
    
    print(f"\nTotal families with samples: {len(families_with_samples)}")
    print(f"Total test samples: {sum(support)}")
    
    # 显示所有家族的完整列表（包括没有样本的）
    print(f"\n=== COMPLETE FAMILY LIST ===")
    for i in range(num_classes):
        family_name = FAMILY_ID_TO_NAME.get(i, f"unknown_{i}")
        if support[i] > 0:
            print(f"Family {i:2d} ({family_name:20s}): "
                  f"Acc={accuracy[i]:.1%}, P={precision[i]:.1%}, R={recall[i]:.1%}, F1={f1[i]:.1%} "
                  f"({support[i]:4d} samples)")
        else:
            print(f"Family {i:2d} ({family_name:20s}): No test samples")


def save_results(metrics, confused_pairs, output_file):
    """保存结果到JSON文件"""
    results = {
        'overall_metrics': {
            'accuracy': metrics['accuracy'],
            'top3_accuracy': metrics['top3_accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'macro_avg': metrics['macro_avg'],
            'micro_avg': metrics['micro_avg'],
            'weighted_avg': metrics['weighted_avg']
        },
        'per_family_metrics': {},
        'confusion_matrix': metrics['confusion_matrix'],
        'all_confused_pairs': confused_pairs  # 保存所有混淆对
    }
    
    # 添加每个家族的详细指标
    for i in range(len(metrics['per_class']['precision'])):
        family_name = FAMILY_ID_TO_NAME.get(i, f"unknown_{i}")
        results['per_family_metrics'][str(i)] = {
            'family_name': family_name,
            'accuracy': metrics['per_class']['accuracy'][i],
            'precision': metrics['per_class']['precision'][i],
            'recall': metrics['per_class']['recall'][i],
            'f1': metrics['per_class']['f1'][i],
            'support': metrics['per_class']['support'][i]
        }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test Family Classifier")
    parser.add_argument('--config', required=True, help='Family classifier config file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--metadata', default='./tools/dflip3k_meta_fixed.json',
                       help='Metadata file (should use fixed version with real image family_ids)')
    parser.add_argument('--image-root', default='/media/dataset/person_dataset/DFLIP3K_processed_traintest')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    parser.add_argument('--num-families', type=int, default=27, help='Number of families')
    args = parser.parse_args()
    
    print(f"Testing Family Classifier...")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Metadata: {args.metadata}")
    
    # 检查文件是否存在
    if not Path(args.metadata).exists():
        print(f"[Error] Metadata file not found: {args.metadata}")
        print("Hint: You may need to use the fixed metadata file with real image family_ids.")
        print("Try using: --metadata ./tools/dflip3k_meta_fixed.json")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"[Error] Checkpoint file not found: {args.checkpoint}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("Loading family classifier model...")
    model, config = load_family_classifier(args.config, args.checkpoint, args.device)
    
    # 更新配置中的数据路径
    if "data" not in config:
        config["data"] = {}
    config["data"]["metadata_path"] = args.metadata
    config["data"]["image_root"] = args.image_root
    
    # 创建transforms和dataloader
    print("Creating test dataloader...")
    test_transform = build_transforms_from_config(config, 'test')
    
    _, _, test_loader = create_profiling_dataloaders(
        config,
        train_transform=None,
        val_transform=test_transform,
        test_transform=test_transform
    )
    
    # 运行测试
    print("Running inference...")
    predictions, labels, probabilities, family_ids, is_fake = test_family_classifier(
        model, test_loader, args.device
    )
    
    # 过滤有效样本
    print("Filtering valid samples...")
    valid_preds, valid_labels, valid_probs, valid_family_ids = filter_valid_samples(
        predictions, labels, probabilities, family_ids, is_fake
    )
    
    if valid_preds is None:
        print("[Error] No valid samples found for testing!")
        return
    
    # 计算评估指标
    print("Computing evaluation metrics...")
    metrics = compute_multiclass_metrics(
        valid_labels, valid_preds, valid_probs, args.num_families
    )
    
    # 分析混淆矩阵
    print("Analyzing confusion matrix...")
    confused_pairs = analyze_confusion_matrix(
        np.array(metrics['confusion_matrix']), args.num_families
    )
    
    # 打印详细结果
    print_detailed_results(metrics, confused_pairs, args.num_families)
    
    # 创建可视化
    print("Creating visualizations...")
    cm_plot_path = output_dir / "confusion_matrix.png"
    perf_plot_path = output_dir / "per_family_performance.png"
    
    create_confusion_matrix_plot(
        np.array(metrics['confusion_matrix']), cm_plot_path, args.num_families
    )
    create_performance_plot(metrics, perf_plot_path, args.num_families)
    
    # 保存结果
    results_file = output_dir / "family_classifier_results.json"
    save_results(metrics, confused_pairs, results_file)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total valid test samples: {len(valid_labels)}")
    print(f"Overall accuracy: {metrics['accuracy']:.1%}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)



if __name__ == '__main__':
    main()
