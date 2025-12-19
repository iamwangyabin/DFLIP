#!/usr/bin/env python3
"""Test AIDE Binary Classifier.

This script evaluates a trained AIDE model on the test set and provides
detailed performance metrics for real/fake detection.

Usage examples:
    python scripts/test_aide.py --config configs/aide_train_config.yaml \
        --checkpoint ./checkpoints/aide_binary/best_model.pt

    python scripts/test_aide.py --config configs/aide_train_config.yaml \
        --checkpoint ./checkpoints/aide_binary/checkpoint_epoch_24.pt \
        --save_predictions
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import json

import torch
import torch.nn as nn
import yaml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.transforms import build_train_val_transforms  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from models.aide import create_aide  # noqa: E402
from dataset import create_profiling_dataloaders  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test AIDE Binary Classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def test_aide(
    model: nn.Module,
    test_loader,
    device: torch.device,
    save_predictions: bool = False,
    output_dir: str = "./test_results"
):
    """Test AIDE model and compute detailed metrics."""
    
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_image_paths = []
    
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    print("Running inference on test set...")
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        pixel_values = batch["pixel_values"].to(device)
        is_fake = batch["is_fake"].to(device)
        
        # Forward pass
        outputs = model(pixel_values)
        loss = criterion(outputs['logits'], is_fake)
        
        # Get predictions and probabilities
        probabilities = torch.softmax(outputs['logits'], dim=1)
        _, predicted = outputs['logits'].max(1)
        
        # Store results
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_labels.extend(is_fake.cpu().numpy())
        
        total_loss += loss.item()
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    avg_loss = total_loss / len(test_loader)
    
    # Classification report
    class_names = ['Real', 'Fake']
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # ROC AUC
    fake_probabilities = all_probabilities[:, 1]  # Probability of being fake
    auc_score = roc_auc_score(all_labels, fake_probabilities)
    
    # Print results
    print("\n" + "=" * 60)
    print("AIDE Test Results")
    print("=" * 60)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    print("Confusion Matrix:")
    print("Predicted ->")
    print(f"Actual   Real  Fake")
    print(f"Real     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"Fake     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Calculate per-class metrics
    real_precision = report['Real']['precision']
    real_recall = report['Real']['recall']
    real_f1 = report['Real']['f1-score']
    
    fake_precision = report['Fake']['precision']
    fake_recall = report['Fake']['recall']
    fake_f1 = report['Fake']['f1-score']
    
    print(f"\nDetailed Metrics:")
    print(f"Real Images - Precision: {real_precision:.4f}, Recall: {real_recall:.4f}, F1: {real_f1:.4f}")
    print(f"Fake Images - Precision: {fake_precision:.4f}, Recall: {fake_recall:.4f}, F1: {fake_f1:.4f}")
    
    # Save results
    if save_predictions:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'roc_auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions.tolist(),
            'probabilities': all_probabilities.tolist(),
            'labels': all_labels.tolist(),
        }
        
        with open(output_path / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels, fake_probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AIDE ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('AIDE Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to: {output_path}")
        print(f"- test_results.json: Detailed metrics and predictions")
        print(f"- roc_curve.png: ROC curve plot")
        print(f"- confusion_matrix.png: Confusion matrix plot")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'auc_score': auc_score,
        'classification_report': report,
        'confusion_matrix': cm,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set random seed
    seed_everything(config["hardware"]["seed"])
    
    print("Building transforms...")
    _, val_transform = build_train_val_transforms(config)
    
    print("Creating test dataset...")
    _, _, test_loader = create_profiling_dataloaders(
        config, val_transform, val_transform
    )
    
    print("Creating AIDE model...")
    model = create_aide(config, verbose=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {epoch}")
    else:
        # Assume it's a direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    # Run test
    test_results = test_aide(
        model,
        test_loader,
        device,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()