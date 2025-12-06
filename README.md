# DFLIP: Linguistic Profiling of Deepfakes

Official implementation of **"Linguistic Profiling of Deepfakes"** - A next-generation deepfake detection framework based on Qwen2.5-VL-7B-Instruct.

## 📖 Overview

Unlike traditional binary classification approaches, DFLIP implements **Linguistic Profiling** for comprehensive deepfake analysis:

1. **Detection**: Distinguish real from fake images
2. **Identification**: Identify the specific generator model (e.g., MidJourney, DALL·E, Stable Diffusion)
3. **Prompt Prediction**: Predict the text prompt used to generate the image
4. **Localization**: Highlight manipulated regions with heatmaps

## 🏗️ Architecture: DFLIP-Net

DFLIP-Net uses a two-stage architecture for efficient multi-task learning:

### Stage 1: The Profiler (Vision Expert)
- **Purpose**: Visual forensics - detect, identify, and localize forgeries
- **Model**: Qwen2.5-VL Vision Encoder + LoRA
- **Tasks**:
  - Binary classification (real/fake)
  - Multi-class identification (generator model)
  - Segmentation (forgery localization)
- **Output**: Hard metrics and forgery heatmaps

### Stage 2: The Interpreter (Language Expert)
- **Purpose**: Natural language interpretation and prompt prediction
- **Model**: Full Qwen2.5-VL with frozen Stage 1 weights + LLM LoRA
- **Tasks**:
  - Prompt reconstruction
  - Human-readable analysis reports
- **Output**: Natural language explanations and predicted prompts

## 📊 Dataset: DFLIP-3K

- **Scale**: 300K images from 3K generator models
- **Prompts**: 190K unique generation prompts
- **Annotations**: 
  - Real/Fake labels
  - Generator model IDs
  - Ground truth prompts
  - Forgery masks (where available)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DFLIP.git
cd DFLIP

# Install dependencies
pip install -r requirements.txt
```

### Prepare Dataset

```bash
# Convert your DFLIP-3K dataset to the required format
python dataset/build_dflip3k.py \
    --data-root /path/to/dflip3k \
    --output assets/dflip3k_meta.json \
    --validate-images
```

Expected directory structure:
```
/path/to/dflip3k/
├── real/
│   └── images/
│       ├── img001.jpg
│       └── ...
└── fake/
    ├── stable-diffusion-v1.5/
    │   ├── images/
    │   ├── prompts.txt
    │   └── masks/
    ├── midjourney-v5/
    │   └── ...
    └── ...
```

### Training

#### Stage 1: Train the Profiler

```bash
python scripts/train_profiler.py \
    --config configs/dflip_config.yaml
```

This will train the vision encoder with LoRA for multi-task learning (detection, identification, localization).

**Output**: `checkpoints/stage1/best_model/`

#### Stage 2: Train the Interpreter

```bash
python scripts/train_interpreter.py \
    --config configs/dflip_config.yaml \
    --stage1-checkpoint checkpoints/stage1/best_model
```

This will train the LLM with LoRA for prompt prediction, using frozen Stage 1 weights.

**Output**: `checkpoints/stage2/best_model/`

### Inference

#### Full Pipeline (Both Stages)

```bash
python scripts/run_inference.py \
    --image path/to/test_image.jpg \
    --stage both \
    --stage1-checkpoint checkpoints/stage1/best_model \
    --stage2-checkpoint checkpoints/stage2/best_model \
    --save-visualization
```

#### Stage 1 Only (Fast Profiling)

```bash
python scripts/run_inference.py \
    --image path/to/test_image.jpg \
    --stage profiling \
    --stage1-checkpoint checkpoints/stage1/best_model \
    --save-visualization
```

#### Stage 2 Only (Prompt Prediction)

```bash
python scripts/run_inference.py \
    --image path/to/test_image.jpg \
    --stage interpreting \
    --stage1-checkpoint checkpoints/stage1/best_model \
    --stage2-checkpoint checkpoints/stage2/best_model
```

## 📁 Project Structure

```
DFLIP/
├── dflip_dataset/          # Data processing
│   ├── dataset.py          # DFLIPDataset with dual modes
│   ├── formatting.py       # Prompt formatting utilities
│   └── build_dflip3k.py    # Dataset preprocessing script
├── dflip_models/           # Model definitions
│   ├── qwen_vision.py      # Stage 1: Profiler
│   ├── qwen_full.py        # Stage 2: Interpreter
│   ├── heads.py            # Task-specific heads
│   └── __init__.py
├── scripts/                # Training & inference
│   ├── train_profiler.py   # Stage 1 training
│   ├── train_interpreter.py # Stage 2 training
│   └── run_inference.py    # End-to-end inference
├── configs/
│   └── dflip_config.yaml   # All hyperparameters
├── assets/
│   └── dflip3k_meta.json   # Dataset metadata
├── checkpoints/            # Model weights (gitignored)
├── outputs/                # Inference results
└── requirements.txt
```

## ⚙️ Configuration

All training and inference settings are in [`configs/dflip_config.yaml`](configs/dflip_config.yaml):

- **Model**: Base model, LoRA configurations
- **Data**: Paths, splits, augmentation
- **Training**: Learning rates, batch sizes, loss weights
- **Inference**: Generation parameters

Key parameters:

```yaml
model:
  base_model: "Qwen/Qwen2.5-VL-7B-Instruct"
  stage1_lora:
    r: 16
    lora_alpha: 32
  stage2_lora:
    r: 16
    lora_alpha: 32

stage1_training:
  batch_size: 8
  learning_rate: 2.0e-4
  loss_weights:
    detection: 1.0
    identification: 1.0
    localization: 0.5

stage2_training:
  batch_size: 4
  learning_rate: 1.0e-4
  freeze_vision: true
```

## 🎯 Features

- **Dual-Mode Dataset**: Automatic task mode switching for Stage 1 (profiling) and Stage 2 (interpreting)
- **Multi-Task Learning**: Joint optimization of detection, identification, and localization
- **LoRA Efficiency**: Parameter-efficient fine-tuning with PEFT
- **Mixed Precision**: BF16 training for efficiency
- **Distributed Training**: Multi-GPU support with DDP/DeepSpeed
- **Checkpointing**: Automatic best model saving and resumption
- **Logging**: WandB integration for experiment tracking
- **Visualization**: Heatmap overlays and result plots

## 🔬 Model Architecture Details

### Stage 1: DFLIPProfiler

```
Qwen2.5-VL Vision Encoder (with LoRA)
    ↓
├─→ Detection Head → Real/Fake (BCE Loss)
├─→ Identification Head → Generator ID (CE Loss)
└─→ Localization Head → Forgery Mask (Dice Loss)
```

**Loss Function**: λ₁·BCE + λ₂·CrossEntropy + λ₃·Dice

### Stage 2: DFLIPInterpreter

```
┌─ Qwen2.5-VL Vision Encoder (Frozen from Stage 1)
│
├─ Qwen2.5-VL Language Model (with LoRA)
│
└─→ Generated Text: Prompt + Analysis
```

**Training**: Supervised fine-tuning (SFT) with conversation format

## 📊 Example Output

### Stage 1 Output
```
Detection: FAKE (98.5%)
Generator ID: 3 (Stable-Diffusion-v1.5)
Forgery Localization: [Heatmap saved]
```

### Stage 2 Output
```
Analysis Result:

1. Authenticity: This image is AI-generated (deepfake).

2. Generator Model: Stable-Diffusion-v1.5

3. Predicted Prompt:
a futuristic city with flying cars at sunset, highly detailed, 
cinematic lighting, digital art, 8k resolution

The image exhibits typical characteristics of AI-generated content from this model.
```

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 24GB+ VRAM (for 7B model)

## 🔧 Advanced Usage

### Debug Mode

```bash
# Quick test with limited steps
python scripts/train_profiler.py --debug --steps 10
```

### Resume Training

```bash
python scripts/train_profiler.py --resume checkpoints/stage1/checkpoint-epoch-5
```

### Custom Config

```bash
python scripts/train_profiler.py --config my_custom_config.yaml
```

## 📝 Citation

If you use DFLIP in your research, please cite:

```bibtex
@article{yourname2024dflip,
  title={Linguistic Profiling of Deepfakes},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent Qwen2.5-VL foundation model
- **PEFT Library** for efficient LoRA implementation
- **DFLIP-3K Dataset** contributors

## 🐛 Issues & Contributing

Found a bug or want to contribute? Please open an issue or submit a pull request!

## 📧 Contact

For questions or collaborations, please contact: [your.email@example.com]

---

**Built with ❤️ for advancing deepfake detection research**
