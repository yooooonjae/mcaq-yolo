# MCAQ-YOLO: Morphological Complexity-Aware Quantization for YOLO

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **MCAQ-YOLO: Morphological Complexity-Aware Quantization for Efficient Object Detection with Curriculum Learning**

## 📋 Overview

MCAQ-YOLO introduces a novel spatial quantization framework for object detection that dynamically allocates bit precision based on morphological complexity. By analyzing local visual characteristics through five complementary metrics (fractal dimension, texture entropy, gradient variance, edge density, and contour complexity), the framework achieves superior detection accuracy with aggressive compression ratios.

### Key Features

- **Morphological Complexity Analysis**: Multi-metric assessment of spatial regions for informed bit allocation
- **Curriculum Learning**: Progressive training strategy for stable optimization
- **Spatial Adaptive Quantization**: Tile-wise mixed-precision with smooth transitions
- **Hardware-Aware Design**: Optimized for modern accelerators with kernel fusion

### Performance Highlights

- **3.5% mAP improvement** over uniform 4-bit quantization
- **7.6× model compression** with minimal accuracy loss
- **40% faster convergence** with curriculum learning
- **Strong correlation** (ρ=0.89) between complexity and quantization sensitivity

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yooooonjae/mcaq-yolo.git
cd mcaq-yolo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Training
```bash
# Train with default configuration
mcaq-yolo-train --config configs/train_config.yaml

# Train with custom settings
mcaq-yolo-train \
    --config configs/train_config.yaml \
    --device cuda:0 \
    --output-dir outputs/experiment_1

# Resume from checkpoint
mcaq-yolo-train \
    --config configs/train_config.yaml \
    --resume outputs/experiment_1/latest.pth
```

### Inference
```bash
# Single image inference
mcaq-yolo-infer \
    --model outputs/best.pth \
    --source image.jpg \
    --visualize

# Batch inference on directory
mcaq-yolo-infer \
    --model outputs/best.pth \
    --source /path/to/images \
    --save-dir results
```

## 📊 Model Architecture
```
MCAQ-YOLO
├── YOLOv8 Backbone
├── Morphological Complexity Analyzer
│   ├── Fractal Dimension
│   ├── Texture Entropy (LBP)
│   ├── Gradient Variance
│   ├── Edge Density
│   └── Contour Complexity
├── Complexity-to-Bit Mapping Network
│   └── Learnable monotonic mapping
└── Spatial Adaptive Quantization
    └── Tile-wise mixed precision
```

## 🎯 Training Configuration

Create a configuration file `configs/train_config.yaml`:
```yaml
# Model configuration
model:
  name: yolov8n
  pretrained: true
  teacher_path: yolov8n.pt

# Dataset configuration
data:
  train_path: /path/to/train
  val_path: /path/to/val
  img_size: 640
  num_workers: 8

# Training configuration
epochs: 300
batch_size: 16
learning_rate: 0.001

# Quantization configuration
quantization:
  min_bits: 2
  max_bits: 8
  target_bits: 4.0

# Curriculum learning
curriculum:
  enabled: true
  warmup_epochs: 30
  initial_complexity: 0.2
  initial_temperature: 10.0
  type: exponential

# Optimizer
optimizer:
  type: adamw
  weight_decay: 0.05

# Learning rate scheduler
scheduler:
  type: cosine

# Loss weights
loss:
  lambda_bit: 0.01
  lambda_smooth: 0.001
  lambda_kd: 0.5
  lambda_reg: 0.0001

# Training settings
training:
  grad_clip: 1.0
  save_interval: 10
  eval_interval: 5

# Hardware
device: cuda
output_dir: outputs
```

## 📈 Results

### Quantization Performance

| Method | Bits | mAP@0.5 | mAP@0.5:0.95 | Size (MB) | FPS |
|--------|------|---------|--------------|-----------|-----|
| YOLOv8-FP32 | 32 | 89.3% | 68.1% | 108.3 | 92 |
| Uniform-4bit | 4 | 82.1% | 58.3% | 13.5 | 156 |
| **MCAQ-YOLO** | 4.2 | **85.6%** | **63.2%** | 14.2 | 151 |

### Complexity Analysis

| Class | Complexity | Allocated Bits | mAP Drop (3-bit) |
|-------|------------|----------------|------------------|
| Person | 0.72 | 5.8 | 17.2% |
| Helmet | 0.25 | 4.1 | 5.3% |
| Background | 0.21 | 3.8 | 2.1% |

## 🔧 Advanced Usage

### Custom Morphological Metrics
```python
from mcaq_yolo.core.morphology import MorphologicalComplexityAnalyzer

# Create custom analyzer
analyzer = MorphologicalComplexityAnalyzer(
    tile_sizes=[8, 16, 32],
    cache_size=2000,
    device='cuda'
)

# Compute complexity for your data
complexity_map = analyzer(features)
```

### Bit Allocation Policies
```python
from mcaq_yolo.core.bit_allocation import AdaptiveBitAllocation

# Use different allocation policies
allocator = AdaptiveBitAllocation(
    min_bits=2,
    max_bits=8,
    target_bits=4.0,
    policy='exponential'  # 'linear', 'exponential', 'threshold', 'learned'
)
```

### Curriculum Learning Strategies
```python
from mcaq_yolo.core.curriculum import CurriculumScheduler

# Configure curriculum
curriculum = CurriculumScheduler(
    warmup_epochs=30,
    total_epochs=300,
    curriculum_type='cosine'  # 'linear', 'exponential', 'cosine', 'step'
)
```

## 📁 Project Structure
```
mcaq_yolo/
├── core/
│   ├── morphology.py       # Complexity analysis
│   ├── bit_allocation.py   # Bit mapping network
│   ├── quantization.py     # Spatial quantization (Updated with CUDA support)
│   └── curriculum.py       # Curriculum learning
├── models/
│   ├── mcaq_yolo.py        # Main model
│   └── __init__.py
├── ops/                    # (New) CUDA Kernel & Ops
│   └── src/
│       ├── mcaq_kernel.cu  # (New) CUDA Kernel implementation
│       └── mcaq_ops.cpp    # (New) C++ Binding for PyTorch
├── engine/                 # (New) TensorRT Plugin
│   └── MCAQPlugin.cpp      # (New) TensorRT Plugin implementation
├── utils/
│   ├── dataset.py          # Data utilities
│   ├── evaluation.py       # Metrics
│   ├── visualization.py    # Plotting
│   └── model_utils.py
├── configs/                # Configuration files
│   └── train_config.yaml
├── train.py                # Training script
├── inference.py            # Inference script
└── setup.py                # (Modified) Build script for CUDA extensions
```

## 🎓 Citation

If you use MCAQ-YOLO in your research, please cite:
```bibtex
@article{mcaqyolo2025,
  title={MCAQ-YOLO: Morphological Complexity-Aware Quantization for Efficient Object Detection with Curriculum Learning},
  author={Seo, Yoonjae and Elbasani, E. and Lee, Jaehong},
  journal={arXiv preprint arXiv:2511.12976},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- PyTorch team for the deep learning framework
- Open source community for valuable feedback

## 📧 Contact

- **Corresponding Author**: Jaehong Lee (jlee@sejong.ac.kr)
- **First Author**: Yoonjae Seo (22013378@sju.ac.kr)
- **Second Author**: E. Elbasani (ermal.elbasani@sejong.ac.kr)

---

**Note**: This is research code. While we strive for reliability, please use with appropriate caution in production environments.
