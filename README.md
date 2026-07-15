# MCAQ-YOLO: Morphological Complexity-Aware Quantization for YOLO

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **MCAQ-YOLO: Morphological Complexity-Aware Quantization for Efficient Object Detection with Curriculum Learning**

## 📋 Overview

MCAQ-YOLO introduces a spatial quantization framework for object detection that dynamically allocates bit precision based on morphological complexity. By analyzing local visual characteristics through five complementary metrics (fractal dimension, texture entropy, gradient variance, edge density, and contour complexity), the framework allocates higher precision to complex regions and aggressive compression to simple ones.

### Key Features

- **Morphological Complexity Analysis**: Five tile-wise metrics fused by a learnable MLP (exact OpenCV path for offline scoring, vectorized GPU surrogates for training)
- **Complexity-to-Bit Mapping**: Learnable monotonic MLP (paper Eq.13-17) or a parameter-free linear mapper (the paper's ablation baseline)
- **Spatial Adaptive Quantization**: Tile-wise mixed precision (2-8 bits) applied to the backbone C3/C4/C5 feature maps via forward hooks, with a learned spatially-smoothed soft mask (Eq.19)
- **Curriculum Learning**: 3-stage schedule (high-precision warm-up → transition → full MCAQ) with temperature annealing and complexity-ordered data sampling
- **Knowledge Distillation**: Logit-level and feature-level matching against an FP32 teacher
- **Hardware-Aware Design**: Custom CUDA kernel for inference (with an automatic pure-PyTorch fallback) and a TensorRT plugin reference

For experimental results, see the paper (arXiv:2511.12976).

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yooooonjae/mcaq-yolo.git
cd mcaq-yolo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package (builds the CUDA kernel — requires torch + CUDA toolkit/nvcc)
pip install -e .
```

**CPU-only machines** (no CUDA toolkit): `pip install -e .` will fail while
building the CUDA extension. Install the dependencies and use the package
from the repo root instead — quantization automatically falls back to the
(slower) pure-PyTorch implementation:

```bash
pip install -r requirements.txt
python -m mcaq_yolo.train --config configs/train_config.yaml --device cpu
```

### Training
```bash
# Train with a configuration file (see "Training Configuration" below)
mcaq-yolo-train --config configs/train_config.yaml

# Override device / output directory from the CLI
mcaq-yolo-train \
    --config configs/train_config.yaml \
    --device cuda:0 \
    --output-dir outputs/experiment_1
```

The best weights (lowest validation loss) are saved to `<output_dir>/best.pt`.

### Inference
```bash
# Single image inference
mcaq-yolo-infer \
    --model outputs/best.pt \
    --source image.jpg \
    --visualize

# Batch inference on a directory
mcaq-yolo-infer \
    --model outputs/best.pt \
    --source /path/to/images \
    --save-dir results
```

### Tests
```bash
# From the repo root (CPU-friendly smoke tests)
python -m pytest mcaq_yolo/tests/test_smoke.py -v
```

## 📊 Model Architecture
```
MCAQ-YOLO
├── YOLOv8 Backbone (quantization hooks on the C3/C4/C5 outputs)
├── Morphological Complexity Analyzer
│   ├── Fractal Dimension
│   ├── Texture Entropy (LBP)
│   ├── Gradient Variance
│   ├── Edge Density
│   └── Contour Complexity
├── Complexity-to-Bit Mapping Network
│   └── Learnable monotonic mapping (or linear ablation)
└── Spatial Adaptive Quantization
    └── Tile-wise mixed precision + learned soft mask
```

## 🎯 Training Configuration

Example `configs/train_config.yaml` (every key below is consumed by the
`Trainer`; the dataset itself is described by a standard YOLOv8 dataset yaml):

```yaml
# Model configuration
model:
  name: yolov8n
  teacher_path: yolov8n.pt   # FP32 teacher for knowledge distillation
  num_classes: 80

# Dataset configuration (YOLOv8 format)
data:
  yaml_path: /path/to/dataset.yaml  # standard YOLOv8 yaml (path/train/val/names)
  train: images/train               # relative to the dataset root
  val: images/val
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
  grid_size: 8          # tiles per spatial dimension
  bit_mapping: mlp      # 'mlp' (Eq.13-17) | 'linear' (paper's ablation)
  normalize_complexity: false

# Curriculum learning (3-stage schedule)
curriculum:
  enabled: true
  warmup_epochs: 20     # Stage 1 boundary (Twarm)
  transition_epochs: 50 # Stage 2 boundary
  initial_complexity: 0.2
  initial_temperature: 10.0
  lambda_smooth: 0.1

# Knowledge distillation
distillation:
  enabled: true

# Optimizer
optimizer:
  type: adamw           # 'adamw' | anything else falls back to Adam
  weight_decay: 0.05
  betas: [0.9, 0.999]

# Learning rate scheduler
scheduler:
  type: cosine          # cosine annealing with linear LR warmup
  warmup_epochs: 5
  eta_min: 0.000001

# Training settings
training:
  amp: true             # mixed precision

# Hardware
device: cuda
output_dir: outputs
```

> **Note**: The loss weights of Eq.(20) — λ1 (bit budget, annealed 0.01→0.1),
> λ2 (smoothness), λ3 (distillation, 0.5), λ4 (regularization, 1e-4) — follow
> the paper's Table X schedule inside `CurriculumScheduler` and are not set
> from the config file (λ2's base value is `curriculum.lambda_smooth`).

## 🔧 Advanced Usage

### Morphological Complexity Analysis
```python
import torch
from mcaq_yolo.core.morphology import MorphologicalComplexityAnalyzer

analyzer = MorphologicalComplexityAnalyzer(
    grid_size=8,           # tiles per spatial dimension (8x8 grid)
    cache_size=10000,
    device='cuda',
    metric_backend='gpu',  # 'gpu' (vectorized, training) | 'cv2' (exact, offline)
)

features = torch.rand(2, 3, 640, 640, device='cuda')
complexity_map = analyzer(features)          # (B, ht, wt) in [0, 1]
scores = analyzer.score_image(features)      # (B,) per-image score (Eq.8)
```

### Bit Allocation
```python
from mcaq_yolo.core.bit_allocation import (
    ComplexityToBitMappingNetwork,  # learnable monotonic MLP (Eq.13-17)
    LinearBitMapper,                # parameter-free linear ablation
)

mapper = ComplexityToBitMappingNetwork(
    min_bits=2,
    max_bits=8,
    hidden_dims=[32, 64, 32],   # paper Table X
    enforce_monotonicity=True,  # Eq.18: |W| re-projection
)

# temperature = alpha_t (Algorithm 3 line 13): anneals 10 -> 1 during training
bit_map = mapper(complexity_map, temperature=1.0)  # (B, ht, wt), integer bits
```

### Curriculum Learning
```python
from mcaq_yolo.core.curriculum import CurriculumScheduler

curriculum = CurriculumScheduler(
    warmup_epochs=20,       # Stage 1 boundary (Twarm, paper Table X)
    transition_epochs=50,   # Stage 2 boundary (paper Fig.3)
    total_epochs=300,
    curriculum_type='exponential',  # 'linear' | 'exponential' | 'cosine' | 'step'
)

stage = curriculum.get_stage(epoch=30)             # 1 | 2 | 3
alpha_t = curriculum.get_temperature(epoch=30)     # 1 + 9*exp(-5t/T)
tau_t = curriculum.get_complexity_threshold(30)    # data-curriculum threshold
weights = curriculum.get_loss_weights(epoch=30)    # Eq.(20) lambdas
```

### Analysis Scripts
```bash
# M3: post-hoc bit-placement permutation test (does placement matter?)
python -m mcaq_yolo.scripts.m3_permutation \
    --checkpoint outputs/<run>/best.pt --data <dataset.yaml> \
    --train-rel images/train --val-rel images/val

# M4: within-image complexity variation vs. MCAQ gain
python -m mcaq_yolo.scripts.m4_variation_gain \
    --ckpt-spatial outputs/<spatial>/best.pt \
    --ckpt-uniform outputs/<uniform>/best.pt \
    --data <dataset.yaml> --val-rel images/val
```

## 📁 Project Structure
```
mcaq-yolo/                      # repo root
├── mcaq_yolo/                  # the Python package
│   ├── core/
│   │   ├── morphology.py       # complexity analysis (cv2 + GPU backends)
│   │   ├── bit_allocation.py   # bit mapping network + linear ablation
│   │   ├── quantization.py     # spatial adaptive quantization (STE / CUDA)
│   │   └── curriculum.py       # curriculum learning
│   ├── models/
│   │   └── mcaq_yolo.py        # main model + combined loss
│   ├── ops/                    # custom CUDA kernel
│   │   ├── setup.py            # standalone kernel build script
│   │   └── src/
│   │       ├── mcaq_kernel.cu
│   │       └── mcaq_ops.cpp
│   ├── engine/
│   │   └── MCAQPlugin.cpp      # TensorRT plugin (reference)
│   ├── scripts/                # analysis scripts (M3 / M4)
│   ├── tests/
│   │   └── test_smoke.py       # CPU-friendly smoke tests
│   ├── utils/
│   │   ├── dataset.py          # data utilities + dataset complexity scoring
│   │   ├── evaluation.py       # mAP and quantization-impact evaluation
│   │   ├── visualization.py    # plotting
│   │   └── model_utils.py
│   ├── train.py                # Trainer + `mcaq-yolo-train` CLI
│   └── inference.py            # Predictor + `mcaq-yolo-infer` CLI
├── configs/
│   └── train_config.yaml
├── examples/
│   └── train_examples.py
├── requirements.txt
└── setup.py                    # package installer (builds the CUDA extension)
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

Contributions are welcome — please open an issue or a pull request.

## 📄 License

This project is licensed under the MIT License.

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
