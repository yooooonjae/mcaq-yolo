# MCAQ-YOLO: Morphological Complexity-Aware Quantization for YOLO

[![CI](https://github.com/yooooonjae/mcaq-yolo/actions/workflows/ci.yml/badge.svg)](https://github.com/yooooonjae/mcaq-yolo/actions/workflows/ci.yml)
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

This project is licensed under the MIT License — see [LICENSE](LICENSE).

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

## v0.2.0 — Review patch notes (2026-07)

Behavior-changing fixes from an independent code review. **Numbers produced
with v0.1.x are not directly comparable** where noted.

- **Surrogate/reference parity (behavior change).** The GPU morphology
  surrogates now replicate the cv2 recipe's operator semantics
  (`canny_impl='cv2compat'`: intensity-Otsu thresholds consumed in L1
  gradient units; `binarize_impl='adaptive'`: tensorized
  `adaptiveThreshold(GAUSSIAN,11,2)`; `contour_components=True`:
  Euler-number K correction toward Eq.24's per-contour mean). Measured on
  natural images, fused-map correlation gpu-vs-cv2 improved from r≈0.45 to
  r≈0.88. Pre-review dynamics remain available via
  `canny_impl='legacy', binarize_impl='otsu', contour_components=False`.
- **Curriculum scores are augmentation-free (behavior change).** Algorithm 3's
  SortByComplexity previously ordered one random mosaic composite per index;
  scores now come from a val-mode (no-augment) copy of the train set, are
  path-aligned to train order, and the cache self-invalidates on
  backend/imgsz/file-list changes (`curriculum.score_backend: train|gpu|cv2`).
- **Reproducibility.** `seed`/`deterministic` config keys and `--seed` CLI;
  see `mcaq_yolo/utils/repro.py` for stated limits.
- **Validation regime match.** `Trainer.evaluate()` now receives the epoch's
  annealed temperature instead of a fixed 1.0.
- **Flat-map bit allocation.** `LinearBitMapper` maps spatially flat
  complexity through absolute values (uniform C=0.5 → 5-bit) instead of
  collapsing to `b_min`.
- **Packaging.** `import mcaq_yolo` no longer requires `ultralytics`
  (PEP 562 lazy model/trainer); `pip install .` works without torch/nvcc
  (CUDA kernel builds only when a toolchain is present; `MCAQ_SKIP_CUDA=1`
  to skip); `pip install -e .[dev]` brings pytest; line endings normalized.
- **CPU note.** Set `amp: false` in the config when training on CPU.
- **Diagnostics.** `python -m mcaq_yolo.scripts.backend_agreement --data
  data.yaml` reports per-metric surrogate/reference correlations (use
  `--legacy` to reproduce the pre-fix gap). After training, call
  `model.complexity_analyzer.fit_feature_weights(loader)` to refit Eq.8's
  α to the trained MLP before recomputing curriculum scores.

## v0.2.1 — External review fixes (2026-07)

- **Best-checkpoint selection (behavior change, was broken).** `best.pt` was
  chosen by minimum `val_loss` across quantization regimes, so the
  un-quantized Stage-1 warm-up loss pinned `best.pt` to an FP checkpoint for
  the rest of training. Best is now the highest **quantized mAP@0.5 from
  Stage 3 onward** (the AP implementation in `utils/evaluation.py` is wired
  into the training loop); `last.pt` is saved every epoch, and runs that end
  before Stage 3 fall back to the final model with an explicit notice.
- **LICENSE added** (MIT — the README previously claimed MIT without a
  license file, which meant no license was actually granted).
- **CUDA kernel status made explicit.** The kernel has never been compiled or
  executed by the authorsʼ CPU-only environment; a kernel-vs-PyTorch parity
  test now exists (`test_cuda_kernel_parity`) and runs automatically on
  machines with CUDA + the built extension. Until someone runs it on a GPU,
  treat the kernel as inspection-verified only.
- **Dead code removed / isolated.** The never-used tile cache (paper Table X
  claims a 10k-entry cache — **not implemented**, now stated as a documented
  deviation), `ComplexityBasedSampler`, `ProgressiveQuantizationScheduler`,
  `AdaptiveLearningRateScheduler` and the write-only curriculum history lists
  are gone. `LearnedRoundingQuantization` (α untrainable in the current
  pipeline) and MSE calibration (offline-only cost) are documented as
  experimental. Entropy calibration now uses `torch.histc` (the previous
  `torch.histogram` is CPU-only and crashed on CUDA tensors).
- **Feature-domain complexity documented.** Morphological metrics run on the
  channel-mean of the C3/C4/C5 feature maps every forward — a different
  operator from the paper's image-domain, calibration-time analysis; this
  repo does not reproduce the paper's 0.3 ms / 151 FPS latency path
  (stated in the hook and here).
- **Hygiene.** `Trainer` builds `MCAQYOLO` with an explicit constructor call
  (no more `inspect.signature` probing); migrated to `torch.amp` (AMP now
  auto-disables off-CUDA — the old CPU `amp: false` advice is automatic);
  `MCQLYOLOLoss` typo renamed to `MCAQYOLOLoss` (old name kept as an alias);
  import-time prints are `warnings.warn`; internal review-artifact comment
  references reworded to be self-contained; line endings normalized.
- **Reproducibility status (unchanged, stated plainly):** no trained
  weights, no paper-table configs/results are distributed yet, and
  dependencies are lower-bounded only. Verified environment for the test
  suite and smoke runs: Python 3.13, torch 2.12, ultralytics 8.4.63 (CPU).

## v0.2.2 — CI, lock file, evaluation cadence (2026-07)

- **CI added.** GitHub Actions runs the smoke test suite on every push/PR
  (CPU-only ubuntu runner; the CUDA parity test self-skips there).
- **`requirements-lock.txt` added** — exact pins of the verified
  environment (Python 3.13.13, torch 2.12.0, ultralytics 8.4.63, …) as a
  reference pin set for reproducing reported numbers. All comments and
  messages are now English-only.
- **`training.map_interval` config key** — compute the full-validation
  mAP@0.5 every N epochs instead of every epoch (best-checkpoint selection
  updates only on epochs where mAP is computed); the stale "placeholder"
  wording in `Trainer.evaluate()`'s docstring was corrected to describe the
  actual behavior.
