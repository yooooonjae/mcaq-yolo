"""
MCAQ-YOLO Main Model Implementation (Fixed Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import DEFAULT_CFG
from typing import Dict, Tuple, Optional, List
import warnings
import copy

from ..core.morphology import MorphologicalComplexityAnalyzer
from ..core.bit_allocation import (ComplexityToBitMappingNetwork, LinearBitMapper,
                                   interpolate_precision, apply_allocation_policy, feature_std_complexity)
from ..core.quantization import SpatialAdaptiveQuantization


def _extract_raw_maps(outputs) -> List[torch.Tensor]:
    """
    Normalize YOLO outputs to the list of raw Detect-head maps.

    - train mode: DetectionModel returns the raw list directly
    - eval mode: Detect returns (y_cat, raw_list)
    """
    if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
        return _extract_raw_maps(outputs[1])
    if isinstance(outputs, dict):
        return [outputs[key] for key in ('boxes', 'scores') if torch.is_tensor(outputs.get(key))]
    if isinstance(outputs, (list, tuple)):
        if (
            len(outputs) == 2
            and isinstance(outputs[1], (list, tuple))
            and all(torch.is_tensor(t) for t in outputs[1])
        ):
            return list(outputs[1])
        return [t for t in outputs if torch.is_tensor(t)]
    return [outputs] if torch.is_tensor(outputs) else []


def kd_logit_loss(student_outputs, teacher_outputs) -> Optional[torch.Tensor]:
    """
    Knowledge distillation, logit-level matching (paper Sec IV-E: LKD aligns the
    quantized model with an FP32 teacher using logit-level and feature-level
    matching). MSE over the matched Detect-head raw maps; scales whose shapes
    differ (e.g. different nc) are skipped defensively.
    """
    s_list = _extract_raw_maps(student_outputs)
    t_list = _extract_raw_maps(teacher_outputs)

    losses = []
    for s, t in zip(s_list, t_list):
        if s.shape == t.shape:
            losses.append(F.mse_loss(s.float(), t.detach().float()))

    if not losses:
        return None
    return sum(losses) / len(losses)


class MCAQYOLOLoss(nn.Module):
    """
    Combined loss function for MCAQ-YOLO with proper YOLO detection loss.

    (REVIEW FIX: renamed from the original typo `MCQLYOLOLoss`; the old name
    is kept as a module-level alias for backward compatibility.)
    """
    
    def __init__(
        self,
        model,  # Add model parameter for YOLO loss
        target_bits: float = 4.0,
        lambda_bit: float = 0.01,    # Paper Table X: lambda1 (0.01 -> 0.1 annealed)
        lambda_smooth: float = 0.1,  # Paper Table X: lambda2 = 0.1
        lambda_kd: float = 0.5,      # Paper Table X: lambda3 = 0.5
        lambda_reg: float = 1e-4     # Paper Table X: lambda4 = 1e-4
    ):
        super().__init__()
        self.target_bits = target_bits
        self.lambda_bit = lambda_bit
        self.lambda_smooth = lambda_smooth
        self.lambda_kd = lambda_kd
        self.lambda_reg = lambda_reg
        
        # Initialize YOLO detection loss
        self.detection_loss = v8DetectionLoss(model)
        
    def compute_smoothness_loss(self, bit_map) -> torch.Tensor:
        """
        Spatial smoothness loss (paper Sec IV-E):
            Lsmooth = sum_{i,j} |b_ij - b_{i+1,j}| + |b_ij - b_{i,j+1}|
        Summed over tiles per the paper, averaged over the batch. Accepts a single
        (B,Ht,Wt) map or a list of per-scale maps (averaged over scales).
        """
        if isinstance(bit_map, (list, tuple)):
            losses = [self.compute_smoothness_loss(m) for m in bit_map]
            return sum(losses) / max(1, len(losses))

        if bit_map.dim() == 2:
            bit_map = bit_map.unsqueeze(0)

        # Per-edge mean total variation (defensible extension of the paper's
        # raw sum: the sum scales with tile count, so finer grids would be
        # penalized ~7x harder than the paper's 8x8 reference; the per-edge
        # mean keeps lambda2 comparable across grid sizes).
        dx = torch.abs(bit_map[:, 1:, :] - bit_map[:, :-1, :])
        dy = torch.abs(bit_map[:, :, 1:] - bit_map[:, :, :-1])
        n_edges = dx.numel() + dy.numel()
        return (dx.sum() + dy.sum()) / max(1, n_edges)

    def compute_bit_budget_loss(
        self,
        avg_bits: torch.Tensor,
        target_bits: Optional[float] = None
    ) -> torch.Tensor:
        """Paper Sec IV-E: Lbit = (b_bar - b_target)^2."""
        if target_bits is None:
            target_bits = self.target_bits

        return (avg_bits - target_bits) ** 2
    
    def forward(
        self,
        outputs: torch.Tensor,
        batch: Dict,  # Changed from targets to batch dict
        aux_info: Dict,
        teacher_outputs: Optional[torch.Tensor] = None,
        model_params: Optional[nn.Module] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        target_bits: Optional[float] = None  # curriculum get_target_bits(t): 8 -> 4
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute total loss with proper YOLO detection loss."""
        
        if loss_weights is None:
            loss_weights = {
                'detection': 1.0,
                'bit_budget': self.lambda_bit,
                'smoothness': self.lambda_smooth,
                'distillation': self.lambda_kd,
                'regularization': self.lambda_reg
            }
        
        loss_dict = {}
        
        # YOLO Detection loss - use the actual YOLO loss
        if hasattr(self, 'detection_loss'):
            loss_tensor, loss_items = self.detection_loss(outputs, batch)
            # loss_tensor is shape (3,) containing [box_loss, cls_loss, dfl_loss]
            # Sum to get scalar loss
            loss_det = loss_tensor.sum()
            loss_dict['loss_det'] = loss_det
            # Add individual loss components
            if isinstance(loss_items, dict):
                loss_dict.update({key: float(value) for key, value in loss_items.items()})
            else:
                loss_dict.update({name: float(loss_items[i]) for i, name in
                                  enumerate(('box_loss', 'cls_loss', 'dfl_loss'))})
        else:
            # Fallback to simple MSE
            targets = batch.get('labels', batch)
            loss_det = F.mse_loss(outputs, targets)
            loss_dict['loss_det'] = loss_det
        
        # Get device from loss_det (already computed tensor)
        device = loss_det.device

        # Bit budget loss
        if 'avg_bits' in aux_info:
            loss_bit = self.compute_bit_budget_loss(aux_info['avg_bits'], target_bits)
            loss_dict['loss_bit'] = loss_bit
        else:
            loss_bit = torch.tensor(0.0, device=device)

        # Smoothness loss
        if 'bit_map' in aux_info:
            loss_smooth = self.compute_smoothness_loss(aux_info['bit_map'])
            loss_dict['loss_smooth'] = loss_smooth
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        # Knowledge distillation loss (paper Sec IV-E: logit-level matching against
        # the FP32 teacher; feature-level matching is added via aux_info when the
        # backbone features are available)
        loss_kd = torch.tensor(0.0, device=device)
        if teacher_outputs is not None:
            kd = kd_logit_loss(outputs, teacher_outputs)
            if kd is not None:
                loss_kd = kd
        if 'kd_feature_loss' in aux_info and torch.is_tensor(aux_info['kd_feature_loss']):
            loss_kd = loss_kd + aux_info['kd_feature_loss']
        loss_dict['loss_kd'] = loss_kd

        # Regularization loss — paper Sec IV-E: L2 penalty on the *mapping network
        # weights* only (the complexity->bit MLP of Sec IV-C), lambda4 = 1e-4.
        # Weight matrices only (dim > 1) — biases and BN affine terms are not
        # "weights" in the paper's sense.
        if model_params is not None:
            loss_reg = torch.tensor(0.0, device=device)
            for param in model_params.parameters():
                if param.requires_grad and param.dim() > 1:
                    loss_reg = loss_reg + param.pow(2).sum()
            loss_dict['loss_reg'] = loss_reg
        else:
            loss_reg = torch.tensor(0.0, device=device)
        
        # Weighted sum of losses
        total_loss = (
            loss_weights['detection'] * loss_det +
            loss_weights['bit_budget'] * loss_bit +
            loss_weights['smoothness'] * loss_smooth +
            loss_weights['distillation'] * loss_kd +
            loss_weights['regularization'] * loss_reg
        )
        
        loss_dict['loss_total'] = total_loss

        return total_loss, loss_dict


# Backward-compatibility alias for the original (typo'd) class name
MCQLYOLOLoss = MCAQYOLOLoss


class MCAQYOLO(nn.Module):
    """
    MCAQ-YOLO: Fixed version with proper YOLO integration
    """

    def __init__(
        self,
        model_name: str = 'yolov8n',
        pretrained: bool = True,
        min_bits: int = 2,
        max_bits: int = 8,
        target_bits: float = 4.0,
        device: str = 'cuda',
        num_classes: int = 80,  # Added num_classes parameter
        grid_size: Optional[int] = None,  # Explicit scalar retains the legacy uniform grid.
        bit_mapping: str = 'mlp',  # 'mlp' (Eq.13-17) | 'linear' (paper's Linear-mapping ablation)
        normalize_complexity: bool = False,
        allocation_policy: str = 'adaptive',
        smooth_transitions: bool = True,
        control_seed: int = 0,
        grid_sizes: Optional[List[int]] = None,  # C3/C4/C5 defaults: 8/4/2.
    ):
        super().__init__()

        if grid_size is not None and grid_sizes is not None:
            raise ValueError('specify grid_sizes or legacy grid_size, not both')
        if grid_size is not None and (type(grid_size) is not int or grid_size < 1):
            raise ValueError('grid_size must be a positive integer')
        if grid_sizes is not None and (
                not isinstance(grid_sizes, (list, tuple)) or not grid_sizes
                or any(type(grid) is not int or grid < 1 for grid in grid_sizes)):
            raise ValueError('grid_sizes must contain positive integers')
        if allocation_policy not in ('adaptive', 'uniform', 'shuffle', 'layerwise', 'std'):
            raise ValueError('unknown allocation_policy')
        if not 2 <= min_bits <= target_bits <= max_bits <= 8:
            raise ValueError('require 2 <= min_bits <= target_bits <= max_bits <= 8')
        self.allocation_policy = allocation_policy
        self.smooth_transitions = bool(smooth_transitions)
        self.control_seed = int(control_seed)
        # Load base YOLOv8 model
        # Note: Use temporary variable to avoid registering YOLO wrapper as submodule
        if pretrained:
            _yolo_wrapper = YOLO(model_name if str(model_name).endswith('.pt') else f'{model_name}.pt')
        else:
            _yolo_wrapper = YOLO(model_name)

        # Extract the actual nn.Module model from YOLO wrapper
        self.model = _yolo_wrapper.model.to(device)
        self.num_classes = num_classes

        # IMPORTANT: Unfreeze all YOLO model parameters for training
        # By default, Ultralytics YOLO loads with frozen parameters
        for param in self.model.parameters():
            param.requires_grad = True

        # Ensure model.args is a SimpleNamespace (required by v8DetectionLoss)
        # v8DetectionLoss expects model.args to have attributes like 'box', 'cls', 'dfl'
        if not hasattr(self.model, 'args') or self.model.args is None:
            # Create new SimpleNamespace from DEFAULT_CFG dict
            self.model.args = SimpleNamespace(**vars(DEFAULT_CFG))
        elif isinstance(self.model.args, dict):
            # Merge DEFAULT_CFG into model.args to ensure all required keys exist
            merged_args = dict(vars(DEFAULT_CFG))  # Start with DEFAULT_CFG
            merged_args.update(self.model.args)     # Override with model's args
            self.model.args = SimpleNamespace(**merged_args)
        else:
            # Ensure it has required loss attributes from DEFAULT_CFG
            for key in ['box', 'cls', 'dfl', 'pose', 'kobj']:
                if not hasattr(self.model.args, key):
                    setattr(self.model.args, key, getattr(DEFAULT_CFG, key, 1.0))

        # Store YOLO wrapper reference without registering as nn.Module attribute
        # Use object.__setattr__ to bypass nn.Module's attribute registration
        object.__setattr__(self, '_yolo_wrapper', _yolo_wrapper)
        
        # One shared analyzer: the base grid is for input-image sample scoring;
        # hooks pass their level's grid explicitly without changing module state.
        # NOTE (documented deviation): the paper's Table X tile cache is not
        # implemented — see MorphologicalComplexityAnalyzer.__init__.
        self.complexity_analyzer = MorphologicalComplexityAnalyzer(
            grid_size=grid_size if grid_size is not None else 8,
            device=device
        )
        
        if bit_mapping == 'linear':
            # Paper Table V/VIII 'Linear mapping' ablation — parameter-free,
            # exposes the complexity map's relative spatial structure directly
            # (practical at small training scales; see LinearBitMapper docs).
            self.bit_mapper = LinearBitMapper(min_bits=min_bits, max_bits=max_bits).to(device)
        else:
            self.bit_mapper = ComplexityToBitMappingNetwork(
                min_bits=min_bits,
                max_bits=max_bits,
                hidden_dims=[32, 64, 32]  # Paper Table X
            ).to(device)
        
        # Defensible extension (paper leaves C's absolute calibration unspecified;
        # Table IV implies a wide C range 0.21-0.72): per-image percentile
        # normalization of C before bit mapping. Mechanism: normalization is
        # invariant to global shifts of C, so the Lbit budget pressure can no
        # longer be satisfied by collapsing C globally (the measured flat-map
        # attractor) — it must reshape the mapper, and only Ldet's tile-wise
        # signal moves relative structure.
        self.normalize_complexity = normalize_complexity

        # Paper Sec IV-D / Table X: per-channel min/max calibration with EMA 0.99.
        # NOTE: one quantizer PER backbone scale — P3/P4/P5 have different channel
        # counts, so sharing a single per-channel EMA buffer across scales would
        # produce a shape mismatch on the second hook.
        # The ModuleDict is populated in _register_mcaq_hooks() once the backbone
        # output indices are known.
        self.quantizers = nn.ModuleDict()
        
        # Initialize loss function with model
        self.loss_fn = MCAQYOLOLoss(
            model=self.model,
            target_bits=target_bits,
            lambda_bit=0.01,    # Paper Table X: lambda1 (annealed 0.01 -> 0.1 via curriculum)
            lambda_smooth=0.1,  # Paper Table X: lambda2 = 0.1
            lambda_kd=0.5,      # Paper Table X: lambda3 = 0.5
            lambda_reg=1e-4     # Paper Table X: lambda4 = 1e-4
        )

        # NOTE: the curriculum schedule lives in the Trainer (single source of
        # truth, paper Table X) — a per-model scheduler here would shadow-update
        # independently of the one actually driving training.

        # Training parameters
        self.device = device
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.target_bits = target_bits

        # Store original model for teacher
        self.teacher_model = None

        # ------------------------------------------------------------------
        # Paper Sec IV-D: "The spatial quantization is applied at the output of
        # the backbone (C3/C4/C5 feature maps) before the FPN neck."
        # Locate the backbone output layers consumed by the neck and register
        # forward hooks that quantize those feature maps in-graph.
        # ------------------------------------------------------------------
        self._mcaq_state: Dict = {'active': False}
        self.backbone_out_indices = self._find_backbone_out_indices()
        self.feature_grid_sizes = ((grid_size,) * len(self.backbone_out_indices)
                                   if grid_size is not None else tuple(grid_sizes or (8, 4, 2)))
        if len(self.feature_grid_sizes) != len(self.backbone_out_indices):
            raise ValueError('grid_sizes must contain one grid per discovered backbone feature level')
        self.feature_grid_by_layer = dict(zip(self.backbone_out_indices, self.feature_grid_sizes))
        self._register_mcaq_hooks()
        
    # ------------------------------------------------------------------
    # Backbone C3/C4/C5 discovery + MCAQ quantization hooks (paper Sec IV-D)
    # ------------------------------------------------------------------
    def _find_backbone_out_indices(self) -> List[int]:
        """
        Locate the indices of the backbone layers whose outputs (P3/P4/P5 —
        the C3/C4/C5 feature maps) feed the FPN neck.

        Primary logic (dynamic): the backbone ends at SPPF;
        every neck layer's `.f` (from-index) that points back into the backbone
        identifies a skip connection (P3/P4). P5 is the SPPF output itself.
        Fallback: [4, 6, 9] (standard YOLOv8 n/s/m/l/x topology).
        """
        fallback = [4, 6, 9]
        try:
            layers = list(self.model.model)

            sppf_idx = None
            for i, m in enumerate(layers):
                if m.__class__.__name__ == 'SPPF':
                    sppf_idx = i
            if sppf_idx is None:
                warnings.warn(
                    "[MCAQ] SPPF layer not found — falling back to backbone "
                    f"indices {fallback}."
                )
                return [i for i in fallback if i < len(layers)]

            refs = set()
            for m in layers[sppf_idx + 1:]:
                f = getattr(m, 'f', None)
                if f is None:
                    continue
                f_list = f if isinstance(f, (list, tuple)) else [f]
                for j in f_list:
                    # strictly BELOW SPPF: the neck's P5-route Concat references
                    # the SPPF index itself (f=[-1, sppf]), which must not be
                    # double-counted (a '<=' comparison here yielded [4,6,9,9]
                    # and two hooks on layer 9)
                    if isinstance(j, int) and j != -1 and 0 <= j < sppf_idx:
                        refs.add(j)

            indices = sorted(refs | {sppf_idx})
            if len(indices) < 2:
                warnings.warn(
                    "[MCAQ] Could not derive backbone->neck connections — "
                    f"falling back to {fallback}."
                )
                return [i for i in fallback if i < len(layers)]
            return indices
        except Exception as e:  # pragma: no cover — defensive (cannot run here)
            warnings.warn(f"[MCAQ] Backbone discovery failed ({e}) — using {fallback}.")
            return fallback

    def _make_mcaq_hook(self, layer_idx: int):
        """
        Forward hook applying tile-wise mixed-precision quantization (Eq.19) to a
        backbone output. Returning a tensor from a forward hook replaces the
        layer's output, so the FPN neck consumes the quantized features and
        gradients flow through the STE quantizer into the backbone (QAT).
        """
        def hook(module, inputs, output):
            state = self._mcaq_state
            if not state.get('active', False):
                return None  # pass-through outside MCAQ forward
            if not torch.is_tensor(output) or output.dim() != 4:
                return None

            feat = output
            grid_size = self.feature_grid_by_layer[layer_idx]
            detailed = {}
            precomputed = state.get('precomputed_bit_maps')
            if precomputed is not None:
                complexity = None  # deliberately not recomputed on this path
                bit_map = precomputed[layer_idx].to(device=feat.device, dtype=torch.float32)
                expected_grid = tuple(min(grid_size, size) for size in feat.shape[-2:])
                if tuple(bit_map.shape[-2:]) != expected_grid:
                    raise ValueError(f'precomputed bit_map grid for layer {layer_idx} must be {expected_grid}')
            else:
                if self.allocation_policy == 'std':
                    complexity = feature_std_complexity(feat, grid_size)
                else:
                    complexity, detailed = self.complexity_analyzer(
                        feat, return_detailed=True, grid_size=grid_size)
                if self.normalize_complexity:
                    B = complexity.shape[0]
                    flat = complexity.reshape(B, -1)
                    lo = torch.quantile(flat, 0.02, dim=1, keepdim=True).unsqueeze(-1)
                    hi = torch.quantile(flat, 0.98, dim=1, keepdim=True).unsqueeze(-1)
                    complexity = ((complexity - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
                bit_map = self.bit_mapper(
                    complexity, state.get('temperature', 1.0),
                    return_continuous=True,
                )
                bit_map = apply_allocation_policy(
                    bit_map, feat.shape[-2:], 'adaptive' if self.allocation_policy == 'std' else self.allocation_policy,
                    uniform_bits=self.target_bits, seed=self.control_seed + layer_idx,
                )
                progress = state.get('precision_progress')
                if progress is not None:
                    bit_map = interpolate_precision(bit_map, progress, self.max_bits)
                if not self.training:
                    bit_map = bit_map.round()

            quantize = state.get('quantize', True)
            quantizer = self.quantizers[str(layer_idx)]
            q_training = self.training or state.get('calibrating', False)
            feat_q = quantizer(feat, bit_map, training=q_training) if quantize else feat

            state['aux'].append({
                'layer': layer_idx,
                'complexity': complexity,
                'bit_map': bit_map,
                'features_q': feat_q,
                'descriptors': detailed,
            })
            return feat_q if quantize else None

        return hook

    def _register_mcaq_hooks(self):
        """Register the MCAQ quantization hooks on the backbone output layers."""
        layers = list(self.model.model)
        self._mcaq_hooks = []
        for idx in self.backbone_out_indices:
            if 0 <= idx < len(layers):
                # Independent quantizer per scale (distinct per-channel EMA stats)
                self.quantizers[str(idx)] = SpatialAdaptiveQuantization(
                    calibration_mode='minmax',
                    smooth_transitions=self.smooth_transitions,
                    per_channel=True,
                ).to(self.device)
                self._mcaq_hooks.append(
                    layers[idx].register_forward_hook(self._make_mcaq_hook(idx))
                )

    @torch.no_grad()
    def paired_shuffle_forward(self, x):
        """Paired endpoint intervention with identical per-level bit budgets.

        Compute every reference map once, then replay their shuffled versions.
        Later feature changes cannot silently change downstream map histograms.
        """
        if self.training or self.allocation_policy != 'adaptive':
            raise ValueError('paired shuffle requires an adaptive model in eval mode')
        reference = self(x, precision_progress=1.)
        aux = reference[1]
        maps = [apply_allocation_policy(bits, feat.shape[-2:], 'shuffle',
                                       seed=self.control_seed + layer)
                for layer, bits, feat in zip(aux['feature_layers'], aux['bit_map'], aux['quantized_features'])]
        shuffled = self(x, precomputed_bit_maps=maps, precision_progress=1.)
        return reference, shuffled

    @torch.no_grad()
    def precompute_bit_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute detached per-scale maps for this exact preprocessed input.

        Call after calibration in eval mode. The maps are valid for the same
        input, model weights, calibration and preprocessing only. There is no
        implicit cache or reuse on novel images.
        """
        if self.training:
            raise ValueError("precomputed bit maps are an inference-only path; call eval()")
        _, aux = self(x, temperature=1.0)
        return [bits.detach().clone() for bits in aux['bit_map']]

    @torch.no_grad()
    def calibrate(self, dataloader, num_images: int = 1000):
        """Refresh EMA statistics on at most num_images, then freeze them.

        Repeated calls start a new calibration run, including when an earlier
        run froze the buffers. The caller's train/eval mode is restored.
        """
        if not isinstance(num_images, int) or num_images <= 0:
            raise ValueError("num_images must be a positive integer")
        training_modes = {module: module.training for module in self.modules()}
        self.eval()
        for q in self.quantizers.values():
            q.reset_calibration()
        seen = 0
        try:
            for batch in dataloader:
                if torch.is_tensor(batch):
                    imgs = batch
                else:
                    imgs = batch['img'] if isinstance(batch, dict) else batch[0]
                imgs = imgs[:num_images - seen].to(next(self.parameters()).device).float()
                if imgs.numel() == 0:
                    continue
                if imgs.max() > 1.5:
                    imgs = imgs / 255.0
                self._mcaq_state = {
                    'active': True, 'temperature': 1.0, 'quantize': True,
                    'calibrating': True, 'aux': [],
                }
                self.model(imgs)
                seen += imgs.shape[0]
                if seen >= num_images:
                    break
            if seen == 0:
                raise ValueError("cannot calibrate on an empty dataloader")
            for q in self.quantizers.values():
                q.freeze_calibration()
        finally:
            self._mcaq_state = {'active': False}
            # Preserve mixed modes, e.g. an eval-only teacher or frozen BN.
            for module, mode in training_modes.items():
                module.training = mode
        return seen

    def setup_teacher(self):
        """Setup teacher model for distillation."""
        self.teacher_model = copy.deepcopy(self.model)
        # The copy inherits the MCAQ forward hooks — strip them so the teacher
        # always runs full precision
        for m in self.teacher_model.modules():
            m._forward_hooks.clear()
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    # NOTE: the legacy pre-Tier3 complexity path (cv1/cv2/cv3 feature hooks +
    # a separate no-grad forward) was removed — complexity is now computed
    # inside the MCAQ quantization hooks on the actual C3/C4/C5 outputs in a
    # single forward pass (Sec IV-D). Use forward(..., return_aux=True) to
    # obtain per-scale complexity maps.

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        return_aux: bool = True,
        targets: Optional[Dict] = None,
        quantize: bool = True,
        precomputed_bit_maps: Optional[List[torch.Tensor]] = None,
        precision_progress: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with adaptive quantization.

        Paper Sec IV-D: spatial quantization is applied to the backbone C3/C4/C5
        feature maps before the FPN neck — realized via the forward hooks on
        self.backbone_out_indices in a single forward pass (each scale gets its
        own complexity map and bit map).

        Args:
            x: Input images (B, 3, H, W)
            temperature: Legacy multiplicative schedule; use 1 with precision_progress.
            precision_progress: Convex continuation from max_bits to the base policy;
                0 for warm-up, 1 for the final policy. None retains legacy API behavior.
            return_aux: Whether to return auxiliary information
            targets: Target labels for training
            precomputed_bit_maps: Optional integer maps in backbone_out_indices order.
                Inference only; must correspond to this exact input/model/calibration.
                Skips morphology and mapping; complexity_map entries are then None.
            quantize: If False (curriculum Stage 1 warm-up, paper Fig.3), skip the
                quantization step (high precision) while still producing complexity
                and bit maps so Lbit/Lsmooth keep training the mapping networks.

        Returns:
            outputs: Detection outputs
            aux_info: Auxiliary information (if return_aux=True)
        """
        if precision_progress is not None:
            if not 0 <= precision_progress <= 1 or temperature != 1.0:
                raise ValueError('precision_progress requires [0,1] and temperature=1')
        prepared_maps = None
        if precomputed_bit_maps is not None:
            if (self.training or not quantize or temperature != 1.0
                    or precision_progress not in (None, 1.0)):
                raise ValueError("precomputed maps require quantized inference at temperature=1")
            if len(precomputed_bit_maps) != len(self.backbone_out_indices):
                raise ValueError("precomputed_bit_maps must contain one map per backbone scale")
            prepared_maps = {}
            for layer, bits in zip(self.backbone_out_indices, precomputed_bit_maps):
                if (not torch.is_tensor(bits) or bits.dim() != 3
                        or bits.shape[0] != x.shape[0] or any(n == 0 for n in bits.shape)):
                    raise ValueError("each precomputed bit_map must be nonempty (B,Ht,Wt)")
                if not bool(torch.isfinite(bits).all() & (bits >= 2).all()
                            & (bits <= 8).all() & (bits == bits.round()).all()):
                    raise ValueError("precomputed bit_map values must be integers in [2,8]")
                prepared_maps[layer] = bits.detach()
        self._mcaq_state = {
            'active': True,
            'temperature': temperature,
            'precision_progress': precision_progress,
            'quantize': quantize,
            'aux': [],
            'precomputed_bit_maps': prepared_maps,
        }
        try:
            outputs = self.model(x)
        finally:
            aux_records = self._mcaq_state.get('aux', [])
            self._mcaq_state = {'active': False}

        if return_aux:
            bit_maps = [r['bit_map'] for r in aux_records]
            complexity_maps = [r['complexity'] for r in aux_records]

            if bit_maps:
                # The policy budget gives each feature level equal weight,
                # regardless of its tile count. Element-weighted bits are a
                # separate diagnostic, never substituted into this objective.
                avg_bits = torch.stack([m.float().mean() for m in bit_maps]).mean()
            else:
                avg_bits = torch.tensor(float(self.target_bits), device=x.device)

            aux_info = {
                'complexity_map': complexity_maps,
                'bit_map': bit_maps,
                'avg_bits': avg_bits,
                'quantized_features': [r['features_q'] for r in aux_records],
                'feature_layers': [r['layer'] for r in aux_records],
                'detailed_metrics': [r['descriptors'] for r in aux_records],
            }
            return outputs, aux_info

        return outputs
