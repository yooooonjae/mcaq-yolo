"""
Spatial Adaptive Quantization Module
Complete implementation for MCAQ-YOLO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings

# CUDA Extension 로드 시도
try:
    import mcaq_cuda_ops
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("[MCAQ-YOLO] Warning: 'mcaq_cuda_ops' not found. Falling back to slow PyTorch implementation.")


class QuantizationParameters:
    """Container for quantization parameters."""
    
    def __init__(self, bits: int):
        """
        Initialize quantization parameters.
        
        Args:
            bits: Number of bits for quantization
        """
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.levels = 2 ** bits
        
    def compute_scale_zeropoint(
        self,
        x_min: torch.Tensor,
        x_max: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scale and zero-point for quantization.
        
        Args:
            x_min: Minimum value
            x_max: Maximum value
            
        Returns:
            scale, zero_point for quantization
        """
        # Avoid division by zero
        x_range = x_max - x_min
        x_range = torch.clamp(x_range, min=1e-8)
        
        scale = x_range / (self.qmax - self.qmin)
        zero_point = self.qmin - x_min / scale
        
        # Clamp zero_point to valid range
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        
        return scale, zero_point


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for quantization in training."""
    
    @staticmethod
    def forward(ctx, input, scale, zero_point, qmin, qmax):
        """
        Forward pass with quantization.
        
        Args:
            input: Input tensor
            scale: Quantization scale
            zero_point: Quantization zero point
            qmin: Minimum quantized value
            qmax: Maximum quantized value
            
        Returns:
            Quantized and dequantized tensor
        """
        # Quantize
        output = torch.round(input / scale + zero_point)
        output = torch.clamp(output, qmin, qmax)
        
        # Dequantize
        output = (output - zero_point) * scale
        
        # Save for backward
        ctx.save_for_backward(input, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with straight-through estimator.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradients for input and None for other parameters
        """
        input, scale = ctx.saved_tensors
        
        # Straight-through estimator: pass gradient as-is
        # Optional: apply gradient clipping based on quantization range
        grad_input = grad_output.clone()
        
        return grad_input, None, None, None, None


class LearnedRoundingQuantization(nn.Module):
    """Learned rounding for better quantization."""
    
    def __init__(self, num_channels: Optional[int] = None):
        """
        Initialize learned rounding.
        
        Args:
            num_channels: Number of channels for per-channel rounding
        """
        super().__init__()
        
        if num_channels is not None:
            # Per-channel learned rounding
            self.alpha = nn.Parameter(torch.zeros(num_channels, 1, 1))
        else:
            # Global learned rounding
            self.alpha = nn.Parameter(torch.zeros(1))
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned rounding.
        
        Args:
            x: Input tensor
            
        Returns:
            Rounded tensor with learned adjustment
        """
        # Learned rounding: interpolate between floor and ceil
        alpha = self.sigmoid(self.alpha)
        x_floor = torch.floor(x)
        x_ceil = torch.ceil(x)
        
        return x_floor + alpha * (x_ceil - x_floor)


class LearnedSoftMask(nn.Module):
    """
    Paper Eq.(19) m(p): "a learned soft mask that varies smoothly over space ...
    produced by a softmax-based module followed by spatial smoothing".

    The paper does not specify the module's architecture, so this is a minimal
    literal realization (documented implementation choice):
      - input: per-tile statistics (allocated bits, mean activation magnitude)
        — channel-agnostic so one module serves any feature width
      - a small conv head produces 2 logits; a channel-wise softmax bounds
        m to [0,1] (the "softmax-based" part)
      - nearest upsampling keeps the single-tile assignment per position;
        Gaussian spatial smoothing then makes m(p) change gradually across
        tile boundaries (the "spatial smoothing" part)
      - initialized so m(p) ~= 0.98 (near-identity) at the start of training
    """

    def __init__(self, hidden: int = 8, kernel_size: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),
        )
        # Near-identity init: the last layer uses NEAR-zero (std=1e-3, not
        # exactly zero) weights so the bias logit gap of 4 dominates
        # (softmax ~= 0.982 ~ identity) while the gradient path through
        # W2^T @ grad stays alive from the very first backward pass — an
        # exactly-zero W2 would give net[0] a zero gradient on step 1
        # (final verification finding). The first conv keeps its default init.
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([4.0, 0.0]))

        # Gaussian smoothing kernel for the spatial-smoothing step
        k = kernel_size
        sigma = k / 3.0
        x = torch.arange(k, dtype=torch.float32) - k // 2
        g1 = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g1 = g1 / g1.sum()
        g2 = (g1.unsqueeze(0) * g1.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('smooth_kernel', g2)
        self.kernel_size = k

    def forward(
        self,
        bit_map: torch.Tensor,   # (B, Ht, Wt) — continuous during training
        x: torch.Tensor,         # (B, C, H, W) — activations being quantized
    ) -> torch.Tensor:
        """Returns m(p) of shape (B, 1, H, W) in [0, 1]."""
        B, C, H, W = x.shape
        Ht, Wt = bit_map.shape[-2:]

        # Per-tile mean activation magnitude (side information, no grad to x)
        with torch.no_grad():
            act = F.adaptive_avg_pool2d(x.detach().abs().mean(1, keepdim=True), (Ht, Wt))
            act = act / (act.amax(dim=(2, 3), keepdim=True) + 1e-8)

        bits_norm = ((bit_map.unsqueeze(1).float() - 2.0) / 6.0).clamp(0.0, 1.0)
        feats = torch.cat([bits_norm, act.float()], dim=1)  # (B,2,Ht,Wt)

        logits = self.net(feats)
        m = torch.softmax(logits, dim=1)[:, :1]  # softmax-based bounding to [0,1]

        # Single-tile assignment per position, then spatial smoothing.
        # Replicate padding: implicit zero padding would decay the mask at
        # image borders (a constant m would not survive the border rows).
        m = F.interpolate(m, size=(H, W), mode='nearest')
        p = self.kernel_size // 2
        m = F.conv2d(F.pad(m, (p, p, p, p), mode='replicate'), self.smooth_kernel)
        return m


class SpatialAdaptiveQuantization(nn.Module):
    """
    Hardware-aware spatial adaptive quantization module.

    Implements tile-wise mixed-precision quantization (Eq.19):
        X_q(p) = m(p) * Q_{bT(p)}(X(p))
    with a single-tile assignment per spatial position and a learned,
    spatially-smoothed soft mask m(p).
    """
    
    def __init__(
        self,
        calibration_mode: str = 'minmax',  # 'minmax', 'percentile', 'entropy', 'mse'
        smooth_transitions: bool = True,
        per_channel: bool = True,
        learned_rounding: bool = False,
        momentum: float = 0.99
    ):
        """
        Initialize spatial adaptive quantization.

        Args:
            calibration_mode: Method for computing quantization parameters
            smooth_transitions: Whether to use smooth transitions between tiles
            per_channel: Whether to quantize per-channel or per-tensor
            learned_rounding: Whether to use learned rounding
            momentum: EMA decay for running statistics (paper Table X: EMA momentum
                      0.99, i.e. running <- 0.99*running + 0.01*new)
        """
        super().__init__()
        self.calibration_mode = calibration_mode
        self.smooth_transitions = smooth_transitions
        self.per_channel = per_channel
        self.momentum = momentum
        
        # Running statistics for calibration
        self.register_buffer('running_min', None)
        self.register_buffer('running_max', None)
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        # Paper Sec IV-D: stats are collected over 1,000 calibration images with
        # EMA(0.99), "then frozen to compute scale and zero-point per channel".
        self.register_buffer('stats_frozen', torch.tensor(False))
        
        # Learned rounding module
        self.learned_rounding = None
        if learned_rounding:
            self.learned_rounding = LearnedRoundingQuantization()
        
        # Eq.(19) learned soft mask m(p): softmax-based module + spatial smoothing
        self.soft_mask = LearnedSoftMask() if smooth_transitions else None
        
        # Calibration statistics
        self.register_buffer('calibration_histogram', None)
        self.histogram_bins = 2048
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Materialize lazily-created (None) stat buffers before loading so
        checkpointed EMA calibration statistics can be restored. Buffers
        registered as None are absent from a fresh module's state_dict, so a
        trained checkpoint's running_min/max would otherwise fail strict
        loading and be silently dropped under strict=False (observed when
        loading outputs/coco128_run30/best.pt).
        """
        for name in ('running_min', 'running_max'):
            key = prefix + name
            if key in state_dict and getattr(self, name) is None:
                setattr(self, name, torch.zeros_like(state_dict[key]))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def freeze_calibration(self):
        """Freeze calibration statistics (paper Sec IV-D: EMA over 1,000
        calibration images, then frozen for scale/zero-point computation)."""
        self.stats_frozen = torch.tensor(True, device=self.stats_frozen.device)

    def update_running_stats(self, x: torch.Tensor):
        """
        Update running statistics for calibration.

        Args:
            x: Input tensor
        """
        if bool(self.stats_frozen):
            return  # calibration frozen — stats fixed (paper Sec IV-D)
        with torch.no_grad():
            # Compute min/max
            if self.per_channel:
                # Per-channel statistics
                dims = [0] + list(range(2, x.dim()))
                x_min = x.amin(dim=dims, keepdim=True)
                x_max = x.amax(dim=dims, keepdim=True)
            else:
                # Per-tensor statistics
                x_min = x.min()
                x_max = x.max()
            
            # Update running statistics — EMA with decay = momentum (paper: 0.99):
            # running <- momentum * running + (1 - momentum) * new
            if self.running_min is None:
                self.running_min = x_min
                self.running_max = x_max
            else:
                self.running_min = self.momentum * self.running_min + (1 - self.momentum) * x_min
                self.running_max = self.momentum * self.running_max + (1 - self.momentum) * x_max
            
            self.num_batches_tracked += 1
            
            # Update histogram for entropy calibration
            if self.calibration_mode == 'entropy':
                self._update_calibration_histogram(x)
    
    def _update_calibration_histogram(self, x: torch.Tensor):
        """Update histogram for entropy-based calibration."""
        with torch.no_grad():
            x_flat = x.flatten()
            
            # Compute histogram
            hist, bin_edges = torch.histogram(
                x_flat,
                bins=self.histogram_bins,
                density=True
            )
            
            if self.calibration_histogram is None:
                self.calibration_histogram = hist
            else:
                # Exponential moving average (decay = momentum)
                self.calibration_histogram = (
                    self.momentum * self.calibration_histogram +
                    (1 - self.momentum) * hist
                )
    
    def get_calibration_params(
        self,
        x: torch.Tensor,
        bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get calibration parameters for quantization.
        
        Args:
            x: Input tensor
            bits: Bit-width for quantization
            
        Returns:
            scale, zero_point for quantization
        """
        if self.calibration_mode == 'minmax':
            scale, zero_point = self._calibrate_minmax(x, bits)
        elif self.calibration_mode == 'percentile':
            scale, zero_point = self._calibrate_percentile(x, bits)
        elif self.calibration_mode == 'entropy':
            scale, zero_point = self._calibrate_entropy(x, bits)
        elif self.calibration_mode == 'mse':
            scale, zero_point = self._calibrate_mse(x, bits)
        else:
            raise ValueError(f"Unknown calibration mode: {self.calibration_mode}")
        
        return scale, zero_point
    
    def _calibrate_minmax(
        self,
        x: torch.Tensor,
        bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Min-max calibration."""
        use_running = self.running_min is not None and (
            self.training or bool(self.stats_frozen)
        )
        if use_running:
            # Frozen (post-calibration) or training-time EMA statistics
            x_min = self.running_min
            x_max = self.running_max
        else:
            if self.per_channel:
                dims = [0] + list(range(2, x.dim()))
                x_min = x.amin(dim=dims, keepdim=True)
                x_max = x.amax(dim=dims, keepdim=True)
            else:
                x_min = x.min()
                x_max = x.max()
        
        qparams = QuantizationParameters(bits)
        scale, zero_point = qparams.compute_scale_zeropoint(x_min, x_max)
        
        return scale, zero_point
    
    def _calibrate_percentile(
        self,
        x: torch.Tensor,
        bits: int,
        percentile_min: float = 0.01,
        percentile_max: float = 99.99
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Percentile-based calibration for outlier robustness."""
        # 큰 텐서에서 quantile 계산 시 메모리 문제 방지를 위해 샘플링 사용
        max_samples = 100000  # 최대 샘플 수

        if self.per_channel:
            # Reshape for per-channel percentile
            x_reshaped = x.transpose(0, 1).flatten(1)  # (C, N)
            C, N = x_reshaped.shape

            if N > max_samples:
                # 랜덤 샘플링
                indices = torch.randperm(N, device=x.device)[:max_samples]
                x_sampled = x_reshaped[:, indices]
            else:
                x_sampled = x_reshaped

            # 샘플링된 데이터로 percentile 계산
            x_min = torch.quantile(x_sampled, percentile_min / 100, dim=1, keepdim=True)
            x_max = torch.quantile(x_sampled, percentile_max / 100, dim=1, keepdim=True)

            # Reshape back
            x_min = x_min.view(1, -1, 1, 1)
            x_max = x_max.view(1, -1, 1, 1)
        else:
            x_flat = x.flatten()
            if x_flat.numel() > max_samples:
                indices = torch.randperm(x_flat.numel(), device=x.device)[:max_samples]
                x_sampled = x_flat[indices]
            else:
                x_sampled = x_flat

            x_min = torch.quantile(x_sampled, percentile_min / 100)
            x_max = torch.quantile(x_sampled, percentile_max / 100)

        qparams = QuantizationParameters(bits)
        scale, zero_point = qparams.compute_scale_zeropoint(x_min, x_max)

        return scale, zero_point
    
    def _calibrate_entropy(
        self,
        x: torch.Tensor,
        bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Entropy-based calibration (KL divergence minimization)."""
        # Simplified entropy calibration
        # Full implementation would minimize KL divergence
        
        if self.calibration_histogram is not None:
            # Use histogram to find optimal range
            # This is a simplified version
            cumsum = torch.cumsum(self.calibration_histogram, dim=0)
            
            # Find range that preserves 99.9% of distribution
            threshold = 0.999
            idx_min = torch.searchsorted(cumsum, (1 - threshold) / 2)
            idx_max = torch.searchsorted(cumsum, threshold + (1 - threshold) / 2)
            
            # Map indices back to values
            x_abs_max = x.abs().max()
            x_min = -x_abs_max * (idx_min / self.histogram_bins)
            x_max = x_abs_max * (idx_max / self.histogram_bins)
        else:
            # Fallback to symmetric quantization
            x_abs_max = x.abs().max()
            x_min = -x_abs_max
            x_max = x_abs_max
        
        qparams = QuantizationParameters(bits)
        scale, zero_point = qparams.compute_scale_zeropoint(x_min, x_max)
        
        return scale, zero_point
    
    def _calibrate_mse(
        self,
        x: torch.Tensor,
        bits: int,
        num_candidates: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MSE-optimal calibration."""
        # Search for scale that minimizes reconstruction error
        x_min = x.min()
        x_max = x.max()
        
        best_scale = None
        best_zero_point = None
        min_error = float('inf')
        
        # Grid search over candidate ranges
        for alpha in torch.linspace(0.8, 1.0, num_candidates):
            candidate_min = x_min * alpha
            candidate_max = x_max * alpha
            
            qparams = QuantizationParameters(bits)
            scale, zero_point = qparams.compute_scale_zeropoint(
                candidate_min, candidate_max
            )
            
            # Quantize and compute error
            x_q = torch.round(x / scale + zero_point)
            x_q = torch.clamp(x_q, qparams.qmin, qparams.qmax)
            x_dq = (x_q - zero_point) * scale
            
            error = F.mse_loss(x, x_dq)
            
            if error < min_error:
                min_error = error
                best_scale = scale
                best_zero_point = zero_point
        
        return best_scale, best_zero_point
    
    def quantize_tensor(
        self,
        x: torch.Tensor,
        bits: int,
        training: bool = True
    ) -> torch.Tensor:
        """
        Quantize tensor with given bit-width.
        
        Args:
            x: Input tensor
            bits: Bit-width
            training: Whether in training mode
            
        Returns:
            Quantized tensor
        """
        # Get quantization parameters
        scale, zero_point = self.get_calibration_params(x, bits)
        
        # Create quantization parameters
        qparams = QuantizationParameters(bits)
        
        if training:
            # Use straight-through estimator for training
            x_quant = StraightThroughEstimator.apply(
                x, scale, zero_point, qparams.qmin, qparams.qmax
            )
        else:
            # Actual quantization for inference
            if self.learned_rounding and self.learned_rounding is not None:
                # Use learned rounding
                x_normalized = x / scale + zero_point
                x_q = self.learned_rounding(x_normalized)
            else:
                # Standard rounding
                x_q = torch.round(x / scale + zero_point)
            
            x_q = torch.clamp(x_q, qparams.qmin, qparams.qmax)
            x_quant = (x_q - zero_point) * scale
        
        return x_quant
    
    def forward(
        self,
        x: torch.Tensor,
        bit_map: torch.Tensor,
        training: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Perform spatial adaptive quantization.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            bit_map: Bit allocation map of shape (B, H_tiles, W_tiles)
            training: Whether in training mode (uses self.training if None)
            
        Returns:
            Spatially quantized tensor
        """
        if training is None:
            training = self.training
        
        # Update running statistics if training
        if training:
            self.update_running_stats(x)
            # Use PyTorch implementation for training (supports autograd)
            return self._forward_pytorch(x, bit_map, training)
        
        # For Inference: Use CUDA acceleration if available
        if HAS_CUDA and x.is_cuda:
            return self._forward_cuda(x, bit_map)
        else:
            return self._forward_pytorch(x, bit_map, training)

    def _forward_cuda(self, x: torch.Tensor, bit_map: torch.Tensor) -> torch.Tensor:
        """Optimized CUDA forward pass for inference."""
        B, C, H, W = x.shape
        _, H_tiles, W_tiles = bit_map.shape # bit_map assumed (B, Ht, Wt)

        tile_h = H // H_tiles
        tile_w = W // W_tiles

        # Paper Sec IV-D: after calibration the EMA statistics are frozen and
        # reused at inference; dynamic per-batch min/max is the fallback when
        # no calibration has been performed.
        if bool(self.stats_frozen) and self.running_min is not None:
            x_min = self.running_min.reshape(1, -1, 1, 1)
            x_max = self.running_max.reshape(1, -1, 1, 1)
        elif self.per_channel:
             # Calculate per-channel min/max over (Batch, Height, Width) for consistent scaling per channel
             # Note: Kernel expects (1, C, 1, 1) layout
             x_min = x.amin(dim=(0, 2, 3), keepdim=True)
             x_max = x.amax(dim=(0, 2, 3), keepdim=True)
        else:
             x_min = x.min().reshape(1, 1, 1, 1)
             x_max = x.max().reshape(1, 1, 1, 1)

        # The kernel indexes min_vals[c]/max_vals[c] per channel — broadcast
        # per-tensor (or scalar frozen) statistics to C entries so a
        # per_channel=False configuration cannot read out of bounds.
        if x_min.numel() != C:
            x_min = x_min.expand(1, C, 1, 1)
            x_max = x_max.expand(1, C, 1, 1)

        # Eq.(19) learned soft mask m(p) — handed to the kernel, which fuses
        # the multiply with quantization (paper Listing 2).
        m = None
        if self.smooth_transitions and self.soft_mask is not None:
            m = self.soft_mask(bit_map, x).float().contiguous()

        return mcaq_cuda_ops.spatial_quantize(
            x.contiguous(),
            bit_map.float().contiguous(),
            x_min.float().contiguous(),
            x_max.float().contiguous(),
            tile_h, tile_w,
            m,
        )

    def _forward_pytorch(self, x: torch.Tensor, bit_map: torch.Tensor, training: bool) -> torch.Tensor:
        """Original PyTorch forward pass (slow, but differentiable)."""
        B, C, H, W = x.shape
        B_b, H_tiles, W_tiles = bit_map.shape

        assert B == B_b, f"Batch size mismatch: {B} vs {B_b}"

        tile_h = H // H_tiles
        tile_w = W // W_tiles

        # Get unique bit widths in the map
        unique_bits = torch.unique(bit_map)

        # NOTE (paper fidelity): smooth transitions come from spatially smoothing
        # the learned mask m(p) — "rather than by summing contributions from
        # multiple tiles" (Sec IV-D). The previous Gaussian-mask-summing branch
        # contradicted that sentence and was removed; every spatial position is
        # quantized with exactly one tile's bit-width, then scaled by m(p).
        if training:
            # Paper Eq.(19): tile-wise mixed precision must be maintained during
            # training. Fractional-bit composition keeps the quantized forward
            # differentiable w.r.t. the (continuous) bit map:
            #     x_q = (1-frac) * Q_floor(b)(x) + frac * Q_ceil(b)(x),
            #     frac = b - floor(b)
            # so d(x_q)/db = Q_ceil(x) - Q_floor(x) and Ldet/LKD gradients reach
            # the bit-mapping network and complexity MLP through the quantization
            # operator (paper Training note / Algorithm 3 QuantizedForward).
            # Integer bit maps (inference-style) reduce to plain per-tile STE.
            b_floor = torch.floor(bit_map)
            frac = (bit_map - b_floor).unsqueeze(1)  # (B,1,Ht,Wt), carries grad
            frac_up = F.interpolate(frac, size=(H, W), mode='nearest')

            x_quantized = torch.zeros_like(x)
            for bf in torch.unique(b_floor.detach()):
                bf_int = int(bf.item())
                bc_int = bf_int + 1
                sel = (b_floor == bf).float().unsqueeze(1)  # hard tile selection
                sel_up = F.interpolate(sel, size=(H, W), mode='nearest')

                q_lo = self.quantize_tensor(x, bf_int, training)
                if bc_int <= 8:
                    q_hi = self.quantize_tensor(x, bc_int, training)
                else:
                    q_hi = q_lo  # b == bmax exactly; frac is 0 there
                x_quantized = x_quantized + sel_up * (
                    (1.0 - frac_up) * q_lo + frac_up * q_hi
                )

        else:
            # Inference: integer per-tile composition via nearest masks
            # (single-tile assignment per spatial position, Sec IV-D)
            x_quantized = torch.zeros_like(x)
            for bits_value in unique_bits:
                bits_int = int(round(float(bits_value.item())))
                x_q = self.quantize_tensor(x, bits_int, training)
                tile_mask = (bit_map == bits_value).float().unsqueeze(1)
                mask = F.interpolate(tile_mask, size=(H, W), mode='nearest')
                x_quantized = x_quantized + x_q * mask

        # Eq.(19): X_quantized(p) = m(p) * Q_{bT(p)}(X(p)) — learned soft mask,
        # softmax-based + spatially smoothed
        if self.smooth_transitions and self.soft_mask is not None:
            m = self.soft_mask(bit_map, x)
            x_quantized = x_quantized * m

        return x_quantized
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f'calibration_mode={self.calibration_mode}, '
            f'smooth_transitions={self.smooth_transitions}, '
            f'per_channel={self.per_channel}'
        )


