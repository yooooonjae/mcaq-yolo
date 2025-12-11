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


class SpatialAdaptiveQuantization(nn.Module):
    """
    Hardware-aware spatial adaptive quantization module.
    
    Implements tile-wise mixed-precision quantization with smooth transitions.
    """
    
    def __init__(
        self,
        calibration_mode: str = 'minmax',  # 'minmax', 'percentile', 'entropy', 'mse'
        smooth_transitions: bool = True,
        per_channel: bool = True,
        learned_rounding: bool = False,
        momentum: float = 0.1
    ):
        """
        Initialize spatial adaptive quantization.
        
        Args:
            calibration_mode: Method for computing quantization parameters
            smooth_transitions: Whether to use smooth transitions between tiles
            per_channel: Whether to quantize per-channel or per-tensor
            learned_rounding: Whether to use learned rounding
            momentum: Momentum for running statistics
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
        
        # Learned rounding module
        self.learned_rounding = None
        if learned_rounding:
            self.learned_rounding = LearnedRoundingQuantization()
        
        # Smooth transition parameters
        if smooth_transitions:
            self.transition_kernel_size = 3
            self.register_buffer(
                'transition_kernel',
                self._create_transition_kernel()
            )
        
        # Calibration statistics
        self.register_buffer('calibration_histogram', None)
        self.histogram_bins = 2048
    
    def _create_transition_kernel(self) -> torch.Tensor:
        """Create Gaussian kernel for smooth transitions between tiles."""
        k = self.transition_kernel_size
        sigma = k / 3.0
        
        # Create 1D Gaussian
        x = torch.arange(k, dtype=torch.float32) - k // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        return kernel_2d
    
    def update_running_stats(self, x: torch.Tensor):
        """
        Update running statistics for calibration.
        
        Args:
            x: Input tensor
        """
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
            
            # Update running statistics
            if self.running_min is None:
                self.running_min = x_min
                self.running_max = x_max
            else:
                self.running_min = (1 - self.momentum) * self.running_min + self.momentum * x_min
                self.running_max = (1 - self.momentum) * self.running_max + self.momentum * x_max
            
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
                # Exponential moving average
                self.calibration_histogram = (
                    (1 - self.momentum) * self.calibration_histogram +
                    self.momentum * hist
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
        if self.training and self.running_min is not None:
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
        if self.per_channel:
            # Reshape for per-channel percentile
            x_reshaped = x.transpose(0, 1).flatten(1)
            x_min = torch.quantile(x_reshaped, percentile_min / 100, dim=1, keepdim=True)
            x_max = torch.quantile(x_reshaped, percentile_max / 100, dim=1, keepdim=True)
            
            # Reshape back
            x_min = x_min.view(1, -1, 1, 1)
            x_max = x_max.view(1, -1, 1, 1)
        else:
            x_min = torch.quantile(x, percentile_min / 100)
            x_max = torch.quantile(x, percentile_max / 100)
        
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
    
    def create_tile_masks(
        self,
        shape: Tuple[int, ...],
        tile_shape: Tuple[int, int],
        bit_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Create smooth transition masks for tiles.
        
        Args:
            shape: Shape of input tensor (B, C, H, W)
            tile_shape: Shape of each tile (tile_h, tile_w)
            bit_map: Bit allocation map
            
        Returns:
            Masks for smooth transitions between tiles
        """
        B, C, H, W = shape
        tile_h, tile_w = tile_shape
        B_b, h_tiles, w_tiles = bit_map.shape
        
        # Create base masks
        masks = torch.zeros(B, h_tiles * w_tiles, H, W, device=bit_map.device)
        
        for b in range(B):
            mask_idx = 0
            for i in range(h_tiles):
                for j in range(w_tiles):
                    # Create binary mask for this tile
                    mask = torch.zeros(H, W, device=bit_map.device)
                    
                    # Set tile region to 1
                    h_start = i * tile_h
                    h_end = min((i + 1) * tile_h, H)
                    w_start = j * tile_w
                    w_end = min((j + 1) * tile_w, W)
                    
                    mask[h_start:h_end, w_start:w_end] = 1.0
                    
                    if self.smooth_transitions and self.transition_kernel is not None:
                        # Apply Gaussian smoothing for smooth transitions
                        mask = mask.unsqueeze(0).unsqueeze(0)
                        mask = F.conv2d(
                            mask,
                            self.transition_kernel.to(mask.device),
                            padding=self.transition_kernel_size // 2
                        )
                        mask = mask.squeeze()
                    
                    masks[b, mask_idx] = mask
                    mask_idx += 1
        
        # Normalize masks so they sum to 1 at each position
        mask_sum = masks.sum(dim=1, keepdim=True)
        masks = masks / (mask_sum + 1e-8)
        
        return masks
    
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
        
        # Calculate min/max stats on the fly (Dynamic Quantization for Inference)
        # Or you could use self.running_min/max if you prefer static quantization
        if self.per_channel:
             # Calculate per-channel min/max over (Batch, Height, Width) for consistent scaling per channel
             # Note: Kernel expects (1, C, 1, 1) layout
             x_min = x.amin(dim=(0, 2, 3), keepdim=True)
             x_max = x.amax(dim=(0, 2, 3), keepdim=True)
        else:
             x_min = x.min().view(1, 1, 1, 1)
             x_max = x.max().view(1, 1, 1, 1)

        # Ensure contiguous memory for CUDA
        return mcaq_cuda_ops.spatial_quantize(
            x.contiguous(),
            bit_map.float().contiguous(),
            x_min.contiguous(),
            x_max.contiguous(),
            tile_h, tile_w
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
        
        if self.smooth_transitions:
            # Use smooth transitions between different bit widths
            x_quantized = torch.zeros_like(x)
            
            # Create masks for smooth transitions
            masks = self.create_tile_masks(x.shape, (tile_h, tile_w), bit_map)
            
            for bits_value in unique_bits:
                bits_int = int(bits_value.item())
                
                # Quantize entire tensor with this bit width
                x_quant_bits = self.quantize_tensor(x, bits_int, training)
                
                # Create combined mask for all tiles with this bit width
                combined_mask = torch.zeros(B, 1, H, W, device=x.device)
                
                for b in range(B):
                    mask_idx = 0
                    for i in range(H_tiles):
                        for j in range(W_tiles):
                            if bit_map[b, i, j] == bits_value:
                                combined_mask[b, 0] += masks[b, mask_idx]
                            mask_idx += 1
                
                # Apply weighted quantization
                x_quantized += x_quant_bits * combined_mask
        
        else:
            # Hard boundaries between tiles
            x_quantized = torch.zeros_like(x)
            
            for b in range(B):
                for i in range(H_tiles):
                    for j in range(W_tiles):
                        # Get bit width for this tile
                        tile_bits = int(bit_map[b, i, j].item())
                        
                        # Extract tile
                        h_start = i * tile_h
                        h_end = min((i + 1) * tile_h, H)
                        w_start = j * tile_w
                        w_end = min((j + 1) * tile_w, W)
                        
                        tile = x[b:b+1, :, h_start:h_end, w_start:w_end]
                        
                        # Quantize tile
                        tile_quant = self.quantize_tensor(tile, tile_bits, training)
                        
                        # Place back
                        x_quantized[b, :, h_start:h_end, w_start:w_end] = tile_quant[0]
        
        return x_quantized
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f'calibration_mode={self.calibration_mode}, '
            f'smooth_transitions={self.smooth_transitions}, '
            f'per_channel={self.per_channel}'
        )


class MixedPrecisionQuantizer(nn.Module):
    """
    Advanced mixed-precision quantizer with hardware awareness.
    """
    
    def __init__(
        self,
        weight_quant: bool = True,
        activation_quant: bool = True,
        hardware_type: str = 'gpu',  # 'gpu', 'cpu', 'npu', 'edge', 'mobile'
        symmetric: bool = True,
        per_channel_weight: bool = True,
        per_tensor_activation: bool = True
    ):
        """
        Initialize mixed-precision quantizer.
        
        Args:
            weight_quant: Whether to quantize weights
            activation_quant: Whether to quantize activations
            hardware_type: Target hardware platform
            symmetric: Whether to use symmetric quantization
            per_channel_weight: Per-channel quantization for weights
            per_tensor_activation: Per-tensor quantization for activations
        """
        super().__init__()
        self.weight_quant = weight_quant
        self.activation_quant = activation_quant
        self.hardware_type = hardware_type
        self.symmetric = symmetric
        
        # Hardware-specific constraints
        self.hardware_constraints = self._get_hardware_constraints()
        
        # Quantization modules
        if weight_quant:
            self.weight_quantizer = SpatialAdaptiveQuantization(
                calibration_mode='minmax',
                per_channel=per_channel_weight,
                smooth_transitions=False
            )
        
        if activation_quant:
            self.activation_quantizer = SpatialAdaptiveQuantization(
                calibration_mode='percentile',
                per_channel=not per_tensor_activation,
                smooth_transitions=True
            )
        
        # Hardware-specific optimizations
        self.optimizations = self._get_hardware_optimizations()
    
    def _get_hardware_constraints(self) -> Dict:
        """Get hardware-specific constraints."""
        constraints = {
            'gpu': {
                'supported_bits': [4, 8, 16],
                'tile_size': 32,
                'preferred_bits': 8,
                'max_tile_bits_variance': 4,  # Max difference in bits between tiles
                'supports_mixed_precision': True
            },
            'cpu': {
                'supported_bits': [8, 16, 32],
                'tile_size': 16,
                'preferred_bits': 8,
                'max_tile_bits_variance': 2,
                'supports_mixed_precision': False
            },
            'npu': {
                'supported_bits': [4, 8],
                'tile_size': 64,
                'preferred_bits': 4,
                'max_tile_bits_variance': 4,
                'supports_mixed_precision': True
            },
            'edge': {
                'supported_bits': [2, 4, 8],
                'tile_size': 8,
                'preferred_bits': 4,
                'max_tile_bits_variance': 2,
                'supports_mixed_precision': True
            },
            'mobile': {
                'supported_bits': [8, 16],
                'tile_size': 16,
                'preferred_bits': 8,
                'max_tile_bits_variance': 0,
                'supports_mixed_precision': False
            }
        }
        return constraints.get(self.hardware_type, constraints['gpu'])
    
    def _get_hardware_optimizations(self) -> Dict:
        """Get hardware-specific optimizations."""
        optimizations = {
            'gpu': {
                'use_tensor_cores': True,
                'fusion_enabled': True,
                'cache_friendly_tiling': True
            },
            'cpu': {
                'use_vectorization': True,
                'fusion_enabled': False,
                'cache_friendly_tiling': True
            },
            'npu': {
                'use_tensor_cores': True,
                'fusion_enabled': True,
                'cache_friendly_tiling': False
            },
            'edge': {
                'use_vectorization': False,
                'fusion_enabled': False,
                'cache_friendly_tiling': True
            },
            'mobile': {
                'use_vectorization': True,
                'fusion_enabled': False,
                'cache_friendly_tiling': True
            }
        }
        return optimizations.get(self.hardware_type, optimizations['gpu'])
    
    def adjust_bit_map_for_hardware(
        self,
        bit_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Adjust bit map to match hardware constraints.
        
        Args:
            bit_map: Original bit allocation map
            
        Returns:
            Hardware-compatible bit map
        """
        supported_bits = torch.tensor(
            self.hardware_constraints['supported_bits'],
            device=bit_map.device
        )
        
        # Round to nearest supported bit-width
        adjusted_map = bit_map.clone()
        
        for i in range(bit_map.shape[0]):
            for j in range(bit_map.shape[1]):
                for k in range(bit_map.shape[2]):
                    current_bits = bit_map[i, j, k]
                    
                    # Find nearest supported bit width
                    distances = torch.abs(supported_bits - current_bits)
                    nearest_idx = torch.argmin(distances)
                    adjusted_map[i, j, k] = supported_bits[nearest_idx]
        
        # Enforce maximum variance constraint if not supporting mixed precision
        if not self.hardware_constraints['supports_mixed_precision']:
            # Use uniform bit width (the mode)
            mode_bits = torch.mode(adjusted_map.flatten())[0]
            adjusted_map.fill_(mode_bits)
        elif self.hardware_constraints['max_tile_bits_variance'] > 0:
            # Limit variance between neighboring tiles
            self._enforce_variance_constraint(adjusted_map)
        
        return adjusted_map
    
    def _enforce_variance_constraint(self, bit_map: torch.Tensor):
        """Enforce maximum bit variance between neighboring tiles."""
        max_var = self.hardware_constraints['max_tile_bits_variance']
        
        B, H, W = bit_map.shape
        
        # Iteratively smooth the bit map to respect variance constraints
        for _ in range(5):  # Max iterations
            changed = False
            
            for b in range(B):
                for i in range(H):
                    for j in range(W):
                        current = bit_map[b, i, j]
                        
                        # Check neighbors
                        neighbors = []
                        if i > 0:
                            neighbors.append(bit_map[b, i-1, j])
                        if i < H-1:
                            neighbors.append(bit_map[b, i+1, j])
                        if j > 0:
                            neighbors.append(bit_map[b, i, j-1])
                        if j < W-1:
                            neighbors.append(bit_map[b, i, j+1])
                        
                        if neighbors:
                            # Adjust if variance is too high
                            for neighbor in neighbors:
                                if abs(current - neighbor) > max_var:
                                    # Move current towards neighbor
                                    if current > neighbor:
                                        bit_map[b, i, j] = neighbor + max_var
                                    else:
                                        bit_map[b, i, j] = neighbor - max_var
                                    changed = True
            
            if not changed:
                break
    
    def forward(
        self,
        weights: Optional[torch.Tensor],
        activations: Optional[torch.Tensor],
        weight_bit_map: Optional[torch.Tensor],
        activation_bit_map: Optional[torch.Tensor],
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights and activations with hardware awareness.
        
        Args:
            weights: Weight tensor
            activations: Activation tensor
            weight_bit_map: Bit allocation for weights
            activation_bit_map: Bit allocation for activations
            training: Whether in training mode
            
        Returns:
            Quantized weights and activations
        """
        # Adjust bit maps for hardware if provided
        if weight_bit_map is not None:
            weight_bit_map = self.adjust_bit_map_for_hardware(weight_bit_map)
        
        if activation_bit_map is not None:
            activation_bit_map = self.adjust_bit_map_for_hardware(activation_bit_map)
        
        # Quantize weights
        weights_q = weights
        if self.weight_quant and weights is not None and weight_bit_map is not None:
            weights_q = self.weight_quantizer(weights, weight_bit_map, training)
        
        # Quantize activations
        activations_q = activations
        if self.activation_quant and activations is not None and activation_bit_map is not None:
            activations_q = self.activation_quantizer(
                activations, activation_bit_map, training
            )
        
        return weights_q, activations_q
    
    def get_hardware_info(self) -> Dict:
        """Get hardware configuration information."""
        return {
            'hardware_type': self.hardware_type,
            'constraints': self.hardware_constraints,
            'optimizations': self.optimizations
        }