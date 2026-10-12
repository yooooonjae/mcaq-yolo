"""
Bit Allocation Module for Complexity-Aware Quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


@torch.no_grad()
def feature_std_complexity(features, grid_size=8):
    """Simple baseline: within-tile standard deviation of channel-mean features.

    Normalize by the per-image maximum (flat images map to zero). No labels,
    morphology, learned combination or changes to the downstream bit mapper.
    """
    gray = features.detach().float().mean(1)
    b, h, w = gray.shape
    ht, wt = min(grid_size, h), min(grid_size, w)
    score = gray.new_zeros(b, ht, wt)
    for i in range(ht):
        y0, y1 = (i * h + ht - 1) // ht, ((i + 1) * h + ht - 1) // ht
        for j in range(wt):
            x0, x1 = (j * w + wt - 1) // wt, ((j + 1) * w + wt - 1) // wt
            score[:, i, j] = gray[:, y0:y1, x0:x1].flatten(1).std(1, correction=0)
    return score / score.amax((-2, -1), keepdim=True).clamp_min(1e-8)


def interpolate_precision(base: torch.Tensor, progress: float, max_bits: float = 8.) -> torch.Tensor:
    """Convex precision continuation; its derivative w.r.t. base is progress."""
    if not 0. <= progress <= 1.:
        raise ValueError('precision_progress must be finite and in [0,1]')
    return (1. - progress) * max_bits + progress * base


def tile_areas(bits: torch.Tensor, spatial_shape) -> torch.Tensor:
    """Areas for floor(pixel * tiles / size), including nondivisible edges."""
    h, w = spatial_shape
    ht, wt = bits.shape[-2:]
    rows = torch.div(torch.arange(ht + 1, device=bits.device) * h + ht - 1, ht,
                     rounding_mode='floor').diff()
    cols = torch.div(torch.arange(wt + 1, device=bits.device) * w + wt - 1, wt,
                     rounding_mode='floor').diff()
    return (rows[:, None] * cols[None, :]).to(bits.dtype)


def apply_allocation_policy(bits, spatial_shape, policy='adaptive', uniform_bits=4., seed=0):
    """Control interventions before continuation/rounding, shared by train/eval.

    Shuffle permutes within equal-area groups independently per batch position.
    Its local CPU generator does not perturb training/augmentation RNG streams.
    """
    if policy == 'adaptive':
        return bits
    if policy == 'uniform':
        return torch.full_like(bits, uniform_bits)
    areas = tile_areas(bits, spatial_shape)
    if policy == 'layerwise':
        mean = (bits * areas).sum((-2, -1), keepdim=True) / areas.sum()
        return mean.expand_as(bits)
    if policy != 'shuffle':
        raise ValueError(f'unknown allocation_policy: {policy}')
    flat = bits.flatten(1)
    # Equal areas preserve both the tile histogram and activation element cost.
    cpu_areas = areas.detach().cpu().flatten()
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    permutations = []
    for _ in range(bits.shape[0]):
        order = torch.arange(flat.shape[1])
        for area in cpu_areas.unique():
            indices = (cpu_areas == area).nonzero().flatten()
            order[indices] = indices[torch.randperm(len(indices), generator=generator)]
        permutations.append(order)
    order = torch.stack(permutations).to(bits.device)
    return flat.gather(1, order).reshape_as(bits)


class LinearBitMapper(nn.Module):
    """
    Paper Table V / Table VIII 'Linear mapping (no MLP)' ablation configuration:

        b = b_min + (b_max - b_min) * normalize(C)

    with per-image min-max normalization of the complexity map (the absolute
    scale of C is arbitrary across training regimes; normalization exposes the
    RELATIVE spatial structure, which is what drives tile-wise allocation).
    Parameter-free — useful at small training scales, where the learned MLP
    path collapses to a spatially flat map (the Lbit/Lsmooth flat-map
    attractor; see project notes), and as the paper's ablation baseline.

    Same interface as ComplexityToBitMappingNetwork (temperature multiply per
    Algorithm 3 line 13, clamp, straight-through rounding).
    """

    def __init__(self, min_bits: int = 2, max_bits: int = 8,
                 eps_spread: float = 1e-3):
        super().__init__()
        self.min_bits = float(min_bits)
        self.max_bits = float(max_bits)
        # Minimum 2-98% percentile spread below which the map is treated as
        # spatially FLAT and mapped through absolute complexity instead of
        # relative normalization (see forward; review fix).
        self.eps_spread = float(eps_spread)

    def enforce_weight_constraints(self):
        """No-op (parameter-free); kept for interface parity (Eq.18)."""

    def forward(
        self,
        complexity: torch.Tensor,
        temperature: Optional[float] = None,
        return_continuous: bool = False,
    ) -> torch.Tensor:
        c = ComplexityToBitMappingNetwork._normalize_complexity_shape(complexity)

        # Per-image percentile (2-98%) normalization -> relative spatial
        # structure. Plain min-max is fragile: a single outlier tile (e.g. an
        # Otsu artifact) squashes every other tile to b_min (observed on
        # coco128 000000000034: bits {2: 399, 8: 1}).
        B = c.shape[0]
        flat = c.reshape(B, -1).float()
        lo = torch.quantile(flat, 0.02, dim=1, keepdim=True).unsqueeze(-1)
        hi = torch.quantile(flat, 0.98, dim=1, keepdim=True).unsqueeze(-1)
        spread = hi - lo
        rel = ((c - lo) / (spread + 1e-8)).clamp(0.0, 1.0)
        # REVIEW FIX (measured degenerate case): a spatially FLAT map
        # (spread ~ 0, e.g. a uniform-texture scene) carries no relative
        # structure, and relative normalization then collapsed EVERY tile to
        # b_min regardless of absolute complexity (constant C=0.5 -> all
        # 2-bit, measured) — maximal compression exactly where nothing says
        # compression is safe. Hard gate: below eps_spread, map the ABSOLUTE
        # complexity through the same affine so a uniformly mid-complexity
        # image lands on mid bits. The gate is per image.
        cn = torch.where(spread > self.eps_spread, rel, c.clamp(0.0, 1.0))

        bit_map = self.min_bits + (self.max_bits - self.min_bits) * cn

        if temperature is not None:
            bit_map = bit_map * max(float(temperature), 0.1)

        clamped = torch.clamp(bit_map, self.min_bits, self.max_bits)
        bit_map = bit_map + (clamped - bit_map).detach()

        if not return_continuous:
            bit_map = bit_map + (torch.round(bit_map) - bit_map).detach()
        return bit_map


class ComplexityToBitMappingNetwork(nn.Module):
    """
    Learnable network that maps morphological complexity scores to bit allocations.

    Supports inputs with shape:
        - (H, W)
        - (B, H, W)
        - (B, 1, H, W)
        - (B, C, H, W)  (used after channel averaging)

    Internally normalized to (B, H, W) form before use.
    """

    def __init__(
        self,
        min_bits: int = 2,
        max_bits: int = 8,
        hidden_dims: list = [32, 64, 32],  # Paper Table X: Mapping MLP hidden dims [32, 64, 32]
        enforce_monotonicity: bool = True,
    ):
        """
        Initialize the bit mapping network.

        Args:
            min_bits: Minimum bit-width allocation
            max_bits: Maximum bit-width allocation
            hidden_dims: Hidden layer dimensions
            enforce_monotonicity: Whether to enforce monotonic mapping
        """
        super().__init__()
        self.min_bits = float(min_bits)
        self.max_bits = float(max_bits)
        self.enforce_monotonicity = enforce_monotonicity

        # Build network layers — paper Eq.(14)-(16): h = ReLU(BN(W z + b)) x3
        # (no dropout in the paper's formulation)
        layers = []
        input_dim = 3  # Eq.(13): z0 = [C, C^2, log(1+C)]

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            input_dim = hidden_dim

        # Eq.(17): b = bmin + (bmax - bmin) * sigmoid(w4^T h3 + b4)
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.mapping_network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    # ---------------------------------------------------
    # SHAPE normalization utility
    # ---------------------------------------------------
    @staticmethod
    def _normalize_complexity_shape(complexity: torch.Tensor) -> torch.Tensor:
        """
        Normalize complexity tensor to shape (B, H, W).

        Accepted inputs:
            - (H, W)           → (1, H, W)
            - (B, H, W)        → unchanged
            - (B, 1, H, W)     → (B, H, W) after channel averaging
            - (B, C, H, W)     → (B, H, W) after channel averaging
        """
        if not isinstance(complexity, torch.Tensor):
            raise TypeError(f"complexity must be torch.Tensor, got {type(complexity)}")

        if complexity.dim() == 2:
            # (H, W) → (1, H, W)
            complexity = complexity.unsqueeze(0)
        elif complexity.dim() == 3:
            # (B, H, W) unchanged
            pass
        elif complexity.dim() == 4:
            # (B, C, H, W) → (B, H, W) after channel averaging
            complexity = complexity.mean(dim=1)
        else:
            raise ValueError(
                f"Unsupported complexity dim={complexity.dim()}, "
                f"expected 2, 3, or 4."
            )
        return complexity

    def _init_weights(self, m):
        """Initialize network weights for identity-like mapping (complexity → bits)."""
        if isinstance(m, nn.Linear):
            # Initialize to approximate identity mapping
            # So that high complexity → high bits, low complexity → low bits
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if self.enforce_monotonicity:
                m.weight.data = torch.abs(m.weight.data)
            if m.bias is not None:
                # Small positive bias to ensure output is not all zeros
                nn.init.constant_(m.bias, 0.1)

    def enforce_weight_constraints(self):
        """Enforce non-negative weights for monotonicity (Eq.18).

        BatchNorm affine scales are constrained too: a negative gamma would
        invert the sign of its channel and break the monotone composition that
        |W| alone is meant to guarantee (reviewer minor: BN affine)."""
        if self.enforce_monotonicity:
            for module in self.mapping_network.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = torch.abs(module.weight.data)
                elif isinstance(module, nn.BatchNorm1d):
                    module.weight.data = torch.abs(module.weight.data)

    def create_augmented_features(self, complexity: torch.Tensor) -> torch.Tensor:
        """
        Create augmented feature vector from complexity scores.

        Args:
            complexity: Complexity scores of shape (*, 1)

        Returns:
            Augmented features of shape (*, 3)
        """
        return torch.cat(
            [
                complexity,
                complexity ** 2,
                torch.log1p(complexity),  # log(1 + C)
            ],
            dim=-1,
        )

    def forward(
        self,
        complexity: torch.Tensor,
        temperature: Optional[float] = None,
        return_continuous: bool = False,
    ) -> torch.Tensor:
        """
        Map complexity scores to bit allocations via the learnable MLP (paper Eq.13-17).

            z0 = [C, C^2, log(1+C)]                       (Eq.13)
            h  = ReLU(BN(W z + b)) x3                      (Eq.14-16)
            b  = bmin + (bmax-bmin) * sigmoid(w4^T h3+b4)  (Eq.17)
            b  = b * alpha_t                               (Algorithm 3 line 13, literal)

        Temperature semantics: alpha_t anneals 10 -> 1 (Sec IV-C). Early in training
        the multiplied bits saturate at bmax after clamping (high precision everywhere
        — consistent with Stage 1's high-precision warm-up); as alpha_t -> 1 the
        allocation converges to the adaptive Eq.(17) range. Note Eq.(17) already
        bounds b in [bmin,bmax], so the literal multiply is only meaningful through
        this clamped warm-up behavior.

        Args:
            complexity: Tensor with shape (B,H,W) or (B,1,H,W) or (B,C,H,W) or (H,W)
            temperature: alpha_t for bit allocation (None -> 1.0)
            return_continuous: If True, return continuous bit values

        Returns:
            bit_map: Tensor of shape (B, H, W) with bit allocations
        """
        # 1) shape normalization → (B, H, W)
        complexity = self._normalize_complexity_shape(complexity)

        # 2) clamp value range
        complexity = complexity.clamp(0.0, 1.0)

        B, H, W = complexity.shape

        # 3) Eq.(13): polynomial feature expansion per tile
        z0 = self.create_augmented_features(complexity.reshape(-1, 1))  # (N,3)

        # 4) Eq.(14)-(17): MLP -> sigmoid -> scale to [bmin, bmax]
        h = self.mapping_network(z0)  # (N,1) in (0,1)
        bit_map = self.min_bits + (self.max_bits - self.min_bits) * h
        bit_map = bit_map.reshape(B, H, W)

        # 5) Algorithm 3 line 13: b_batch <- f_theta(C_batch) * alpha_t
        if temperature is not None:
            bit_map = bit_map * max(float(temperature), 0.1)

        # 6) Clip to valid range (CUDA kernel also clamps to [2,8]).
        # Straight-through clamp: with alpha_t in [1,10] most tiles saturate at
        # bmax for much of training, and a hard clamp would zero the gradient to
        # the mapping network exactly when Lbit needs to pull the average down.
        # Forward = clamped value, backward = identity.
        clamped = torch.clamp(bit_map, self.min_bits, self.max_bits)
        bit_map = bit_map + (clamped - bit_map).detach()

        if not return_continuous:
            # Straight-through rounding: integer bits forward, identity gradient —
            # keeps Lbit/Lsmooth differentiable w.r.t. the mapping network.
            bit_map = bit_map + (torch.round(bit_map) - bit_map).detach()

        return bit_map

    def get_bit_statistics(self, bit_map: torch.Tensor) -> dict:
        """
        Compute statistics of bit allocation.

        Args:
            bit_map: Bit allocation tensor

        Returns:
            Dictionary with bit allocation statistics
        """
        return {
            "mean": bit_map.mean().item(),
            "std": bit_map.std().item(),
            "min": bit_map.min().item(),
            "max": bit_map.max().item(),
            "histogram": torch.histc(
                bit_map,
                bins=int(self.max_bits - self.min_bits + 1),
                min=self.min_bits,
                max=self.max_bits,
            ),
        }

