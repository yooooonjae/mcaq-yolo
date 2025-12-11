"""
Bit Allocation Module for Complexity-Aware Quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ComplexityToBitMappingNetwork(nn.Module):
    """
    Learnable network that maps morphological complexity scores to bit allocations.
    
    Implements a monotonic mapping function with curriculum learning support.
    """
    
    def __init__(
        self,
        min_bits: int = 2,
        max_bits: int = 8,
        hidden_dims: list = [128, 64, 32],
        enforce_monotonicity: bool = True
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
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.enforce_monotonicity = enforce_monotonicity
        
        # Build network layers
        layers = []
        input_dim = 3  # C, C^2, log(1+C)
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mapping_network = nn.Sequential(*layers)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if self.enforce_monotonicity:
                m.weight.data = torch.abs(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def enforce_weight_constraints(self):
        """Enforce non-negative weights for monotonicity."""
        if self.enforce_monotonicity:
            for module in self.mapping_network.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = torch.abs(module.weight.data)
    
    def create_augmented_features(self, complexity: torch.Tensor) -> torch.Tensor:
        """
        Create augmented feature vector from complexity scores.
        
        Args:
            complexity: Complexity scores of shape (*, 1)
            
        Returns:
            Augmented features of shape (*, 3)
        """
        return torch.cat([
            complexity,
            complexity ** 2,
            torch.log(1 + complexity)
        ], dim=-1)
    
    def forward(
        self,
        complexity: torch.Tensor,
        temperature: Optional[float] = None,
        return_continuous: bool = False
    ) -> torch.Tensor:
        """
        Map complexity scores to bit allocations.
        
        Args:
            complexity: Tensor of shape (B, H, W) with values in [0, 1]
            temperature: Temperature for bit allocation (overrides learned value)
            return_continuous: If True, return continuous bit values
            
        Returns:
            bit_map: Tensor of shape (B, H, W) with bit allocations
        """
        B, H, W = complexity.shape
        
        # Flatten complexity map
        C_flat = complexity.view(-1, 1)
        
        # Create augmented features
        features = self.create_augmented_features(C_flat)
        
        # Get normalized bit allocation [0, 1]
        bit_norm = self.mapping_network(features).view(B, H, W)
        
        # Apply temperature (either provided or learned)
        if temperature is not None:
            temp = temperature
        else:
            temp = F.softplus(self.temperature)  # Ensure positive
        
        # Scale to actual bit range
        bit_range = self.max_bits - self.min_bits
        bit_map = self.min_bits + bit_range * bit_norm
        
        # Apply temperature scaling
        bit_map = bit_map * temp
        
        # Clip to valid range
        bit_map = torch.clamp(bit_map, self.min_bits, self.max_bits)
        
        if not return_continuous:
            # Round to nearest integer for discrete allocation
            bit_map = torch.round(bit_map)
        
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
            'mean': bit_map.mean().item(),
            'std': bit_map.std().item(),
            'min': bit_map.min().item(),
            'max': bit_map.max().item(),
            'histogram': torch.histc(bit_map, bins=self.max_bits - self.min_bits + 1,
                                    min=self.min_bits, max=self.max_bits)
        }


class AdaptiveBitAllocation(nn.Module):
    """
    Advanced bit allocation strategy with multiple policies.
    """
    
    def __init__(
        self,
        min_bits: int = 2,
        max_bits: int = 8,
        target_bits: float = 4.0,
        policy: str = 'learned'  # 'learned', 'linear', 'exponential', 'threshold'
    ):
        """
        Initialize adaptive bit allocation.
        
        Args:
            min_bits: Minimum bit-width
            max_bits: Maximum bit-width
            target_bits: Target average bits
            policy: Bit allocation policy
        """
        super().__init__()
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.target_bits = target_bits
        self.policy = policy
        
        if policy == 'learned':
            self.mapping_network = ComplexityToBitMappingNetwork(
                min_bits, max_bits
            )
        elif policy == 'threshold':
            # Learnable thresholds for piece-wise allocation
            n_thresholds = 3
            self.thresholds = nn.Parameter(
                torch.linspace(0.3, 0.7, n_thresholds)
            )
            self.bit_values = nn.Parameter(
                torch.linspace(min_bits, max_bits, n_thresholds + 1)
            )
    
    def linear_allocation(self, complexity: torch.Tensor) -> torch.Tensor:
        """Linear mapping from complexity to bits."""
        bit_range = self.max_bits - self.min_bits
        return self.min_bits + bit_range * complexity
    
    def exponential_allocation(self, complexity: torch.Tensor) -> torch.Tensor:
        """Exponential mapping from complexity to bits."""
        bit_range = self.max_bits - self.min_bits
        return self.min_bits + bit_range * (torch.exp(complexity) - 1) / (np.e - 1)
    
    def threshold_allocation(self, complexity: torch.Tensor) -> torch.Tensor:
        """Piece-wise constant allocation based on thresholds."""
        bit_map = torch.ones_like(complexity) * self.bit_values[0]
        
        for i, threshold in enumerate(self.thresholds):
            mask = complexity > threshold
            bit_map[mask] = self.bit_values[i + 1]
        
        return bit_map
    
    def forward(
        self,
        complexity: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Allocate bits based on complexity and policy.
        
        Args:
            complexity: Complexity scores
            temperature: Temperature parameter
            
        Returns:
            Bit allocation map
        """
        if self.policy == 'learned':
            return self.mapping_network(complexity, temperature)
        elif self.policy == 'linear':
            bit_map = self.linear_allocation(complexity)
        elif self.policy == 'exponential':
            bit_map = self.exponential_allocation(complexity)
        elif self.policy == 'threshold':
            bit_map = self.threshold_allocation(complexity)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
        
        # Apply temperature and round
        bit_map = bit_map * temperature
        bit_map = torch.clamp(bit_map, self.min_bits, self.max_bits)
        bit_map = torch.round(bit_map)
        
        return bit_map
    
    def compute_bit_budget_loss(
        self,
        bit_map: torch.Tensor,
        target_bits: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute loss for bit budget constraint.
        
        Args:
            bit_map: Current bit allocation
            target_bits: Target average bits (uses self.target_bits if None)
            
        Returns:
            Bit budget loss
        """
        if target_bits is None:
            target_bits = self.target_bits
        
        avg_bits = bit_map.mean()
        
        # Soft constraint with asymmetric penalty
        if avg_bits > target_bits:
            # Higher penalty for exceeding budget
            loss = 2.0 * (avg_bits - target_bits) ** 2
        else:
            # Lower penalty for under-utilizing budget
            loss = (avg_bits - target_bits) ** 2
        
        return loss