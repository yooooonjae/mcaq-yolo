"""
Curriculum Learning Module for MCAQ-YOLO
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm


class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive quantization training.
    
    Implements various curriculum strategies:
    1. Complexity-based curriculum
    2. Temperature annealing
    3. Bit-width progression
    """
    
    def __init__(
        self,
        warmup_epochs: int = 10,
        total_epochs: int = 300,
        initial_complexity: float = 0.2,
        initial_temperature: float = 10.0,
        initial_bits: float = 8.0,
        target_bits: float = 4.0,
        curriculum_type: str = 'exponential'  # 'linear', 'exponential', 'cosine', 'step'
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            initial_complexity: Starting complexity threshold
            initial_temperature: Starting temperature
            initial_bits: Starting bit-width
            target_bits: Target bit-width
            curriculum_type: Type of curriculum progression
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_complexity = initial_complexity
        self.initial_temperature = initial_temperature
        self.initial_bits = initial_bits
        self.target_bits = target_bits
        self.curriculum_type = curriculum_type
        
        # Track current state
        self.current_epoch = 0
        self.complexity_history = []
        self.temperature_history = []
        self.bits_history = []
        
    def get_complexity_threshold(self, epoch: int) -> float:
        """
        Get complexity threshold for current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Complexity threshold in [0, 1]
        """
        if epoch <= self.warmup_epochs:
            # Linear increase during warmup
            progress = epoch / self.warmup_epochs
            threshold = self.initial_complexity + (1.0 - self.initial_complexity) * progress
        else:
            # Full complexity after warmup
            threshold = 1.0
        
        self.complexity_history.append(threshold)
        return threshold
    
    def get_temperature(self, epoch: int) -> float:
        """
        Get temperature for bit allocation annealing.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Temperature value
        """
        if self.curriculum_type == 'linear':
            # Linear decay
            progress = min(epoch / self.total_epochs, 1.0)
            temperature = self.initial_temperature * (1.0 - progress) + 1.0 * progress
            
        elif self.curriculum_type == 'exponential':
            # Exponential decay
            decay_rate = 5000
            temperature = 1.0 + (self.initial_temperature - 1.0) * np.exp(-epoch / decay_rate)
            
        elif self.curriculum_type == 'cosine':
            # Cosine annealing
            progress = min(epoch / self.total_epochs, 1.0)
            temperature = 1.0 + 0.5 * (self.initial_temperature - 1.0) * (1 + np.cos(np.pi * progress))
            
        elif self.curriculum_type == 'step':
            # Step-wise decay
            milestones = [30, 60, 90, 120]
            temperature = self.initial_temperature
            for milestone in milestones:
                if epoch >= milestone:
                    temperature *= 0.5
        else:
            temperature = 1.0
        
        self.temperature_history.append(temperature)
        return temperature
    
    def get_target_bits(self, epoch: int) -> float:
        """
        Get target bit-width for current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Target bit-width
        """
        if epoch < self.warmup_epochs:
            # Start with higher precision
            target = self.initial_bits
        else:
            # Gradually reduce to target
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            
            if self.curriculum_type == 'exponential':
                # Exponential reduction
                target = self.target_bits + (self.initial_bits - self.target_bits) * np.exp(-3 * progress)
            else:
                # Linear reduction
                target = self.initial_bits - (self.initial_bits - self.target_bits) * progress
        
        self.bits_history.append(target)
        return target
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1
    
    def get_current_params(self) -> Dict:
        """Get all curriculum parameters for current epoch."""
        return {
            'epoch': self.current_epoch,
            'complexity_threshold': self.get_complexity_threshold(self.current_epoch),
            'temperature': self.get_temperature(self.current_epoch),
            'target_bits': self.get_target_bits(self.current_epoch)
        }
    
    def should_update_bit_allocation(self, epoch: int) -> bool:
        """Determine if bit allocation should be updated."""
        # Update every N epochs after warmup
        update_interval = 10
        return epoch > self.warmup_epochs and epoch % update_interval == 0
    
    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get loss component weights for current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of loss weights
        """
        # Gradually increase quantization loss importance
        if epoch < self.warmup_epochs:
            # Focus on detection during warmup
            weights = {
                'detection': 1.0,
                'bit_budget': 0.001,
                'smoothness': 0.0001,
                'distillation': 0.1,
                'regularization': 0.0001
            }
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            weights = {
                'detection': 1.0,
                'bit_budget': 0.001 + 0.009 * progress,  # Increase to 0.01
                'smoothness': 0.0001 + 0.0009 * progress,  # Increase to 0.001
                'distillation': 0.1 + 0.4 * progress,  # Increase to 0.5
                'regularization': 0.0001
            }
        
        return weights


class ComplexityBasedSampler:
    """
    Dataset sampler based on morphological complexity.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        complexity_scores: np.ndarray,
        batch_size: int = 32
    ):
        """
        Initialize complexity-based sampler.
        
        Args:
            dataset: Training dataset
            complexity_scores: Pre-computed complexity scores for dataset
            batch_size: Batch size for sampling
        """
        self.dataset = dataset
        self.complexity_scores = complexity_scores
        self.batch_size = batch_size
        
        # Sort indices by complexity
        self.sorted_indices = np.argsort(complexity_scores)
        
    def get_curriculum_subset(
        self,
        complexity_threshold: float,
        min_samples: int = 100
    ) -> Subset:
        """
        Get dataset subset based on complexity threshold.
        
        Args:
            complexity_threshold: Maximum complexity to include
            min_samples: Minimum number of samples to return
            
        Returns:
            Subset of dataset
        """
        # Find samples below threshold
        mask = self.complexity_scores <= complexity_threshold
        indices = np.where(mask)[0]
        
        # Ensure minimum samples
        if len(indices) < min_samples:
            # Include simplest samples up to min_samples
            indices = self.sorted_indices[:min_samples]
        
        return Subset(self.dataset, indices.tolist())
    
    def get_balanced_batch(
        self,
        epoch: int,
        complexity_threshold: float
    ) -> List[int]:
        """
        Get balanced batch with progressive complexity.
        
        Args:
            epoch: Current epoch
            complexity_threshold: Current complexity threshold
            
        Returns:
            List of indices for batch
        """
        # Get eligible samples
        eligible_mask = self.complexity_scores <= complexity_threshold
        eligible_indices = np.where(eligible_mask)[0]
        
        if len(eligible_indices) < self.batch_size:
            # Use all eligible samples and pad with simplest
            batch_indices = eligible_indices.tolist()
            remaining = self.batch_size - len(batch_indices)
            batch_indices.extend(self.sorted_indices[:remaining].tolist())
        else:
            # Sample from eligible indices
            # Use stratified sampling across complexity ranges
            n_strata = 4
            complexity_range = complexity_threshold / n_strata
            batch_indices = []
            
            for i in range(n_strata):
                lower = i * complexity_range
                upper = (i + 1) * complexity_range
                
                stratum_mask = (self.complexity_scores >= lower) & (self.complexity_scores < upper)
                stratum_indices = np.where(stratum_mask)[0]
                
                if len(stratum_indices) > 0:
                    n_samples = self.batch_size // n_strata
                    if i == n_strata - 1:
                        # Last stratum gets remaining samples
                        n_samples = self.batch_size - len(batch_indices)
                    
                    samples = np.random.choice(
                        stratum_indices,
                        size=min(n_samples, len(stratum_indices)),
                        replace=False
                    )
                    batch_indices.extend(samples.tolist())
        
        return batch_indices


class ProgressiveQuantizationScheduler:
    """
    Scheduler for progressive quantization during training.
    """
    
    def __init__(
        self,
        layer_groups: List[List[str]],
        initial_bits: int = 32,
        target_bits: int = 4,
        transition_epochs: List[int] = [30, 60, 90, 120]
    ):
        """
        Initialize progressive quantization scheduler.
        
        Args:
            layer_groups: Groups of layer names for staged quantization
            initial_bits: Starting bit-width
            target_bits: Target bit-width
            transition_epochs: Epochs at which to transition groups
        """
        self.layer_groups = layer_groups
        self.initial_bits = initial_bits
        self.target_bits = target_bits
        self.transition_epochs = transition_epochs
        
        # Initialize bit assignments
        self.current_bits = {
            layer: initial_bits
            for group in layer_groups
            for layer in group
        }
        
    def update(self, epoch: int) -> Dict[str, int]:
        """
        Update bit assignments based on epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Dictionary mapping layer names to bit-widths
        """
        for i, transition_epoch in enumerate(self.transition_epochs):
            if epoch >= transition_epoch:
                # Quantize corresponding layer group
                if i < len(self.layer_groups):
                    group = self.layer_groups[i]
                    for layer in group:
                        # Gradually reduce bits
                        progress = (epoch - transition_epoch) / 30  # 30 epochs transition
                        progress = min(progress, 1.0)
                        
                        current = self.initial_bits
                        target = self.target_bits
                        bits = int(current - (current - target) * progress)
                        
                        self.current_bits[layer] = bits
        
        return self.current_bits
    
    def get_quantized_layers(self, epoch: int) -> List[str]:
        """Get list of layers that should be quantized at current epoch."""
        self.update(epoch)
        return [
            layer for layer, bits in self.current_bits.items()
            if bits < self.initial_bits
        ]


class AdaptiveLearningRateScheduler:
    """
    Learning rate scheduler aware of quantization progression.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 0.001,
        quantization_lr_scale: float = 0.1,
        warmup_epochs: int = 10
    ):
        """
        Initialize adaptive learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            quantization_lr_scale: LR scale factor when quantizing
            warmup_epochs: Number of warmup epochs
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.quantization_lr_scale = quantization_lr_scale
        self.warmup_epochs = warmup_epochs
        
    def step(self, epoch: int, quantization_active: bool = False):
        """
        Update learning rate.
        
        Args:
            epoch: Current epoch
            quantization_active: Whether quantization is active
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (300 - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
            
            # Reduce LR when quantization is active
            if quantization_active:
                lr *= self.quantization_lr_scale
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]