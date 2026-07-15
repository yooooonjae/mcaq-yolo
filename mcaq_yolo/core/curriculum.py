"""
Curriculum Learning Module for MCAQ-YOLO

REVIEW CLEANUP: the unused ComplexityBasedSampler /
ProgressiveQuantizationScheduler / AdaptiveLearningRateScheduler classes and
the write-only *_history lists were removed — nothing in the pipeline
referenced them, and the LR scheduler hardcoded 300 epochs.
"""

import numpy as np
from typing import Dict


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
        warmup_epochs: int = 20,   # Paper Table X: Twarm = 20 (Stage 1 boundary)
        transition_epochs: int = 50,  # Paper Fig.3: Stage 2 ends at epoch 50
        total_epochs: int = 300,   # Paper Table X: total epochs = 300
        initial_complexity: float = 0.2,   # Paper: tau0 = 0.2
        initial_temperature: float = 10.0,  # Paper: initial temperature 10.0
        initial_bits: float = 8.0,
        target_bits: float = 4.0,
        curriculum_type: str = 'exponential',  # 'linear', 'exponential', 'cosine', 'step'
        lambda_smooth: float = 0.1  # Table X lambda2; scale down for finer grids
                                    # (Lsmooth sums |db| over tile pairs — a 20x20
                                    # grid has ~7x the terms of the paper's 8x8)
    ):
        """
        Initialize curriculum scheduler.

        Args:
            warmup_epochs: Number of warmup epochs (Stage 1: epochs 0-Twarm)
            transition_epochs: End of Stage 2 (transition; paper: epochs 20-50)
            total_epochs: Total training epochs (Stage 3 runs to the end)
            initial_complexity: Starting complexity threshold
            initial_temperature: Starting temperature
            initial_bits: Starting bit-width
            target_bits: Target bit-width
            curriculum_type: Type of curriculum progression
        """
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        self.total_epochs = total_epochs
        self.initial_complexity = initial_complexity
        self.initial_temperature = initial_temperature
        self.initial_bits = initial_bits
        self.target_bits = target_bits
        self.curriculum_type = curriculum_type
        self.lambda_smooth = lambda_smooth
        
        # Track current state (REVIEW FIX: the *_history lists were removed —
        # they were appended on EVERY getter call, so out-of-band calls such
        # as evaluate() polluted them, and nothing ever read them)
        self.current_epoch = 0

    def get_stage(self, epoch: int) -> int:
        """
        Three-stage curriculum schedule (paper Fig.3 / Sec IV-C):

          Stage 1 (epochs 0-Twarm):  warm-up — low-complexity samples only,
                                     high precision (FP16; quantization bypassed)
          Stage 2 (Twarm-transition): transition — mixed-complexity samples,
                                     dynamic bit allocation, temperature annealing
          Stage 3 (transition-end):  full MCAQ — all samples, aggressive quantization

        Returns:
            Stage number in {1, 2, 3}
        """
        if epoch <= self.warmup_epochs:
            return 1
        elif epoch <= self.transition_epochs:
            return 2
        return 3

    def get_complexity_threshold(self, epoch: int) -> float:
        """
        Get complexity threshold for current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Complexity threshold in [0, 1]
        """
        # Paper Algorithm 3 line 5: tau_t = tau0 + (1 - tau0) * t / Twarm for
        # t <= Twarm, then 1.0. NOTE: Fig.3's caption ("Stage 1 uses only
        # low-complexity samples") is the schematic description of early warm-up;
        # the precise spec — Sec IV-C: "the complexity threshold tau_t increases
        # linearly from tau0=0.2 to 1.0 during the warmup phase" — matches
        # Algorithm 3 and is what is implemented here.
        if epoch <= self.warmup_epochs:
            # Linear increase during warmup
            progress = epoch / self.warmup_epochs
            threshold = self.initial_complexity + (1.0 - self.initial_complexity) * progress
        else:
            # Full complexity after warmup
            threshold = 1.0

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
            # Paper Algorithm 3 (line 10) / Sec IV-C: alpha_t = 1 + 9 * exp(-5t/T)
            # initial_temperature controls the "+9" coefficient via (init - 1);
            # with init=10 and T=total_epochs this is exactly 1 + 9*exp(-5t/T).
            t = min(epoch, self.total_epochs)
            temperature = 1.0 + (self.initial_temperature - 1.0) * np.exp(
                -5.0 * t / max(1, self.total_epochs)
            )
            
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
        # Paper Eq.(20) + Table X loss weights:
        #   lambda1 (bit budget): annealed 0.01 -> 0.1 during training
        #   lambda2 (smoothness): 0.1   (constant)
        #   lambda3 (KD):         0.5   (constant)
        #   lambda4 (reg):        1e-4  (constant)
        progress = min(epoch / max(1, self.total_epochs), 1.0)
        lambda1 = 0.01 + (0.1 - 0.01) * progress  # annealed bit-budget weight

        # Smoothness ramp: zero during the high-precision
        # warm-up (no quantization -> nothing to smooth) and ramped across the
        # transition stage, so the flat-map reward cannot dominate before
        # Ldet's tile-wise signal exists.
        span = max(1, self.transition_epochs - self.warmup_epochs)
        ramp = min(1.0, max(0.0, (epoch - self.warmup_epochs) / span))

        weights = {
            'detection': 1.0,
            'bit_budget': lambda1,
            'smoothness': self.lambda_smooth * ramp,
            'distillation': 0.5,
            'regularization': 1e-4,
        }

        return weights
