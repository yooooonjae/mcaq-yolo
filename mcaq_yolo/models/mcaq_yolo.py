"""
MCAQ-YOLO Main Model Implementation (Fixed Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from typing import Dict, Tuple, Optional, List
import warnings
import copy

from ..core.morphology import MorphologicalComplexityAnalyzer
from ..core.bit_allocation import ComplexityToBitMappingNetwork, AdaptiveBitAllocation
from ..core.quantization import SpatialAdaptiveQuantization, MixedPrecisionQuantizer
from ..core.curriculum import CurriculumScheduler


class MCQLYOLOLoss(nn.Module):
    """
    Combined loss function for MCAQ-YOLO with proper YOLO detection loss.
    """
    
    def __init__(
        self,
        model,  # Add model parameter for YOLO loss
        target_bits: float = 4.0,
        lambda_bit: float = 0.01,
        lambda_smooth: float = 0.001,
        lambda_kd: float = 0.5,
        lambda_reg: float = 0.0001
    ):
        super().__init__()
        self.target_bits = target_bits
        self.lambda_bit = lambda_bit
        self.lambda_smooth = lambda_smooth
        self.lambda_kd = lambda_kd
        self.lambda_reg = lambda_reg
        
        # Initialize YOLO detection loss
        self.detection_loss = v8DetectionLoss(model)
        
    def compute_smoothness_loss(self, bit_map: torch.Tensor) -> torch.Tensor:
        """Compute spatial smoothness loss for bit allocation."""
        if bit_map.dim() == 3:
            dx = torch.abs(bit_map[:, 1:, :] - bit_map[:, :-1, :])
            dy = torch.abs(bit_map[:, :, 1:] - bit_map[:, :, :-1])
        else:
            dx = torch.abs(bit_map[1:, :] - bit_map[:-1, :])
            dy = torch.abs(bit_map[:, 1:] - bit_map[:, :-1])
        
        return (dx.mean() + dy.mean()) / 2.0
    
    def compute_bit_budget_loss(
        self,
        avg_bits: torch.Tensor,
        target_bits: Optional[float] = None
    ) -> torch.Tensor:
        """Compute bit budget constraint loss."""
        if target_bits is None:
            target_bits = self.target_bits
        
        diff = avg_bits - target_bits
        if diff > 0:
            loss = 2.0 * (diff ** 2)
        else:
            loss = diff ** 2
        
        return loss
    
    def forward(
        self,
        outputs: torch.Tensor,
        batch: Dict,  # Changed from targets to batch dict
        aux_info: Dict,
        teacher_outputs: Optional[torch.Tensor] = None,
        model_params: Optional[nn.Module] = None,
        loss_weights: Optional[Dict[str, float]] = None
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
            loss_det, loss_items = self.detection_loss(outputs, batch)
            loss_dict['loss_det'] = loss_det
            # Add individual loss components
            loss_dict.update({
                'box_loss': loss_items[0] if len(loss_items) > 0 else 0,
                'cls_loss': loss_items[1] if len(loss_items) > 1 else 0,
                'dfl_loss': loss_items[2] if len(loss_items) > 2 else 0,
            })
        else:
            # Fallback to simple MSE
            targets = batch.get('labels', batch)
            loss_det = F.mse_loss(outputs, targets)
            loss_dict['loss_det'] = loss_det
        
        # Bit budget loss
        if 'avg_bits' in aux_info:
            loss_bit = self.compute_bit_budget_loss(aux_info['avg_bits'])
            loss_dict['loss_bit'] = loss_bit
        else:
            loss_bit = torch.tensor(0.0, device=outputs.device)
        
        # Smoothness loss
        if 'bit_map' in aux_info:
            loss_smooth = self.compute_smoothness_loss(aux_info['bit_map'])
            loss_dict['loss_smooth'] = loss_smooth
        else:
            loss_smooth = torch.tensor(0.0, device=outputs.device)
        
        # Knowledge distillation loss
        if teacher_outputs is not None:
            # Use KL divergence for better distillation
            T = 4.0  # Temperature
            loss_kd = F.kl_div(
                F.log_softmax(outputs / T, dim=-1),
                F.softmax(teacher_outputs / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)
            loss_dict['loss_kd'] = loss_kd
        else:
            loss_kd = torch.tensor(0.0, device=outputs.device)
        
        # Regularization loss
        if model_params is not None:
            loss_reg = torch.tensor(0.0, device=outputs.device)
            for param in model_params.parameters():
                if param.requires_grad:
                    loss_reg += torch.norm(param, 2)
            loss_dict['loss_reg'] = loss_reg
        else:
            loss_reg = torch.tensor(0.0, device=outputs.device)
        
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
        num_classes: int = 80  # Added num_classes parameter
    ):
        super().__init__()
        
        # Load base YOLOv8 model
        if pretrained:
            self.base_model = YOLO(f'{model_name}.pt')
        else:
            self.base_model = YOLO(model_name)
            
        self.model = self.base_model.model.to(device)
        self.num_classes = num_classes
        
        # Initialize components
        self.complexity_analyzer = MorphologicalComplexityAnalyzer(
            tile_sizes=[16, 32, 64],
            cache_size=1000,
            device=device
        )
        
        self.bit_mapper = ComplexityToBitMappingNetwork(
            min_bits=min_bits,
            max_bits=max_bits,
            hidden_dims=[128, 64, 32]
        ).to(device)
        
        self.quantizer = SpatialAdaptiveQuantization(
            calibration_mode='percentile',
            smooth_transitions=True,
            per_channel=True
        ).to(device)
        
        self.mixed_quantizer = MixedPrecisionQuantizer(
            weight_quant=True,
            activation_quant=True,
            hardware_type='gpu'
        ).to(device)
        
        # Initialize loss function with model
        self.loss_fn = MCQLYOLOLoss(
            model=self.model,
            target_bits=target_bits,
            lambda_bit=0.01,
            lambda_smooth=0.001,
            lambda_kd=0.5,
            lambda_reg=0.0001
        )
        
        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(
            warmup_epochs=10,
            total_epochs=300,
            initial_complexity=0.2,
            initial_temperature=10.0
        )
        
        # Training parameters
        self.device = device
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.target_bits = target_bits
        
        # Feature extraction hooks
        self.feature_hooks = []
        self.features = {}
        
        # Store original model for teacher
        self.teacher_model = None
        
    def setup_teacher(self):
        """Setup teacher model for distillation."""
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        """Register hooks for feature extraction."""
        def get_hook(name):
            def hook_fn(module, input, output):
                self.features[name] = output
            return hook_fn
        
        # Target specific YOLO layers
        target_layers = []
        for name, module in self.model.named_modules():
            if 'cv1' in name or 'cv2' in name or 'cv3' in name:
                if len(target_layers) < 3:
                    target_layers.append(name)
                    hook = module.register_forward_hook(get_hook(name))
                    self.feature_hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.feature_hooks:
            hook.remove()
        self.feature_hooks = []
        self.features = {}
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate features from the model."""
        self._register_hooks()
        
        with torch.no_grad():
            _ = self.model(x)
        
        features = list(self.features.values())
        self._remove_hooks()
        
        # If no features extracted, use input
        if not features:
            features = [x]
        
        return features
    
    def compute_complexity(
        self,
        x: torch.Tensor,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute morphological complexity for input."""
        features = self.extract_features(x)
        
        complexity_maps = []
        detailed_metrics = []
        
        for feat in features:
            if feat.shape[2] < 8 or feat.shape[3] < 8:
                # Skip too small feature maps
                continue
                
            complexity, metrics = self.complexity_analyzer(
                feat,
                return_detailed=True
            )
            complexity_maps.append(complexity)
            detailed_metrics.append(metrics)
        
        if not complexity_maps:
            # Fallback to simple complexity
            B, C, H, W = x.shape
            complexity_maps = [torch.rand(B, H//32, W//32).to(x.device)]
            detailed_metrics = [{}]
        
        # Average across feature levels
        avg_complexity = torch.stack(complexity_maps).mean(0)
        
        # Aggregate detailed metrics
        aggregated_metrics = {}
        if detailed_metrics and detailed_metrics[0]:
            for key in detailed_metrics[0].keys():
                aggregated_metrics[key] = torch.stack(
                    [m[key] for m in detailed_metrics if key in m]
                ).mean(0)
        
        return avg_complexity, aggregated_metrics
    
    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        return_aux: bool = True,
        targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with adaptive quantization.
        
        Args:
            x: Input images (B, 3, H, W)
            temperature: Temperature for bit allocation
            return_aux: Whether to return auxiliary information
            targets: Target labels for training
            
        Returns:
            outputs: Detection outputs
            aux_info: Auxiliary information (if return_aux=True)
        """
        # Compute morphological complexity
        complexity, detailed_metrics = self.compute_complexity(x)
        
        # Allocate bits based on complexity
        bit_map = self.allocate_bits(complexity, temperature)
        
        # [Modified] Apply spatial adaptive quantization to the input image
        # This will utilize the CUDA kernel if available for high-performance inference
        x_quantized = self.quantizer(x, bit_map)
        
        # Forward pass through model using quantized input
        outputs = self.model(x_quantized)
        
        if return_aux:
            aux_info = {
                'complexity_map': complexity,
                'bit_map': bit_map,
                'avg_bits': bit_map.mean(),
                'detailed_metrics': detailed_metrics
            }
            return outputs, aux_info
        
        return outputs
    
    def allocate_bits(
        self,
        complexity: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Allocate bits based on complexity."""
        return self.bit_mapper(complexity, temperature, return_continuous=False)