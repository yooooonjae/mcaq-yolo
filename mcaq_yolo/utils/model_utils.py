"""
Model utility functions for MCAQ-YOLO
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: nn.Module, bits: int = 32) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * bits
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * bits
    
    size_mb = (param_size + buffer_size) / 8 / 1024 / 1024
    
    return size_mb


def profile_model(model: nn.Module, input_shape: tuple, device: str = 'cuda') -> Dict:
    """Profile model performance."""
    import time
    
    model.eval()
    model = model.to(device)
    
    # Dummy input
    x = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Time inference
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000  # ms
    fps = 1000 / avg_time
    
    return {
        'inference_time_ms': avg_time,
        'fps': fps
    }


def apply_weight_quantization(
    model: nn.Module,
    bits: int = 8,
    per_channel: bool = True
) -> nn.Module:
    """Apply simple weight quantization to model."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight'):
                weight = module.weight.data
                
                if per_channel:
                    # Per-channel quantization
                    weight_shape = weight.shape
                    weight = weight.reshape(weight_shape[0], -1)
                    
                    min_vals = weight.min(dim=1, keepdim=True)[0]
                    max_vals = weight.max(dim=1, keepdim=True)[0]
                    
                    scale = (max_vals - min_vals) / (2**bits - 1)
                    zero_point = -min_vals / scale
                    
                    weight_q = torch.round(weight / scale + zero_point)
                    weight_q = torch.clamp(weight_q, 0, 2**bits - 1)
                    weight_dq = (weight_q - zero_point) * scale
                    
                    module.weight.data = weight_dq.reshape(weight_shape)
                else:
                    # Per-tensor quantization
                    min_val = weight.min()
                    max_val = weight.max()
                    
                    scale = (max_val - min_val) / (2**bits - 1)
                    zero_point = -min_val / scale
                    
                    weight_q = torch.round(weight / scale + zero_point)
                    weight_q = torch.clamp(weight_q, 0, 2**bits - 1)
                    weight_dq = (weight_q - zero_point) * scale
                    
                    module.weight.data = weight_dq
    
    return model


def calibrate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 100
) -> Dict:
    """Calibrate model for quantization."""
    model.eval()
    
    activation_stats = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'mean': output.mean().item(),
                    'std': output.std().item()
                }
            else:
                activation_stats[name]['min'] = min(
                    activation_stats[name]['min'],
                    output.min().item()
                )
                activation_stats[name]['max'] = max(
                    activation_stats[name]['max'],
                    output.max().item()
                )
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run calibration
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                x = batch['img']
            else:
                x = batch[0]
            
            _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_stats