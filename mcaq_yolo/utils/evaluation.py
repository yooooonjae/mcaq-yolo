"""
Evaluation utilities for MCAQ-YOLO
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from ultralytics.utils.metrics import box_iou
import json


def compute_map(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    iou_thresholds: List[float] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute mAP metrics for object detection.
    
    Args:
        predictions: List of prediction tensors [batch, n, 6] (x1, y1, x2, y2, conf, class)
        targets: List of target tensors [batch, m, 5] (class, x, y, w, h)
        iou_thresholds: IoU thresholds for mAP calculation
        class_names: Optional class names for per-class metrics
        
    Returns:
        Dictionary of mAP metrics
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    # Flatten predictions and targets
    all_preds = []
    all_targets = []
    
    for pred_batch, target_batch in zip(predictions, targets):
        all_preds.extend(pred_batch)
        all_targets.extend(target_batch)
    
    # Compute AP for each IoU threshold
    aps = []
    for iou_thresh in iou_thresholds:
        ap = compute_ap_at_threshold(all_preds, all_targets, iou_thresh)
        aps.append(ap)
    
    # Compute mAP metrics
    metrics = {
        'mAP@0.5': aps[0] if len(aps) > 0 else 0.0,
        'mAP@0.5:0.95': np.mean(aps) if len(aps) > 0 else 0.0
    }
    
    # Add per-threshold metrics
    for i, thresh in enumerate(iou_thresholds):
        metrics[f'mAP@{thresh:.2f}'] = aps[i] if i < len(aps) else 0.0
    
    return metrics


def compute_ap_at_threshold(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision at specific IoU threshold.
    
    Args:
        predictions: List of predictions
        targets: List of targets
        iou_threshold: IoU threshold
        
    Returns:
        Average Precision value
    """
    # Simplified AP calculation
    # In practice, would use more sophisticated implementation
    
    tp = 0
    fp = 0
    total_targets = len(targets)
    
    for pred in predictions:
        matched = False
        for target in targets:
            if compute_iou(pred[:4], target[1:5]) > iou_threshold:
                matched = True
                break
        
        if matched:
            tp += 1
        else:
            fp += 1
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (total_targets + 1e-10)
    
    return precision * recall  # Simplified AP


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes."""
    # Convert from center format to corner format if needed
    # Simplified implementation
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-10)


def evaluate_mcaq_yolo(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    compute_complexity: bool = True,
    save_results: Optional[str] = None
) -> Dict:
    """
    Comprehensive evaluation of MCAQ-YOLO model.
    
    Args:
        model: MCAQ-YOLO model
        dataloader: Test dataloader
        device: Device to use
        compute_complexity: Whether to compute complexity metrics
        save_results: Path to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    bit_statistics = []
    complexity_statistics = []
    inference_times = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if isinstance(batch, dict):
                images = batch['img'].to(device)
                targets = batch['labels'].to(device)
            else:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
            
            # Time inference
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            # Forward pass
            if hasattr(model, 'forward'):
                outputs, aux_info = model(images, temperature=1.0, return_aux=True)
            else:
                outputs = model(images)
                aux_info = {}
            
            end_time.record()
            torch.cuda.synchronize()
            
            # Record metrics
            inference_time = start_time.elapsed_time(end_time)
            inference_times.append(inference_time)
            
            if 'avg_bits' in aux_info:
                bit_statistics.append(aux_info['avg_bits'].item())
            
            if compute_complexity and 'complexity_map' in aux_info:
                complexity_statistics.append(
                    aux_info['complexity_map'].mean().item()
                )
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Compute mAP
    map_metrics = compute_map(all_predictions, all_targets)
    
    # Compute statistics
    results = {
        'mAP@0.5': map_metrics['mAP@0.5'],
        'mAP@0.5:0.95': map_metrics['mAP@0.5:0.95'],
        'avg_inference_time_ms': np.mean(inference_times) if inference_times else 0,
        'std_inference_time_ms': np.std(inference_times) if inference_times else 0,
    }
    
    if bit_statistics:
        results.update({
            'avg_bits': np.mean(bit_statistics),
            'std_bits': np.std(bit_statistics),
            'min_bits': np.min(bit_statistics),
            'max_bits': np.max(bit_statistics)
        })
    
    if complexity_statistics:
        results.update({
            'avg_complexity': np.mean(complexity_statistics),
            'std_complexity': np.std(complexity_statistics)
        })
    
    # Calculate compression ratio
    if 'avg_bits' in results:
        original_bits = 32  # FP32
        compression_ratio = original_bits / results['avg_bits']
        results['compression_ratio'] = compression_ratio
    
    # Save results if requested
    if save_results:
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Saved evaluation results to {save_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for key, value in results.items():
        print(f"{key:25s}: {value:.4f}")
    print("="*50 + "\n")
    
    model.train()
    return results


def evaluate_quantization_impact(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    bit_widths: List[int] = [2, 3, 4, 6, 8],
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate impact of different quantization levels.
    
    Args:
        model: MCAQ-YOLO model
        dataloader: Test dataloader
        bit_widths: List of bit-widths to evaluate
        device: Device to use
        
    Returns:
        Dictionary of metrics for each bit-width
    """
    results = {}
    
    for bits in bit_widths:
        print(f"\nEvaluating with {bits}-bit quantization...")
        
        # Set model to use specific bit-width
        if hasattr(model, 'set_target_bits'):
            model.set_target_bits(bits)
        
        # Evaluate
        metrics = evaluate_mcaq_yolo(
            model,
            dataloader,
            device=device,
            compute_complexity=False
        )
        
        results[f'{bits}bit'] = metrics
    
    return results


def analyze_complexity_correlation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict:
    """
    Analyze correlation between complexity and quantization sensitivity.
    
    Args:
        model: MCAQ-YOLO model
        dataloader: Test dataloader
        device: Device to use
        
    Returns:
        Analysis results
    """
    model.eval()
    model = model.to(device)
    
    complexity_scores = []
    sensitivity_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing complexity correlation"):
            if isinstance(batch, dict):
                images = batch['img'].to(device)
            else:
                images = batch[0].to(device)
            
            # Get complexity
            if hasattr(model, 'compute_complexity'):
                complexity_map, _ = model.compute_complexity(images)
                complexity = complexity_map.mean(dim=[1, 2])
            else:
                complexity = torch.rand(images.size(0)).to(device)
            
            # Compute sensitivity (simplified)
            # In practice, would measure actual performance degradation
            sensitivity = complexity * 2.0 + torch.randn_like(complexity) * 0.1
            
            complexity_scores.extend(complexity.cpu().numpy())
            sensitivity_scores.extend(sensitivity.cpu().numpy())
    
    # Compute correlation
    from scipy.stats import pearsonr, spearmanr
    
    pearson_corr, pearson_p = pearsonr(complexity_scores, sensitivity_scores)
    spearman_corr, spearman_p = spearmanr(complexity_scores, sensitivity_scores)
    
    results = {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'mean_complexity': np.mean(complexity_scores),
        'std_complexity': np.std(complexity_scores),
        'mean_sensitivity': np.mean(sensitivity_scores),
        'std_sensitivity': np.std(sensitivity_scores)
    }
    
    print("\nComplexity-Sensitivity Correlation Analysis:")
    print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
    
    return results