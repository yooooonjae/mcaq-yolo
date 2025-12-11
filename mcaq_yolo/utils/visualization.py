"""
Visualization utilities for MCAQ-YOLO
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import cv2
from pathlib import Path


def visualize_complexity_map(
    image: np.ndarray,
    complexity_map: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Morphological Complexity Map"
) -> plt.Figure:
    """
    Visualize morphological complexity map overlaid on image.
    
    Args:
        image: Original image (H, W, 3)
        complexity_map: Complexity map tensor (H', W')
        save_path: Path to save figure
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Complexity map
    complexity_np = complexity_map.cpu().numpy() if torch.is_tensor(complexity_map) else complexity_map
    im1 = axes[1].imshow(complexity_np, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Complexity Map")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Overlay
    # Resize complexity map to match image size
    if complexity_np.shape != image.shape[:2]:
        complexity_resized = cv2.resize(
            complexity_np,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        complexity_resized = complexity_np
    
    # Create overlay
    overlay = image.copy()
    heatmap = plt.cm.hot(complexity_resized)[:, :, :3]
    overlay = cv2.addWeighted(overlay, 0.7, (heatmap * 255).astype(np.uint8), 0.3, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Complexity Overlay")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_bit_allocation(
    image: np.ndarray,
    bit_map: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Spatial Bit Allocation"
) -> plt.Figure:
    """
    Visualize spatial bit allocation map.
    
    Args:
        image: Original image
        bit_map: Bit allocation map
        save_path: Path to save figure
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Bit allocation map
    bit_np = bit_map.cpu().numpy() if torch.is_tensor(bit_map) else bit_map
    im1 = axes[1].imshow(bit_np, cmap='viridis', vmin=2, vmax=8)
    axes[1].set_title("Bit Allocation Map")
    axes[1].axis('off')
    
    # Add colorbar with bit values
    cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046)
    cbar.set_label('Bits', rotation=270, labelpad=15)
    
    # Histogram of bit allocations
    axes[2].hist(bit_np.flatten(), bins=np.arange(2, 9), edgecolor='black')
    axes[2].set_xlabel("Bit Width")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Bit Distribution")
    axes[2].grid(True, alpha=0.3)
    
    # Add mean line
    mean_bits = bit_np.mean()
    axes[2].axvline(mean_bits, color='r', linestyle='--', label=f'Mean: {mean_bits:.2f}')
    axes[2].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves for MCAQ-YOLO.
    
    Args:
        history: Dictionary of training metrics
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Determine number of subplots needed
    metrics = list(history.keys())
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, (metric_name, values) in enumerate(history.items()):
        if idx < len(axes):
            ax = axes[idx]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, 'b-', label=metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over Training')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(values) > 10:
                z = np.polyfit(epochs, values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "r--", alpha=0.5, label='Trend')
            
            ax.legend()
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("MCAQ-YOLO Training Progress")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_complexity_vs_performance(
    complexity_scores: np.ndarray,
    performance_drops: np.ndarray,
    bit_widths: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize relationship between complexity and performance.
    
    Args:
        complexity_scores: Array of complexity scores
        performance_drops: Array of performance degradation values
        bit_widths: Array of allocated bit-widths
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot: Complexity vs Performance Drop
    scatter = axes[0].scatter(
        complexity_scores,
        performance_drops,
        c=bit_widths,
        cmap='viridis',
        alpha=0.6
    )
    axes[0].set_xlabel("Morphological Complexity")
    axes[0].set_ylabel("Performance Drop (%)")
    axes[0].set_title("Complexity vs Performance Impact")
    plt.colorbar(scatter, ax=axes[0], label='Bits')
    
    # Fit and plot trend line
    z = np.polyfit(complexity_scores, performance_drops, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(complexity_scores.min(), complexity_scores.max(), 100)
    axes[0].plot(x_trend, p(x_trend), 'r--', alpha=0.7, label='Trend')
    axes[0].legend()
    
    # Complexity vs Bit Allocation
    axes[1].scatter(complexity_scores, bit_widths, alpha=0.6)
    axes[1].set_xlabel("Morphological Complexity")
    axes[1].set_ylabel("Allocated Bits")
    axes[1].set_title("Complexity-Aware Bit Allocation")
    
    # Fit and plot allocation curve
    z_bits = np.polyfit(complexity_scores, bit_widths, 2)
    p_bits = np.poly1d(z_bits)
    axes[1].plot(x_trend, p_bits(x_trend), 'g--', alpha=0.7, label='Allocation Policy')
    axes[1].legend()
    
    # Heatmap: Complexity bins vs Performance
    complexity_bins = np.percentile(complexity_scores, [0, 25, 50, 75, 100])
    bit_bins = [2, 3, 4, 6, 8]
    
    # Create 2D histogram
    heatmap_data = np.zeros((len(bit_bins)-1, len(complexity_bins)-1))
    
    for i in range(len(bit_bins)-1):
        for j in range(len(complexity_bins)-1):
            mask = (bit_widths >= bit_bins[i]) & (bit_widths < bit_bins[i+1]) & \
                   (complexity_scores >= complexity_bins[j]) & (complexity_scores < complexity_bins[j+1])
            if mask.sum() > 0:
                heatmap_data[i, j] = performance_drops[mask].mean()
    
    im = axes[2].imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    axes[2].set_xlabel("Complexity Quartile")
    axes[2].set_ylabel("Bit Width")
    axes[2].set_title("Performance Impact Heatmap")
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[2].set_yticks(range(4))
    axes[2].set_yticklabels(['2-3', '3-4', '4-6', '6-8'])
    plt.colorbar(im, ax=axes[2], label='Avg Performance Drop (%)')
    
    plt.suptitle("Morphological Complexity Analysis")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_report(
    model_stats: Dict,
    save_path: str
):
    """
    Create a comprehensive summary report with visualizations.
    
    Args:
        model_stats: Dictionary of model statistics and metrics
        save_path: Path to save the report
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid spec
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. mAP comparison
    ax1 = fig.add_subplot(gs[0, :2])
    methods = list(model_stats.get('map_comparison', {}).keys())
    maps = list(model_stats.get('map_comparison', {}).values())
    bars = ax1.bar(methods, maps, color=['blue', 'green', 'orange', 'red'])
    ax1.set_ylabel('mAP@0.5')
    ax1.set_title('Detection Performance Comparison')
    ax1.set_ylim([0, 100])
    
    # Add value labels on bars
    for bar, val in zip(bars, maps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center')
    
    # 2. Compression vs Accuracy trade-off
    ax2 = fig.add_subplot(gs[0, 2:])
    if 'compression_tradeoff' in model_stats:
        data = model_stats['compression_tradeoff']
        ax2.plot(data['compression_ratios'], data['map_values'], 'o-', markersize=8)
        ax2.set_xlabel('Compression Ratio')
        ax2.set_ylabel('mAP@0.5 (%)')
        ax2.set_title('Compression vs Accuracy Trade-off')
        ax2.grid(True, alpha=0.3)
    
    # 3. Training curves
    ax3 = fig.add_subplot(gs[1, :2])
    if 'training_loss' in model_stats:
        epochs = range(1, len(model_stats['training_loss']) + 1)
        ax3.plot(epochs, model_stats['training_loss'], label='Training')
        if 'validation_loss' in model_stats:
            ax3.plot(epochs, model_stats['validation_loss'], label='Validation')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Bit distribution
    ax4 = fig.add_subplot(gs[1, 2:])
    if 'bit_distribution' in model_stats:
        bit_dist = model_stats['bit_distribution']
        ax4.bar(bit_dist['bits'], bit_dist['counts'], color='purple', edgecolor='black')
        ax4.set_xlabel('Bit Width')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Bit Width Distribution')
        ax4.axvline(bit_dist.get('mean', 4), color='red', linestyle='--',
                   label=f"Mean: {bit_dist.get('mean', 4):.2f}")
        ax4.legend()
    
    # 5. Complexity distribution
    ax5 = fig.add_subplot(gs[2, :2])
    if 'complexity_distribution' in model_stats:
        complexity = model_stats['complexity_distribution']
        ax5.hist(complexity, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Morphological Complexity')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Dataset Complexity Distribution')
        ax5.axvline(np.mean(complexity), color='red', linestyle='--',
                   label=f'Mean: {np.mean(complexity):.3f}')
        ax5.legend()
    
    # 6. Performance metrics table
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('tight')
    ax6.axis('off')
    
    if 'metrics_table' in model_stats:
        table_data = model_stats['metrics_table']
        table = ax6.table(cellText=table_data['values'],
                         rowLabels=table_data['rows'],
                         colLabels=table_data['columns'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    
    ax6.set_title('Performance Metrics Summary')
    
    # Overall title
    fig.suptitle('MCAQ-YOLO Evaluation Report', fontsize=16, fontweight='bold')
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary report to {save_path}")
    
    return fig