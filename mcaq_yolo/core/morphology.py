"""
Morphological Complexity Analysis Module
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from typing import Optional, Tuple
import hashlib


class MorphologicalComplexityAnalyzer(nn.Module):
    """
    Hierarchical Morphological Complexity Analyzer
    
    Computes five morphological metrics:
    1. Fractal dimension
    2. Texture entropy
    3. Gradient variance
    4. Edge density
    5. Contour complexity
    """
    
    def __init__(
        self,
        tile_sizes: list = [16, 32, 64],
        cache_size: int = 1000,
        device: str = 'cuda'
    ):
        """
        Initialize the morphological complexity analyzer.
        
        Args:
            tile_sizes: List of possible tile sizes for adaptive selection
            cache_size: Maximum number of cached complexity values
            device: Device to run computations on
        """
        super().__init__()
        self.tile_sizes = tile_sizes
        self.device = device
        self.cache = {}
        self.cache_size = cache_size
        
        # MLP for combining morphological features into complexity score
        self.complexity_mlp = nn.Sequential(
            nn.Linear(8, 64),  # 5 base features + 3 interaction terms
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Weights for feature combination
        self.feature_weights = nn.Parameter(torch.ones(5) / 5)
        self.interaction_weights = nn.Parameter(torch.ones(3) / 3)
        
    def fast_fractal_dimension(self, edge_map: np.ndarray) -> float:
        """
        Fast fractal dimension estimation using multi-resolution box counting.
        
        Args:
            edge_map: Binary edge map
            
        Returns:
            Fractal dimension value between 1.0 and 2.0
        """
        h, w = edge_map.shape
        min_dim = min(h, w)
        
        if min_dim < 4:
            return 1.0
        
        scales = [2**i for i in range(1, int(np.log2(min_dim)) + 1)]
        counts = []
        
        for s in scales:
            # Downsample using max pooling
            h_new, w_new = h // s, w // s
            if h_new == 0 or w_new == 0:
                continue
                
            pooled = cv2.resize(
                edge_map.astype(np.float32),
                (w_new, h_new),
                interpolation=cv2.INTER_MAX
            )
            n_boxes = np.sum(pooled > 0)
            if n_boxes > 0:
                counts.append((s, n_boxes))
        
        if len(counts) < 2:
            return 1.0
        
        # Weighted linear regression in log-log space
        scales_arr = np.array([c[0] for c in counts])
        counts_arr = np.array([c[1] for c in counts])
        
        log_scales = np.log(scales_arr)
        log_counts = np.log(counts_arr + 1)
        
        # Apply exponential weights (recent scales more important)
        weights = np.exp(-0.1 * np.arange(len(scales_arr)))
        
        # Calculate fractal dimension via weighted regression
        coef = np.polyfit(log_scales, log_counts, 1, w=weights)[0]
        df = -coef
        
        return np.clip(df, 1.0, 2.0)
    
    def compute_texture_entropy(self, tile: np.ndarray) -> float:
        """
        Compute texture entropy using Local Binary Patterns.
        
        Args:
            tile: Image tile (can be grayscale or color)
            
        Returns:
            Normalized texture entropy [0, 1]
        """
        # Convert to grayscale if needed
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile
        
        # Compute LBP
        radius = 1
        n_points = 8
        lbp = local_binary_pattern(gray, P=n_points, R=radius, method='uniform')
        
        # Compute histogram
        n_bins = n_points + 2  # Number of uniform patterns + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, density=True)
        
        # Calculate entropy
        hist = hist + 1e-10  # Avoid log(0)
        texture_entropy = entropy(hist, base=2)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(n_bins)
        return texture_entropy / max_entropy
    
    def compute_gradient_variance(self, tile: np.ndarray) -> float:
        """
        Compute gradient variance as a measure of local contrast.
        
        Args:
            tile: Image tile
            
        Returns:
            Normalized gradient variance [0, 1]
        """
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize variance by mean (coefficient of variation)
        mean_grad = np.mean(grad_mag)
        if mean_grad > 0:
            cv = np.var(grad_mag) / (mean_grad + 1e-10)
            return np.tanh(cv)  # Squash to [0, 1]
        return 0.0
    
    def compute_edge_density(self, tile: np.ndarray) -> float:
        """
        Compute edge density using Canny edge detection.
        
        Args:
            tile: Image tile
            
        Returns:
            Edge density [0, 1]
        """
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Canny edge detection with automatic thresholds
        median_val = np.median(blurred)
        lower = int(max(0, 0.7 * median_val))
        upper = int(min(255, 1.3 * median_val))
        edges = cv2.Canny(blurred, lower, upper)
        
        # Calculate edge density
        return np.sum(edges > 0) / edges.size
    
    def compute_contour_complexity(self, tile: np.ndarray) -> float:
        """
        Compute contour complexity based on shape descriptors.
        
        Args:
            tile: Image tile
            
        Returns:
            Normalized contour complexity [0, 1]
        """
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile
        
        # Adaptive threshold for better contour detection
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        # Compute complexity metrics for each contour
        complexities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Filter out noise
                perimeter = cv2.arcLength(contour, True)
                if area > 0:
                    # Circularity: 4π × area / perimeter²
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Complexity is inverse of circularity
                    complexity = 1.0 - circularity
                    complexities.append(complexity)
        
        if complexities:
            return np.mean(complexities)
        return 0.0
    
    def get_tile_hash(self, tile: torch.Tensor) -> str:
        """Generate hash for tile caching."""
        tile_np = tile.detach().cpu().numpy()
        return hashlib.md5(tile_np.tobytes()).hexdigest()
    
    def bilateral_filter(
        self,
        complexity_map: torch.Tensor,
        sigma_spatial: float = 2.0,
        sigma_range: float = 0.1
    ) -> torch.Tensor:
        """
        Apply bilateral filter for spatial smoothness.
        
        Args:
            complexity_map: Complexity map tensor
            sigma_spatial: Spatial sigma for Gaussian kernel
            sigma_range: Range sigma for intensity similarity
            
        Returns:
            Filtered complexity map
        """
        # Create Gaussian kernel for spatial filtering
        kernel_size = 5
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        x = x.to(complexity_map.device)
        
        gaussian_1d = torch.exp(-(x ** 2) / (2 * sigma_spatial ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply Gaussian smoothing
        B = complexity_map.shape[0]
        smoothed = []
        
        for b in range(B):
            x = complexity_map[b:b+1].unsqueeze(1)
            smooth = F.conv2d(x, gaussian_2d, padding=kernel_size//2)
            smoothed.append(smooth.squeeze(1))
        
        return torch.stack(smoothed)
    
    def forward(
        self,
        features: torch.Tensor,
        return_detailed: bool = False
    ) -> torch.Tensor:
        """
        Compute morphological complexity for feature maps.
        
        Args:
            features: Tensor of shape (B, C, H, W)
            return_detailed: If True, return detailed metrics
            
        Returns:
            complexity_map: Tensor of shape (B, H/s, W/s) with values in [0, 1]
            detailed_metrics: Dict of individual metrics (if return_detailed=True)
        """
        B, C, H, W = features.shape
        
        # Adaptive tile size selection
        if H > 256:
            tile_size = self.tile_sizes[2]  # 64
        elif H > 128:
            tile_size = self.tile_sizes[1]  # 32
        else:
            tile_size = self.tile_sizes[0]  # 16
        
        # Calculate number of tiles
        h_tiles = H // tile_size
        w_tiles = W // tile_size
        
        # Initialize output tensors
        complexity_map = torch.zeros(B, h_tiles, w_tiles).to(features.device)
        
        if return_detailed:
            detailed_metrics = {
                'fractal': torch.zeros(B, h_tiles, w_tiles),
                'texture': torch.zeros(B, h_tiles, w_tiles),
                'gradient': torch.zeros(B, h_tiles, w_tiles),
                'edge': torch.zeros(B, h_tiles, w_tiles),
                'contour': torch.zeros(B, h_tiles, w_tiles)
            }
        
        for b in range(B):
            for i in range(h_tiles):
                for j in range(w_tiles):
                    # Extract tile
                    tile = features[b, :,
                                  i*tile_size:(i+1)*tile_size,
                                  j*tile_size:(j+1)*tile_size]
                    
                    # Check cache
                    tile_hash = self.get_tile_hash(tile)
                    if tile_hash in self.cache:
                        complexity_map[b, i, j] = self.cache[tile_hash]
                        continue
                    
                    # Convert to numpy for morphological analysis
                    tile_np = tile.mean(0).cpu().numpy()  # Average over channels
                    tile_np = (tile_np * 255).astype(np.uint8)
                    
                    # Compute morphological features
                    edges = cv2.Canny(tile_np, 50, 150)
                    φ1 = self.fast_fractal_dimension(edges)
                    φ2 = self.compute_texture_entropy(tile_np)
                    φ3 = self.compute_gradient_variance(tile_np)
                    φ4 = self.compute_edge_density(tile_np)
                    φ5 = self.compute_contour_complexity(tile_np)
                    
                    # Normalize features
                    φ1_norm = φ1 / 2.0
                    φ2_norm = φ2  # Already normalized
                    φ3_norm = φ3  # Already normalized
                    φ4_norm = φ4  # Already normalized
                    φ5_norm = φ5  # Already normalized
                    
                    if return_detailed:
                        detailed_metrics['fractal'][b, i, j] = φ1_norm
                        detailed_metrics['texture'][b, i, j] = φ2_norm
                        detailed_metrics['gradient'][b, i, j] = φ3_norm
                        detailed_metrics['edge'][b, i, j] = φ4_norm
                        detailed_metrics['contour'][b, i, j] = φ5_norm
                    
                    # Create feature vector with interaction terms
                    features_vec = torch.tensor([
                        φ1_norm,
                        φ2_norm,
                        φ3_norm,
                        φ4_norm,
                        φ5_norm,
                        φ1_norm * φ2_norm,  # Boundary-texture interaction
                        φ3_norm ** 2,        # Non-linear gradient response
                        np.sqrt(φ4_norm * φ5_norm)  # Edge-contour coupling
                    ], device=self.device).unsqueeze(0)
                    
                    # Compute complexity score using MLP
                    with torch.no_grad():
                        complexity_score = self.complexity_mlp(features_vec).squeeze()
                    
                    complexity_map[b, i, j] = complexity_score
                    
                    # Update cache
                    if len(self.cache) < self.cache_size:
                        self.cache[tile_hash] = complexity_score.item()
        
        # Apply bilateral filter for smoothness
        complexity_map = self.bilateral_filter(complexity_map)
        
        if return_detailed:
            return complexity_map, detailed_metrics
        return complexity_map