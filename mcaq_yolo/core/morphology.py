"""
Morphological Complexity Analysis Module
"""

import math

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
        grid_size: int = 8,        # Paper Sec IV-D: tile size defaults to H/8 (8x8 grid)
        cache_size: int = 10000,   # Paper Table X: cache size 10,000 entries
        device: str = "cuda",
        metric_backend: str = "gpu",  # 'gpu' (vectorized surrogates) | 'cv2' (exact Eq.21-24)
    ):
        """
        Initialize the morphological complexity analyzer.

        Args:
            grid_size: Number of tiles per spatial dimension (paper default 8x8 grid;
                       finer 16x16 grids may be used for high object density, Eq.12)
            cache_size: Maximum number of cached complexity values
            device: Device to run computations on
            metric_backend:
                'cv2' — exact per-tile metrics via OpenCV (Canny+Otsu edge maps,
                        per-contour Eq.24). CPU-bound; matches the paper's
                        calibration-time analysis exactly. Used for offline
                        dataset scoring and calibration.
                'gpu' — vectorized tensor surrogates for per-batch training use
                        (Otsu-thresholded Sobel magnitude approximates Canny;
                        tile-level single-region circularity approximates the
                        per-contour Eq.24). Documented approximation.
        """
        super().__init__()
        self.grid_size = grid_size
        self.device = device
        self.metric_backend = metric_backend
        self.cache = {}
        self.cache_size = cache_size

        # Algorithm 1 line 14-15: phi = [phi1..phi5, phi1*phi2, phi3^2, phi4*phi5]; C <- MLP(phi)
        # The 8-D feature already carries the Eq.(9) interaction terms (beta12, beta33, beta45
        # are absorbed as learnable weights of this MLP) — do NOT add a separate interaction
        # module on top, that would double-count Eq.(9).
        # LayerNorm instead of BatchNorm (Codex review #5): after
        # phi.reshape(-1, 8) a BatchNorm would share statistics across the
        # mixed tile/image/scale batch, which at small training scales
        # compresses the C range globally (measured: C in [0.009, 0.026]).
        # The morphology MLP's architecture is not fixed by the paper.
        self.complexity_mlp = nn.Sequential(
            nn.Linear(8, 64),  # 5 base features + 3 interaction terms
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        ).to(device)
        # Wider final-layer init (paper leaves the complexity MLP's init
        # unspecified): with the default small weights the sigmoid output
        # collapses to ~0.5 +- 0.05 for many epochs, handing the bit mapper a
        # near-constant input and producing spatially flat bit maps early in
        # training (measured: C in [0.40, 0.50] after 3 epochs).
        nn.init.xavier_uniform_(self.complexity_mlp[-2].weight, gain=3.0)
        nn.init.zeros_(self.complexity_mlp[-2].bias)

        # Eq.(8) weights alpha_i (sum=1, >=0 enforced at use site). Used for the
        # deterministic dataset-scoring path (Algorithm 3 line 1 SortByComplexity);
        # the training-time score itself comes from the MLP (Algorithm 1 line 15).
        self.feature_weights = nn.Parameter(torch.ones(5) / 5)

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
            # Downsample using pooling-like resize
            h_new, w_new = h // s, w // s
            if h_new <= 0 or w_new <= 0:
                continue

            pooled = cv2.resize(
                edge_map.astype(np.float32),
                (w_new, h_new),
                interpolation=cv2.INTER_AREA,  # ✅ 유효한 상수
            )
            n_boxes = np.sum(pooled > 0)
            if n_boxes > 0:
                counts.append((s, n_boxes))

        if len(counts) < 2:
            return 1.0

        scales_arr = np.array([c[0] for c in counts])
        counts_arr = np.array([c[1] for c in counts])

        log_scales = np.log(scales_arr)
        log_counts = np.log(counts_arr + 1)

        # Apply exponential weights (recent scales more important)
        weights = np.exp(-0.1 * np.arange(len(scales_arr)))

        # Calculate fractal dimension via weighted regression
        coef = np.polyfit(log_scales, log_counts, 1, w=weights)[0]
        df = -coef

        return float(np.clip(df, 1.0, 2.0))

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
        lbp = local_binary_pattern(gray, P=n_points, R=radius, method="uniform")

        # Compute histogram
        n_bins = n_points + 2  # Number of uniform patterns + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, density=True)

        # Calculate entropy
        hist = hist + 1e-10  # Avoid log(0)
        texture_entropy = entropy(hist, base=2)

        # Normalize by maximum possible entropy
        max_entropy = np.log2(n_bins)
        return float(texture_entropy / max_entropy)

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

        # Paper Eq.(22): phi3 = (Var(Gx)+Var(Gy)) / (Var(Gx)+Var(Gy)+eps),
        # 3x3 Sobel operators. eps acts as a scale constant (eps=1.0 on
        # [0,1]-normalized input — implementation assumption, paper leaves
        # eps unspecified; consistent with the GPU path).
        gray_f = gray.astype(np.float32)
        if gray_f.max() > 1.5:
            gray_f = gray_f / 255.0
        grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

        v = float(np.var(grad_x) + np.var(grad_y))
        return v / (v + 1.0)

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

        # Apply Gaussian blur to reduce noise (paper Appendix: sigma = 1.0)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Paper Eq.(23): Canny edge detection with adaptive thresholds based on
        # the Otsu method — Otsu's threshold as the high threshold, half as low.
        otsu_thr, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        upper = int(max(1, otsu_thr))
        lower = int(max(0, 0.5 * otsu_thr))
        edges = cv2.Canny(blurred, lower, upper)

        # Calculate edge density
        return float(np.sum(edges > 0) / edges.size)

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
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        # Paper Eq.(24): phi5 = (1/K) * sum_k Pk^2 / (4*pi*Ak)  (mean inverse circularity;
        # a circle gives 1, more complex shapes give larger values).
        inverse_circularities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Filter out noise
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    inverse_circularities.append(
                        float(perimeter**2 / (4.0 * np.pi * area))
                    )

        if not inverse_circularities:
            return 0.0

        ic_mean = max(float(np.mean(inverse_circularities)), 1.0)
        # IMPLEMENTATION ASSUMPTION (not stated in the paper): Eq.(24) is
        # unbounded above (circle = 1), but Eq.(8) requires *normalized*
        # descriptors phi-tilde. We map to [0,1) via 1 - 1/ic — monotone in ic;
        # circle -> 0, complex -> 1. The paper specifies normalizers for
        # phi1 (Df/2) and phi2 (Ht/Hmax) but leaves phi5's unspecified.
        return 1.0 - 1.0 / ic_mean

    def get_tile_hash(self, tile: torch.Tensor) -> str:
        """Generate hash for tile caching."""
        tile_np = tile.detach().cpu().numpy()
        return hashlib.md5(tile_np.tobytes()).hexdigest()

    def bilateral_filter(
        self,
        complexity_map: torch.Tensor,
        sigma_spatial: float = 2.0,
        sigma_range: float = 0.1,
        kernel_size: int = 5,
    ) -> torch.Tensor:
        """
        True bilateral filter (Algorithm 1 line 18: BilateralFilter(C, sigma_s=2, sigma_r=0.1)).

        Combines a spatial Gaussian with a range (intensity-similarity) Gaussian so that
        smoothing does not blur across sharp complexity transitions. Differentiable.

        Args:
            complexity_map: Complexity map tensor (B, H, W)
            sigma_spatial: Spatial sigma for Gaussian kernel
            sigma_range: Range sigma for intensity similarity

        Returns:
            Filtered complexity map (B, H, W)
        """
        B, H, W = complexity_map.shape
        pad = kernel_size // 2
        x4 = complexity_map.unsqueeze(1)  # (B, 1, H, W)

        # Extract k*k neighborhoods around every position: (B, k*k, H*W)
        patches = F.unfold(
            F.pad(x4, (pad, pad, pad, pad), mode="replicate"),
            kernel_size,
        )
        center = complexity_map.reshape(B, 1, H * W)

        # Spatial Gaussian weights (1, k*k, 1)
        coords = torch.arange(
            kernel_size, dtype=torch.float32, device=complexity_map.device
        ) - pad
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        spatial_w = torch.exp(-(yy**2 + xx**2) / (2 * sigma_spatial**2))
        spatial_w = spatial_w.reshape(1, -1, 1)

        # Range Gaussian weights (B, k*k, H*W)
        range_w = torch.exp(-((patches - center) ** 2) / (2 * sigma_range**2))

        weights = spatial_w * range_w
        filtered = (weights * patches).sum(dim=1) / (weights.sum(dim=1) + 1e-8)
        return filtered.reshape(B, H, W)

    # ------------------------------------------------------------------
    # GPU tile-wise metric helpers (vectorized; deterministic side-information)
    # ------------------------------------------------------------------
    def _tile_size(self, H: int) -> int:
        """
        Tile size for an HxH map: largest power of two <= max(4, H // grid_size).

        Power-of-two tiles are required so the dyadic box-counting scales
        (Algorithm 2) divide the tile exactly — non-power-of-two tiles produce
        mismatched per-scale grids (workflow finding [4]) — and match
        Algorithm 1 line 1's s in {16, 32, 64}. The floor of 4 guarantees at
        least two dyadic scales for the fractal regression (finding [5]).

        DOCUMENTED DEVIATION: the paper's tile-size statements conflict —
        Sec IV-D/Eq.12 say "H/8 (8x8 grid)" while Algorithm 1 requires
        power-of-two s and Table VIII defaults to 32. The pow2 floor follows
        Algorithm 1; common sizes then give a 10x10 grid (e.g. 640 -> tile 64,
        80 -> tile 8) rather than literally 8x8.
        """
        raw = max(4, H // self.grid_size)
        return 1 << (raw.bit_length() - 1)

    @staticmethod
    def _normalize01(x: torch.Tensor) -> torch.Tensor:
        """Per-image min-max normalization to [0,1]. x: (B,1,H,W)."""
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    @staticmethod
    def _sobel(gray: torch.Tensor):
        """3x3 Sobel gradients (Eq.22 / Appendix). gray: (B,1,H,W)."""
        dev = gray.device
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=dev
        ).view(1, 1, 3, 3)
        return F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)

    @staticmethod
    def _otsu_threshold(x: torch.Tensor, bins: int = 256) -> torch.Tensor:
        """
        Per-image Otsu threshold (paper Eq.23: 'adaptive thresholds based on the
        Otsu method'). x: (B,1,H,W) in [0,1]. Returns (B,1,1,1) thresholds.
        Small per-image loop (batch-sized) keeps memory bounded.
        """
        B = x.shape[0]
        thrs = torch.zeros(B, 1, 1, 1, device=x.device, dtype=torch.float32)
        centers = (
            torch.arange(bins, dtype=torch.float32, device=x.device) + 0.5
        ) / bins
        for b in range(B):
            v = x[b].flatten()
            hist = torch.histc(v, bins=bins, min=0.0, max=1.0)
            p = hist / hist.sum().clamp(min=1.0)
            omega = torch.cumsum(p, dim=0)
            mu = torch.cumsum(p * centers, dim=0)
            mu_t = mu[-1]
            sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
            thrs[b] = centers[torch.argmax(sigma_b)]
        return thrs

    @classmethod
    def _otsu_binarize(cls, x: torch.Tensor, bins: int = 256) -> torch.Tensor:
        """Per-image Otsu thresholding. x: (B,1,H,W) in [0,1] -> float {0,1} mask."""
        thr = cls._otsu_threshold(x, bins)
        return (x > thr).float()

    @classmethod
    def _gpu_canny(cls, gray: torch.Tensor) -> torch.Tensor:
        """
        Tensorized Canny edge detection with Otsu-adaptive double thresholds
        (paper Eq.23 / Algorithm 2 input: 'Canny edge detection with adaptive
        thresholds based on the Otsu method', Gaussian sigma=1.0):

          1. 5x5 Gaussian blur (sigma=1.0, matching the cv2 path)
          2. Sobel gradients -> magnitude + direction
          3. Non-maximum suppression along 4 quantized gradient directions
          4. Double threshold: high = per-image Otsu, low = 0.5 * high
          5. Hysteresis: weak edges kept when 8-connected to strong edges
             (two dilation passes — bounded approximation of full flood fill)

        DOCUMENTED DEVIATIONS from the cv2 reference recipe (compute_edge_density):
        - the Otsu threshold here is taken on the NMS gradient-magnitude
          distribution (separates edge/non-edge bimodality directly in the
          domain being thresholded), whereas the cv2 recipe derives it from the
          blurred *intensity* histogram and hands it to cv2.Canny — the two are
          not numerically identical;
        - hysteresis is bounded to two dilation passes instead of the full
          flood fill. Exact behavior: metric_backend='cv2'.

        gray: (B,1,H,W) in [0,1]. Returns float {0,1} edge map.
        """
        dev = gray.device
        # 1) Gaussian blur 5x5, sigma=1.0
        x1 = torch.arange(5, dtype=torch.float32, device=dev) - 2
        g1 = torch.exp(-(x1**2) / 2.0)
        g1 = g1 / g1.sum()
        g2 = (g1.unsqueeze(0) * g1.unsqueeze(1)).view(1, 1, 5, 5)
        blurred = F.conv2d(gray, g2, padding=2)

        # 2) Sobel magnitude + direction
        gx, gy = cls._sobel(blurred)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-12)

        # 3) Non-maximum suppression: quantize direction into 4 bins and
        #    compare with both neighbors along the gradient direction
        angle = torch.atan2(gy, gx) * (180.0 / math.pi)
        angle = torch.where(angle < 0, angle + 180.0, angle)

        def shift(t, dy, dx):
            return F.pad(t, (2, 2, 2, 2), mode="replicate")[
                :, :, 2 + dy : 2 + dy + t.shape[2], 2 + dx : 2 + dx + t.shape[3]
            ]

        # direction bins: 0 (E-W), 45 (NE-SW), 90 (N-S), 135 (NW-SE)
        bins = [
            ((angle < 22.5) | (angle >= 157.5), (0, 1), (0, -1)),
            ((angle >= 22.5) & (angle < 67.5), (-1, 1), (1, -1)),
            ((angle >= 67.5) & (angle < 112.5), (-1, 0), (1, 0)),
            ((angle >= 112.5) & (angle < 157.5), (-1, -1), (1, 1)),
        ]
        nms = torch.zeros_like(mag)
        for sel, (dy1, dx1), (dy2, dx2) in bins:
            keep = (mag >= shift(mag, dy1, dx1)) & (mag >= shift(mag, dy2, dx2))
            nms = torch.where(sel & keep, mag, nms)

        # 4) Otsu double threshold on the NMS magnitude:
        #    high = Otsu threshold, low = 0.5 * high (matching the cv2 path)
        nms_n = cls._normalize01(nms)
        thr_high = cls._otsu_threshold(nms_n)
        strong = (nms_n > thr_high).float()
        weak = (nms_n > 0.5 * thr_high).float()

        # 5) Hysteresis: weak pixels kept when 8-connected to a strong pixel
        edge = strong
        for _ in range(2):
            grown = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
            edge = torch.where((weak > 0) & (grown > 0), torch.ones_like(edge), edge)
        return edge

    @staticmethod
    def _fractal_dimension_tiles(edge: torch.Tensor, tile: int) -> torch.Tensor:
        """
        phi1: multi-resolution box-counting fractal dimension per tile (Algorithm 2),
        vectorized across all tiles. edge: (B,1,Hc,Wc) binary float. Returns Df (B,ht,wt)
        clipped to [1,2].
        """
        B, _, Hc, Wc = edge.shape
        ht, wt = Hc // tile, Wc // tile

        # Dyadic scales within a tile: 2, 4, ..., tile (Algorithm 2 line 1)
        scales = []
        s = 2
        while s <= tile:
            scales.append(s)
            s *= 2
        if len(scales) < 2:
            # Too few scales for a regression — degenerate tile, return Df=1
            return torch.ones(B, ht, wt, device=edge.device)

        counts = []
        for s in scales:
            pooled = F.max_pool2d(edge, kernel_size=s, stride=s)  # box occupancy at scale s
            k = tile // s
            # Number of occupied boxes of size s inside each tile (Algorithm 2 line 5)
            n_s = F.avg_pool2d(pooled, kernel_size=k, stride=k) * (k * k)
            counts.append(n_s.squeeze(1))  # (B,ht,wt)

        n = torch.stack(counts, dim=0)  # (S,B,ht,wt)
        S = len(scales)
        x = torch.log(
            torch.tensor(scales, dtype=torch.float32, device=edge.device)
        ).view(S, 1, 1, 1)
        y = torch.log(n + 1.0)
        # Exponential weights e^{-0.1 i} (Algorithm 2 line 8)
        w = torch.exp(
            -0.1 * torch.arange(S, dtype=torch.float32, device=edge.device)
        ).view(S, 1, 1, 1)

        # Weighted least-squares slope; Df = -slope (Algorithm 2 line 9)
        w_sum = w.sum(dim=0)
        x_mean = (w * x).sum(dim=0) / w_sum
        y_mean = (w * y).sum(dim=0) / w_sum
        cov = (w * (x - x_mean) * (y - y_mean)).sum(dim=0)
        var = (w * (x - x_mean) ** 2).sum(dim=0)
        df = -(cov / (var + 1e-12))
        return df.clamp(1.0, 2.0)  # Algorithm 2 line 10

    @staticmethod
    def _lbp_entropy_tiles(gray: torch.Tensor, tile: int) -> torch.Tensor:
        """
        phi2: texture entropy from uniform LBP histograms per tile (Eq.2 / Eq.21).
        P=8 neighbors, R=1; uniform patterns map to #ones (0..8), non-uniform to 9
        (P+2 = 10 bins). Entropy normalized by Hmax = log2(P+2). gray: (B,1,Hc,Wc).
        """
        B, _, Hc, Wc = gray.shape
        g = gray
        gp = F.pad(g, (1, 1, 1, 1), mode="replicate")
        # 8 neighbors at R=1 in circular order (Eq.2)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        bits = []
        for dy, dx in offsets:
            nb = gp[:, :, 1 + dy : 1 + dy + Hc, 1 + dx : 1 + dx + Wc]
            bits.append((nb >= g).float())  # s(gp - gc), s(x) = 1[x >= 0]
        bits_t = torch.cat(bits, dim=1)  # (B,8,Hc,Wc)

        n_ones = bits_t.sum(dim=1)  # (B,Hc,Wc)
        # Circular transition count: uniform LBP iff <= 2 transitions
        trans = (bits_t - torch.roll(bits_t, shifts=1, dims=1)).abs().sum(dim=1)
        label = torch.where(
            trans <= 2.0, n_ones, torch.full_like(n_ones, 9.0)
        ).long()  # 0..9

        onehot = F.one_hot(label, num_classes=10).permute(0, 3, 1, 2).float()
        p_tile = F.avg_pool2d(onehot, kernel_size=tile, stride=tile)  # (B,10,ht,wt)
        # Eq.(21): Ht = -sum p_i log2(p_i + eps), normalized by Hmax = log2(P+2)
        ent = -(p_tile * torch.log2(p_tile + 1e-10)).sum(dim=1)
        return ent / math.log2(10.0)

    @staticmethod
    def _gradient_variance_tiles(
        gx: torch.Tensor, gy: torch.Tensor, tile: int
    ) -> torch.Tensor:
        """
        phi3: Eq.(22) = (Var(Gx)+Var(Gy)) / (Var(Gx)+Var(Gy)+eps) per tile.
        eps acts as a scale constant; eps=1.0 is an implementation assumption
        (the paper leaves eps unspecified; a vanishing eps would saturate phi3 at 1).
        """

        def tile_var(t):
            m = F.avg_pool2d(t, kernel_size=tile, stride=tile)
            m2 = F.avg_pool2d(t * t, kernel_size=tile, stride=tile)
            return (m2 - m * m).clamp(min=0.0)

        v = (tile_var(gx) + tile_var(gy)).squeeze(1)  # (B,ht,wt)
        return v / (v + 1.0)

    @staticmethod
    def _contour_complexity_tiles(binmask: torch.Tensor, tile: int) -> torch.Tensor:
        """
        phi5: Eq.(24) inverse circularity P^2/(4*pi*A) per tile.

        DOCUMENTED APPROXIMATION (K=1): the tile's foreground is treated as a
        single region, whereas Eq.(24) averages over K individually detected
        contours — pure tensor pooling cannot enumerate contours, and when a
        tile contains several blobs this differs from the per-contour mean.
        The exact per-contour path is compute_contour_complexity (cv2), used
        by metric_backend='cv2' for offline scoring/calibration. Normalized to
        [0,1) via 1 - 1/ic, consistent with the cv2 path (see its comment).
        """
        m = binmask  # (B,1,Hc,Wc) float {0,1}
        eroded = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
        boundary = (m - eroded).clamp(min=0.0)

        area = F.avg_pool2d(m, kernel_size=tile, stride=tile) * (tile * tile)
        perim = F.avg_pool2d(boundary, kernel_size=tile, stride=tile) * (tile * tile)

        ic = (perim * perim) / (4.0 * math.pi * area + 1e-6)  # Eq.(24)
        phi5 = 1.0 - 1.0 / ic.clamp(min=1.0)
        # Empty tiles (no foreground) -> 0 complexity
        phi5 = torch.where(area.squeeze(1) > 0, phi5.squeeze(1), torch.zeros_like(phi5.squeeze(1)))
        return phi5

    def _phi_tiles_cv2(self, features: torch.Tensor):
        """
        Exact per-tile metrics via the OpenCV reference implementations
        (paper Eq.21-24 with Canny+Otsu edge maps and per-contour circularity).
        CPU-bound — intended for offline dataset scoring / calibration-time
        analysis (the paper runs morphological analysis at calibration time).
        """
        B, C, H, W = features.shape
        tile = self._tile_size(H)  # consistent grid with the GPU path
        ht, wt = H // tile, W // tile

        imgs = features.detach().float().cpu()
        gray_all = imgs.mean(dim=1)  # (B,H,W)
        # Normalize to uint8 per image for the cv2 metric functions
        phi = torch.zeros(B, ht, wt, 8)
        detailed = {
            k: torch.zeros(B, ht, wt)
            for k in ("fractal", "texture", "gradient", "edge", "contour")
        }

        for b in range(B):
            g = gray_all[b].numpy()
            g_min, g_max = float(g.min()), float(g.max())
            g8 = ((g - g_min) / (g_max - g_min + 1e-8) * 255.0).astype(np.uint8)

            for i in range(ht):
                for j in range(wt):
                    t8 = g8[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile]

                    # Edge map for phi1: Canny with Otsu-adaptive thresholds
                    blurred = cv2.GaussianBlur(t8, (5, 5), 1.0)
                    otsu_thr, _ = cv2.threshold(
                        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    edges = cv2.Canny(
                        blurred, int(max(0, 0.5 * otsu_thr)), int(max(1, otsu_thr))
                    )

                    p1 = self.fast_fractal_dimension((edges > 0).astype(np.uint8)) / 2.0
                    p2 = self.compute_texture_entropy(t8)
                    p3 = self.compute_gradient_variance(t8)
                    p4 = self.compute_edge_density(t8)
                    p5 = self.compute_contour_complexity(t8)

                    detailed["fractal"][b, i, j] = p1
                    detailed["texture"][b, i, j] = p2
                    detailed["gradient"][b, i, j] = p3
                    detailed["edge"][b, i, j] = p4
                    detailed["contour"][b, i, j] = p5
                    phi[b, i, j] = torch.tensor(
                        [p1, p2, p3, p4, p5,
                         p1 * p2, p3**2, math.sqrt(max(p4 * p5, 0.0))]
                    )

        dev = features.device
        return phi.to(dev), {k: v.to(dev) for k, v in detailed.items()}

    def compute_phi_tiles(self, features: torch.Tensor):
        """
        Compute the five normalized morphological descriptors per tile (no_grad —
        paper: 'descriptors are computed as deterministic side-information').

        Dispatches to the exact cv2 backend or the vectorized GPU surrogate
        depending on self.metric_backend.

        Args:
            features: (B, C, H, W)

        Returns:
            phi: (B, ht, wt, 8) tensor [phi1..phi5, phi1*phi2, phi3^2, phi4*phi5]
            detailed: dict of the 5 individual (B,ht,wt) metrics
        """
        if self.metric_backend == "cv2":
            with torch.no_grad():
                return self._phi_tiles_cv2(features)

        # AMP guard: the hooks run inside the trainer's autocast region, where
        # F.conv2d would emit fp16 tensors — but torch.histc (Otsu) does not
        # support fp16 on CUDA. Force full precision for the metric pipeline
        # (deterministic side-information; precision matters more than speed).
        if features.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                return self._phi_tiles_gpu(features.float())
        return self._phi_tiles_gpu(features.float())

    def _phi_tiles_gpu(self, features: torch.Tensor):
        """Vectorized GPU metric path (see compute_phi_tiles for dispatch)."""
        B, C, H, W = features.shape
        # Paper Sec IV-D: tile size defaults to H/8 (8x8 grid); Eq.(12) allows finer
        # grids for high object density. Rounded to a power of two for the dyadic
        # box-counting scales (see _tile_size).
        tile = self._tile_size(H)
        ht, wt = H // tile, W // tile
        Hc, Wc = ht * tile, wt * tile  # crop to an exact tile multiple

        with torch.no_grad():
            gray = features[:, :, :Hc, :Wc].mean(dim=1, keepdim=True).float()
            gray = self._normalize01(gray)

            # Plain Sobel gradients for phi3 (Eq.22 specifies Sobel directly)
            gx, gy = self._sobel(gray)

            # Edge map for phi1/phi4: tensorized Canny with Otsu-adaptive
            # thresholds (Eq.23 / Algorithm 2 input) — blur, NMS, double
            # threshold, hysteresis. See _gpu_canny.
            edge = self._gpu_canny(gray)
            # Foreground mask for phi5 (cv2 path uses adaptive threshold on gray)
            binmask = self._otsu_binarize(gray)

            phi1 = self._fractal_dimension_tiles(edge, tile) / 2.0  # Df/2 in [0.5,1]
            phi2 = self._lbp_entropy_tiles(gray, tile)
            phi3 = self._gradient_variance_tiles(gx, gy, tile)
            phi4 = F.avg_pool2d(edge, kernel_size=tile, stride=tile).squeeze(1)  # Eq.(23)
            phi5 = self._contour_complexity_tiles(binmask, tile)

            # Algorithm 1 line 14: phi <- [phi1..phi5, phi1*phi2, phi3^2, sqrt(phi4*phi5)]
            # (the LaTeX source confirms the 8th feature is sqrt(phi4*phi5))
            phi = torch.stack(
                [phi1, phi2, phi3, phi4, phi5,
                 phi1 * phi2, phi3**2, torch.sqrt(phi4 * phi5 + 1e-12)],
                dim=-1,
            )  # (B,ht,wt,8)

        detailed = {
            "fractal": phi1,
            "texture": phi2,
            "gradient": phi3,
            "edge": phi4,
            "contour": phi5,
        }
        return phi, detailed

    def score_image(self, features: torch.Tensor) -> torch.Tensor:
        """
        Deterministic per-image unified complexity for dataset sorting
        (Algorithm 3 line 1 SortByComplexity). Uses Eq.(8) C = sum alpha_i * phi_i
        with the alpha_i parameters (init 1/5, projected to the simplex), averaged
        over tiles. Deterministic w.r.t. the (untrained) MLP.

        Returns: (B,) tensor in [0,1].
        """
        phi, _ = self.compute_phi_tiles(features)
        with torch.no_grad():
            alpha = self.feature_weights.detach().abs()
            alpha = alpha / alpha.sum().clamp(min=1e-8)  # Eq.(8): sum=1, >=0
            c = (phi[..., :5] * alpha.view(1, 1, 1, 5)).sum(dim=-1)  # (B,ht,wt)
            return c.mean(dim=(1, 2)).clamp(0.0, 1.0)

    def forward(
        self,
        features: torch.Tensor,
        return_detailed: bool = False,
    ):
        """
        Morphological complexity computation (Algorithm 1).

        phi computation is deterministic side-information (no_grad); gradients flow
        into complexity_mlp (whose weights absorb Eq.8's alpha_i and Eq.9's beta_ij
        on the interaction inputs) and onward through the bit-mapping network.

        Args:
            features: Tensor of shape (B, C, H, W)
            return_detailed: If True, also return the 5 individual metrics

        Returns:
            complexity_map: (B, ht, wt) in [0, 1]
            detailed_metrics: dict of individual metrics (if return_detailed=True)
        """
        phi, detailed = self.compute_phi_tiles(features)
        B, ht, wt, _ = phi.shape

        # Algorithm 1 line 15: C <- MLP(phi)
        flat = phi.reshape(-1, 8)
        complexity_map = self.complexity_mlp(flat).reshape(B, ht, wt)

        # Algorithm 1 line 18: BilateralFilter(C, sigma_s=2, sigma_r=0.1)
        complexity_map = self.bilateral_filter(complexity_map)
        complexity_map = complexity_map.clamp(0.0, 1.0)

        if return_detailed:
            return complexity_map, detailed

        return complexity_map
