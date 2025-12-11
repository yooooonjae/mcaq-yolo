"""
Dataset utilities for MCAQ-YOLO
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import yaml
from tqdm import tqdm
from ultralytics.data import YOLODataset


class ComplexityDataset(Dataset):
    """
    Dataset wrapper that includes complexity computation.
    """
    
    def __init__(
        self,
        data_path: str,
        img_size: int = 640,
        augment: bool = True,
        cache_complexity: bool = True,
        complexity_cache_path: Optional[str] = None
    ):
        """
        Initialize complexity dataset.
        
        Args:
            data_path: Path to dataset YAML or directory
            img_size: Image size for training
            augment: Whether to apply augmentations
            cache_complexity: Whether to cache complexity scores
            complexity_cache_path: Path to save/load complexity cache
        """
        # Load base dataset using Ultralytics
        self.base_dataset = YOLODataset(
            img_path=data_path,
            imgsz=img_size,
            augment=augment
        )
        
        self.img_size = img_size
        self.cache_complexity = cache_complexity
        self.complexity_cache_path = complexity_cache_path
        
        # Initialize or load complexity cache
        self.complexity_scores = {}
        if cache_complexity and complexity_cache_path and Path(complexity_cache_path).exists():
            self.load_complexity_cache()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item with complexity information.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing image, labels, and complexity
        """
        # Get base item
        item = self.base_dataset[idx]
        
        # Add complexity if cached
        if idx in self.complexity_scores:
            item['complexity'] = self.complexity_scores[idx]
        else:
            item['complexity'] = None
        
        return item
    
    def compute_complexity_score(
        self,
        image: np.ndarray
    ) -> float:
        """
        Compute simple complexity score for an image.
        
        Args:
            image: Input image
            
        Returns:
            Complexity score
        """
        # Simple complexity based on edge density and texture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture variance
        texture_var = np.var(gray) / (np.mean(gray) + 1e-10)
        
        # Combined score
        complexity = 0.5 * edge_density + 0.5 * np.tanh(texture_var)
        
        return complexity
    
    def precompute_all_complexity(
        self,
        model: Optional[torch.nn.Module] = None,
        batch_size: int = 32,
        device: str = 'cuda'
    ):
        """
        Precompute complexity for entire dataset.
        
        Args:
            model: MCAQ model for complexity computation
            batch_size: Batch size for processing
            device: Device to use
        """
        print("Precomputing complexity scores for dataset...")
        
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        all_complexities = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if model is not None:
                # Use model's complexity analyzer
                images = batch['img'].to(device)
                with torch.no_grad():
                    complexity_map, _ = model.compute_complexity(images)
                    # Average over spatial dimensions
                    complexity_scores = complexity_map.mean(dim=[1, 2]).cpu().numpy()
            else:
                # Use simple complexity computation
                images = batch['img'].numpy()
                complexity_scores = []
                for img in images:
                    # Convert from tensor format to image
                    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                    score = self.compute_complexity_score(img)
                    complexity_scores.append(score)
                complexity_scores = np.array(complexity_scores)
            
            # Store in cache
            start_idx = batch_idx * batch_size
            for i, score in enumerate(complexity_scores):
                self.complexity_scores[start_idx + i] = score
            
            all_complexities.extend(complexity_scores.tolist())
        
        # Save cache
        if self.cache_complexity and self.complexity_cache_path:
            self.save_complexity_cache()
        
        return np.array(all_complexities)
    
    def save_complexity_cache(self):
        """Save complexity cache to file."""
        if self.complexity_cache_path:
            cache_path = Path(self.complexity_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'w') as f:
                json.dump(self.complexity_scores, f)
            
            print(f"Saved complexity cache to {cache_path}")
    
    def load_complexity_cache(self):
        """Load complexity cache from file."""
        if self.complexity_cache_path and Path(self.complexity_cache_path).exists():
            with open(self.complexity_cache_path, 'r') as f:
                cache = json.load(f)
                # Convert string keys to integers
                self.complexity_scores = {int(k): v for k, v in cache.items()}
            
            print(f"Loaded complexity cache with {len(self.complexity_scores)} entries")
    
    def get_complexity_statistics(self) -> Dict:
        """Get statistics of complexity scores."""
        if not self.complexity_scores:
            return {}
        
        scores = list(self.complexity_scores.values())
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75)
        }


class YOLOComplexityDataset(Dataset):
    """
    YOLO dataset with complexity-aware sampling.
    """
    
    def __init__(
        self,
        yaml_path: str,
        mode: str = 'train',
        img_size: int = 640,
        augment: bool = True
    ):
        """
        Initialize YOLO complexity dataset.
        
        Args:
            yaml_path: Path to dataset YAML configuration
            mode: Dataset mode ('train', 'val', 'test')
            img_size: Image size
            augment: Whether to apply augmentations
        """
        # Load dataset configuration
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mode = mode
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        
        # Get image and label paths
        self.root = Path(self.config['path'])
        self.img_dir = self.root / self.config[mode]
        
        # Load image paths
        self.img_paths = sorted(self.img_dir.glob('*.jpg'))
        self.img_paths.extend(sorted(self.img_dir.glob('*.png')))
        
        # Load corresponding label paths
        self.label_dir = self.root / 'labels' / mode
        self.labels = self._load_labels()
        
        # Class names
        self.class_names = self.config['names']
        
        # Complexity scores
        self.complexity_scores = None
    
    def _load_labels(self) -> List:
        """Load YOLO format labels."""
        labels = []
        
        for img_path in self.img_paths:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_data = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]
                            label_data.append([class_id] + bbox)
                    labels.append(np.array(label_data) if label_data else np.empty((0, 5)))
            else:
                labels.append(np.empty((0, 5)))
        
        return labels
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item."""
        # Load image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Load labels
        labels = self.labels[idx].copy()
        
        # Resize image
        img, labels = self._resize(img, labels)
        
        # Apply augmentations
        if self.augment:
            img, labels = self._augment(img, labels)
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(labels).float()
        
        item = {
            'img': img,
            'labels': labels,
            'path': str(img_path),
            'idx': idx
        }
        
        # Add complexity if available
        if self.complexity_scores is not None and idx < len(self.complexity_scores):
            item['complexity'] = self.complexity_scores[idx]
        
        return item
    
    def _resize(self, img: np.ndarray, labels: np.ndarray) -> Tuple:
        """Resize image and adjust labels."""
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        
        if scale != 1:
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        return img, labels
    
    def _augment(self, img: np.ndarray, labels: np.ndarray) -> Tuple:
        """Apply data augmentations."""
        # Simple augmentations
        if np.random.rand() > 0.5:
            # Horizontal flip
            img = np.fliplr(img)
            if len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]
        
        # Color jitter
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        return img, labels


def compute_dataset_complexity(
    dataset: Dataset,
    model: Optional[torch.nn.Module] = None,
    batch_size: int = 32,
    device: str = 'cuda',
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute complexity scores for entire dataset.
    
    Args:
        dataset: Dataset to analyze
        model: MCAQ model (optional)
        batch_size: Batch size for processing
        device: Device to use
        save_path: Path to save complexity scores
        
    Returns:
        Array of complexity scores
    """
    print(f"Computing complexity for {len(dataset)} samples...")
    
    if hasattr(dataset, 'precompute_all_complexity'):
        # Use dataset's built-in method
        complexities = dataset.precompute_all_complexity(
            model=model,
            batch_size=batch_size,
            device=device
        )
    else:
        # Manual computation
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        complexities = []
        
        for batch in tqdm(dataloader):
            if isinstance(batch, dict):
                images = batch['img']
            else:
                images = batch[0]
            
            if model is not None:
                images = images.to(device)
                with torch.no_grad():
                    complexity_map, _ = model.compute_complexity(images)
                    scores = complexity_map.mean(dim=[1, 2]).cpu().numpy()
            else:
                # Simple complexity
                images_np = images.numpy()
                scores = []
                for img in images_np:
                    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    score = np.sum(edges > 0) / edges.size
                    scores.append(score)
                scores = np.array(scores)
            
            complexities.extend(scores.tolist())
    
    complexities = np.array(complexities)
    
    # Save if requested
    if save_path:
        np.save(save_path, complexities)
        print(f"Saved complexity scores to {save_path}")
    
    # Print statistics
    print(f"Complexity statistics:")
    print(f"  Mean: {complexities.mean():.4f}")
    print(f"  Std: {complexities.std():.4f}")
    print(f"  Min: {complexities.min():.4f}")
    print(f"  Max: {complexities.max():.4f}")
    
    return complexities


def create_complexity_balanced_sampler(
    dataset: Dataset,
    complexity_scores: np.ndarray,
    n_bins: int = 10,
    samples_per_bin: int = 100
) -> torch.utils.data.Sampler:
    """
    Create a sampler that balances samples across complexity ranges.
    
    Args:
        dataset: Dataset to sample from
        complexity_scores: Complexity scores for dataset
        n_bins: Number of complexity bins
        samples_per_bin: Samples to draw from each bin
        
    Returns:
        Balanced sampler
    """
    # Compute bin edges
    bin_edges = np.percentile(
        complexity_scores,
        np.linspace(0, 100, n_bins + 1)
    )
    
    # Assign samples to bins
    bins = [[] for _ in range(n_bins)]
    for idx, score in enumerate(complexity_scores):
        bin_idx = np.searchsorted(bin_edges[1:-1], score)
        bins[bin_idx].append(idx)
    
    # Sample from each bin
    sampled_indices = []
    for bin_indices in bins:
        if len(bin_indices) > 0:
            n_samples = min(samples_per_bin, len(bin_indices))
            samples = np.random.choice(bin_indices, n_samples, replace=False)
            sampled_indices.extend(samples.tolist())
    
    # Shuffle final indices
    np.random.shuffle(sampled_indices)
    
    return torch.utils.data.SubsetRandomSampler(sampled_indices)