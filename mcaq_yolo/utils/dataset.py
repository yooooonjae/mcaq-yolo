"""
Dataset utilities for MCAQ-YOLO
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import yaml


# =========================================
# Simple ComplexityDataset (use if needed)
# =========================================

class ComplexityDataset(Dataset):
    """
    Generic dataset wrapper that includes complexity computation.
    (Not required by the current main training, but kept for compatibility.)
    """

    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[np.ndarray],
    ):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self.images[idx]
        labels = self.labels[idx]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels_tensor = torch.from_numpy(labels).float()
        return {
            "img": img_tensor,
            "labels": labels_tensor,
            "idx": idx,
        }


# =========================================
# YOLOComplexityDataset (for main training)
# =========================================

class YOLOComplexityDataset(Dataset):
    """
    YOLO-style dataset with complexity-aware support.

    - Reads a YOLO-format .yaml and loads the train/val images
    - Images: letterboxed and padded to 640x640 (or the configured img_size)
    - Labels: [cls, x, y, w, h] format, 0~1 normalized (relative to the final padded image)
    """

    def __init__(
        self,
        yaml_path: str,
        mode: str = "train",
        img_size: int = 640,
        augment: bool = True,
    ):
        """
        Args:
            yaml_path: dataset yaml path
            mode: 'train' or 'val' or 'test'
            img_size: final square size (e.g. 640)
            augment: recommended True only when mode is 'train'
        """
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        assert mode in ("train", "val", "test"), f"Unknown mode: {mode}"

        self.mode = mode
        self.img_size = img_size
        self.augment = augment and (mode == "train")

        # root path
        self.root = Path(self.config["path"])

        # image directory: path/train, path/val, path/test
        img_rel = self.config[mode]
        self.img_dir = self.root / img_rel

        # list of image files
        self.img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.img_paths.extend(sorted(self.img_dir.glob(ext)))

        # label directory: path/labels/train, path/labels/val ...
        self.label_dir = self.root / "labels" / mode
        self.labels = self._load_labels()

        self.class_names = self.config.get("names", {})
        self.complexity_scores: Optional[np.ndarray] = None

    # --------------------------
    # internal utilities
    # --------------------------
    def _load_labels(self) -> List[np.ndarray]:
        """
        Load labels in YOLO txt format:
        each line: cls x y w h  (normalized)
        """
        labels: List[np.ndarray] = []

        for img_path in self.img_paths:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, "r") as f:
                    label_data = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]
                            label_data.append([cls] + bbox)
                    if label_data:
                        labels.append(np.array(label_data, dtype=np.float32))
                    else:
                        labels.append(np.empty((0, 5), dtype=np.float32))
            else:
                labels.append(np.empty((0, 5), dtype=np.float32))

        return labels

    @staticmethod
    def _letterbox(
        img: np.ndarray,
        labels: np.ndarray,
        new_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        YOLO-style letterbox: aspect-ratio-preserving resize + padding to a
        square (new_size x new_size).
        labels: [cls, x, y, w, h] (0~1, relative to the original size) are
                renormalized to 0~1 relative to the final padded image and returned.
        """
        h0, w0 = img.shape[:2]  # original h, w
        # copy in case there are no labels
        labels = labels.copy()

        # scale (fit new_size based on the longer side)
        r = min(new_size / h0, new_size / w0)
        new_unpad_w = int(round(w0 * r))
        new_unpad_h = int(round(h0 * r))

        # resize
        if (w0, h0) != (new_unpad_w, new_unpad_h):
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        # padding computation (make it square)
        dw = new_size - new_unpad_w
        dh = new_size - new_unpad_h
        dw /= 2  # split left/right
        dh /= 2  # split top/bottom

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        # padding
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # ---- label transform ----
        # labels: [cls, x, y, w, h] (0~1, relative to the original)
        if labels.shape[0] > 0:
            # absolute coordinates in the original frame
            x = labels[:, 1] * w0
            y = labels[:, 2] * h0
            w = labels[:, 3] * w0
            h = labels[:, 4] * h0

            # apply the resize ratio
            x *= r
            y *= r
            w *= r
            h *= r

            # apply padding (shift coordinates)
            x += left
            y += top

            # renormalize to 0~1 relative to new_size
            labels[:, 1] = x / new_size
            labels[:, 2] = y / new_size
            labels[:, 3] = w / new_size
            labels[:, 4] = h / new_size

        return img, labels

    def _augment(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple augmentation:
        - horizontal flip
        - brightness scaling
        """
        # horizontal flip
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
            img = np.ascontiguousarray(img)  # prevent negative stride
            if labels.shape[0] > 0:
                # x_center -> 1 - x_center
                labels[:, 1] = 1.0 - labels[:, 1]

        # slightly change brightness / contrast
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return img, labels

    # --------------------------
    # Dataset interface
    # --------------------------
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.img_paths[idx]
        labels = self.labels[idx]

        # load image (RGB)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        # letterbox
        img, labels = self._letterbox(img, labels, self.img_size)

        # augmentation
        if self.augment:
            img, labels = self._augment(img, labels)

        # convert to torch tensor
        img = np.ascontiguousarray(img)  # guard against a possible negative stride
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels_tensor = torch.from_numpy(labels).float()

        item: Dict[str, Any] = {
            "img": img_tensor,
            "labels": labels_tensor,
            "path": str(img_path),
            "idx": idx,
        }

        if self.complexity_scores is not None and idx < len(self.complexity_scores):
            item["complexity"] = float(self.complexity_scores[idx])

        return item


# =========================================
# Complexity computation utilities
# =========================================

def compute_dataset_complexity(
    dataset: Dataset,
    model: Optional[torch.nn.Module] = None,
    batch_size: int = 1,
    device: str = "cuda",
    save_path: Optional[str] = None,
    backend: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the complexity score over the whole dataset (Algorithm 3 line 1: SortByComplexity).

    - If a model is given, uses the unified morphological complexity C(x)
      (Eq.8, normalized weighted sum of 5 metrics) via
      model.complexity_analyzer.score_image().
    - If model=None, falls back to a simple edge-density estimate.

    backend (REVIEW FIX): None (default) keeps the analyzer's current (=training-
    time) metric_backend — the single-source-of-truth principle, so that the
    curriculum ordering is computed with the exact same complexity signal the
    training actually sees. Specifying 'cv2' / 'gpu' temporarily switches to that
    backend for scoring only. (The old version always forced a switch to 'cv2';
    before the cv2compat parity patch, its fusion-map correlation with the
    training signal was only r~0.45, causing an ordering-vs-training mismatch.)
    """
    print(f"Computing complexity for {len(dataset)} samples...")

    # Unified morphological complexity analyzer (Eq.8) — preferred path
    analyzer = None
    if model is not None:
        analyzer = getattr(model, "complexity_analyzer", None)
        if analyzer is None and hasattr(model, "score_image"):
            analyzer = model  # analyzer passed in directly
    if analyzer is not None and not hasattr(analyzer, "score_image"):
        analyzer = None  # incompatible object — fall back to edge density

    # Backend for offline scoring: keep the analyzer's training-time backend
    # unless the caller explicitly requests 'cv2' (exact Eq.21-24 reference)
    # or 'gpu'. Consistency with the training signal is the default.
    prev_backend = getattr(analyzer, "metric_backend", None)
    if analyzer is not None and prev_backend is not None and backend is not None:
        analyzer.metric_backend = backend

    # batch_size must be 1: _collate_fn returns only batch_list[0], so with
    # batch_size>1 only 1 score is computed per batch and the dataset indices
    # and score array get out of sync (breaking the curriculum filter
    # Dt={C(x)<=tau_t}). The function's batch_size argument is accepted for
    # compatibility but is not used here.
    def _collate_fn(batch_list):
        return batch_list[0]

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn,
    )

    complexities: List[float] = []

    for batch in tqdm(dataloader):
        if isinstance(batch, dict):
            img_tensor = batch["img"]  # (C,H,W)
        else:
            # when the item is an (img, label, ...) tuple
            img_tensor = batch[0]

        # ---- Preferred: unified morphological complexity C(x) (Eq.8) ----
        if analyzer is not None and isinstance(img_tensor, torch.Tensor):
            x = img_tensor
            if x.dim() == 3:
                x = x.unsqueeze(0)  # (1,C,H,W)
            x = x.float()
            if x.max() > 1.5:
                x = x / 255.0
            with torch.no_grad():
                score = float(analyzer.score_image(x.to(device)).mean().item())
            complexities.append(score)
            continue

        # ---- Fallback: edge density ----
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.cpu().numpy()
        else:
            img_np = np.array(img_tensor, dtype=np.float32)

        # (C,H,W) -> (H,W,C)
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))

        # assume 0~1 and convert to 0~255
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # convert to grayscale
        if img_np.ndim == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # simple complexity based on edge density
        edges = cv2.Canny(gray, 50, 150)
        score = float(np.sum(edges > 0) / edges.size)
        complexities.append(score)

    # Restore the analyzer's training-time backend
    if analyzer is not None and prev_backend is not None:
        analyzer.metric_backend = prev_backend

    complexities_arr = np.array(complexities, dtype=np.float32)

    # save
    if save_path is not None:
        save_path = str(save_path)
        np.save(save_path, complexities_arr)
        print(f"Saved complexity scores to {save_path}")

    print("Complexity statistics:")
    print(f"  Mean: {complexities_arr.mean():.4f}")
    print(f"  Std : {complexities_arr.std():.4f}")
    print(f"  Min : {complexities_arr.min():.4f}")
    print(f"  Max : {complexities_arr.max():.4f}")

    return complexities_arr


# =========================================
# Complexity-balanced Sampler (optional)
# =========================================

def create_complexity_balanced_sampler(
    dataset: Dataset,
    complexity_scores: np.ndarray,
    n_bins: int = 10,
    samples_per_bin: int = 100,
) -> torch.utils.data.Sampler:
    """
    Sampler that samples uniformly across the complexity distribution.
    """
    # bin edges
    bin_edges = np.percentile(complexity_scores, np.linspace(0, 100, n_bins + 1))

    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for idx, score in enumerate(complexity_scores):
        bin_idx = int(np.searchsorted(bin_edges[1:-1], score))
        bins[bin_idx].append(idx)

    sampled_indices: List[int] = []
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue
        n_samples = min(samples_per_bin, len(bin_indices))
        chosen = np.random.choice(bin_indices, n_samples, replace=False)
        sampled_indices.extend(chosen.tolist())

    np.random.shuffle(sampled_indices)
    return torch.utils.data.SubsetRandomSampler(sampled_indices)
