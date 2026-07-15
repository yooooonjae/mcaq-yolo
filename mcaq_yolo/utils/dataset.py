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
# 간단 ComplexityDataset (필요시 사용)
# =========================================

class ComplexityDataset(Dataset):
    """
    Generic dataset wrapper that includes complexity computation.
    (현재 메인 트레이닝에는 사용하지 않아도 되지만, 호환성을 위해 남겨둠)
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
# YOLOComplexityDataset (메인 트레이닝용)
# =========================================

class YOLOComplexityDataset(Dataset):
    """
    YOLO-style dataset with complexity-aware support.

    - YOLO 형식의 .yaml을 읽어서 train/val 이미지를 로드
    - 이미지: letterbox 로 640x640 (또는 설정된 img_size)로 패딩
    - 라벨: [cls, x, y, w, h] 형식, 0~1 normalize (최종 padded 이미지 기준)
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
            img_size: 최종 정사각 사이즈 (예: 640)
            augment: train일 때만 True 권장
        """
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        assert mode in ("train", "val", "test"), f"Unknown mode: {mode}"

        self.mode = mode
        self.img_size = img_size
        self.augment = augment and (mode == "train")

        # root path
        self.root = Path(self.config["path"])

        # 이미지 디렉토리: path/train, path/val, path/test
        img_rel = self.config[mode]
        self.img_dir = self.root / img_rel

        # 이미지 파일 목록
        self.img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.img_paths.extend(sorted(self.img_dir.glob(ext)))

        # 라벨 디렉토리: path/labels/train, path/labels/val ...
        self.label_dir = self.root / "labels" / mode
        self.labels = self._load_labels()

        self.class_names = self.config.get("names", {})
        self.complexity_scores: Optional[np.ndarray] = None

    # --------------------------
    # 내부 유틸
    # --------------------------
    def _load_labels(self) -> List[np.ndarray]:
        """
        YOLO txt 형식의 라벨 로드:
        각 줄: cls x y w h  (normalized)
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
        YOLO-style letterbox: 비율 유지 resize + padding 으로 정사각(new_size x new_size) 만들기.
        labels: [cls, x, y, w, h] (0~1, 원본 크기 기준) 이 들어오면,
                최종 padded 이미지 기준으로 다시 0~1로 normalize해서 반환.
        """
        h0, w0 = img.shape[:2]  # 원본 h, w
        # 라벨이 없을 수도 있으므로 copy
        labels = labels.copy()

        # scale (길이가 더 긴 쪽 기준으로 new_size에 맞춤)
        r = min(new_size / h0, new_size / w0)
        new_unpad_w = int(round(w0 * r))
        new_unpad_h = int(round(h0 * r))

        # resize
        if (w0, h0) != (new_unpad_w, new_unpad_h):
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        # padding 계산 (정사각형 만들기)
        dw = new_size - new_unpad_w
        dh = new_size - new_unpad_h
        dw /= 2  # 왼쪽/오른쪽 나누기
        dh /= 2  # 위/아래 나누기

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

        # ---- 라벨 변환 ----
        # labels: [cls, x, y, w, h] (0~1, 원본 기준)
        if labels.shape[0] > 0:
            # 원본 기준 절대 좌표
            x = labels[:, 1] * w0
            y = labels[:, 2] * h0
            w = labels[:, 3] * w0
            h = labels[:, 4] * h0

            # resize 비율 적용
            x *= r
            y *= r
            w *= r
            h *= r

            # 패딩 적용 (좌표 shift)
            x += left
            y += top

            # 다시 new_size 기준 0~1 normalize
            labels[:, 1] = x / new_size
            labels[:, 2] = y / new_size
            labels[:, 3] = w / new_size
            labels[:, 4] = h / new_size

        return img, labels

    def _augment(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        간단한 augmentation:
        - horizontal flip
        - brightness scaling
        """
        # horizontal flip
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
            img = np.ascontiguousarray(img)  # negative stride 방지
            if labels.shape[0] > 0:
                # x_center -> 1 - x_center
                labels[:, 1] = 1.0 - labels[:, 1]

        # brightness / contrast 조금 변경
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return img, labels

    # --------------------------
    # Dataset 인터페이스
    # --------------------------
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.img_paths[idx]
        labels = self.labels[idx]

        # 이미지 로드 (RGB)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        # letterbox
        img, labels = self._letterbox(img, labels, self.img_size)

        # augmentation
        if self.augment:
            img, labels = self._augment(img, labels)

        # torch tensor 변환
        img = np.ascontiguousarray(img)  # 혹시 모를 negative stride 방지
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
# Complexity 계산 유틸
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
    Dataset 전체에 대해 complexity score 계산 (Algorithm 3 line 1: SortByComplexity).

    - model이 주어지면 model.complexity_analyzer.score_image()로 통합 형태학
      복잡도 C(x) (Eq.8, 5개 메트릭의 정규화 가중합)를 사용.
    - model=None이면 단순 edge density 기반 폴백.

    backend (REVIEW FIX): None(기본)은 analyzer의 현재(=학습 시) metric_backend를
    그대로 사용 — 커리큘럼 정렬이 학습이 실제로 보는 복잡도 신호와 동일한
    측정으로 이루어지도록 하는 단일 신호원(single source of truth) 원칙.
    'cv2' / 'gpu'를 명시하면 스코어링 동안만 해당 백엔드로 일시 전환한다.
    (구버전은 무조건 'cv2'로 강제 전환했는데, cv2compat 패리티 패치 이전에는
    학습 신호와의 융합 맵 상관이 r~0.45에 불과해 정렬-학습 불일치를 낳았다.)
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

    # 반드시 batch_size=1: _collate_fn이 batch_list[0]만 반환하므로 batch_size>1이면
    # 배치당 1개 score만 계산되어 dataset 인덱스와 score 배열이 어긋난다
    # (커리큘럼 필터링 Dt={C(x)<=tau_t}가 깨짐 — Codex final review).
    # 함수 인자의 batch_size는 호환성을 위해 받지만 여기서는 사용하지 않는다.
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
            # (img, label, ...) 형태일 때
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

        # 0~1 이라고 가정하고 0~255로 변환
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # 그레이 변환
        if img_np.ndim == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # edge density 기반 간단 complexity
        edges = cv2.Canny(gray, 50, 150)
        score = float(np.sum(edges > 0) / edges.size)
        complexities.append(score)

    # Restore the analyzer's training-time backend
    if analyzer is not None and prev_backend is not None:
        analyzer.metric_backend = prev_backend

    complexities_arr = np.array(complexities, dtype=np.float32)

    # 저장
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
    Complexity 분포를 균등하게 샘플링하는 Sampler.
    """
    # bin 경계
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
