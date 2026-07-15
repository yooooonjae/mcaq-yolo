"""
MCAQ-YOLO Training Module

- Ultralytics YOLO backbone + MCAQ morphological complexity + bit allocation
- Uses Ultralytics build_yolo_dataset + build_dataloader for full compatibility
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Optional

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.cfg import DEFAULT_CFG

from .models.mcaq_yolo import MCAQYOLO  # the MCAQ model in this repo
from .core.curriculum import CurriculumScheduler
from .utils.dataset import compute_dataset_complexity
from .utils.evaluation import compute_map, extract_targets_per_image, _decode_outputs
from .utils.repro import set_global_seed


class Trainer:
    """
    MCAQ-YOLO Trainer

    - Keeps the Ultralytics YOLO training pipeline as intact as possible
    - Uses build_yolo_dataset + build_dataloader
    - Manages morphology / bit allocation / KD / curriculum all here
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # REVIEW FIX (reproducibility): fix all RNGs BEFORE any stochastic
        # construction (weight init of the complexity/bit-mapping MLPs,
        # dataloader shuffling, augmentation). config['seed'] (int) enables;
        # config['deterministic'] (bool) additionally requests deterministic
        # kernels. Absent -> legacy nondeterministic behavior.
        seed = config.get("seed", None)
        if seed is not None:
            set_global_seed(int(seed), deterministic=bool(config.get("deterministic", False)))
            print(f"[MCAQ] Global seed = {seed} "
                  f"(deterministic={bool(config.get('deterministic', False))})")

        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # --------------------------
        # basic settings
        # --------------------------
        # Paper Table X: total epochs = 300. (A default below warmup_epochs=20
        # would keep the curriculum in Stage 1 forever — quantization would
        # never activate.)
        self.epochs: int = int(config.get("epochs", 300))
        self.batch_size: int = int(config.get("batch_size", 128))
        self.output_dir = Path(config.get("output_dir", "outputs/default_run"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = config.get("data", {})
        self.data_cfg = data_cfg

        self.num_workers: int = int(data_cfg.get("num_workers", 8))
        self.imgsz: int = int(data_cfg.get("imgsz", data_cfg.get("img_size", 640)))

        # --------------------------
        # Teacher / Student model configuration
        # --------------------------
        model_cfg = config.get("model", {})
        teacher_path = model_cfg.get("teacher_path", "yolov8n.pt")
        num_classes = int(model_cfg.get("num_classes", 80))

        # convert the teacher path to an absolute path
        teacher_abs = teacher_path
        if not os.path.isabs(teacher_abs):
            project_root = Path(__file__).resolve().parents[1]
            teacher_abs = str((project_root / teacher_path).resolve())

        print(f"[MCAQ] Loaded teacher model from {teacher_abs}")
        self.teacher_model = YOLO(teacher_abs).to(self.device)
        self.teacher_model.eval()  # for KD

        # --------------------------
        # Create the student (MCAQYOLO) instance
        # REVIEW FIX: previously the code probed MCAQYOLO's arguments at runtime
        # via inspect.signature — an anti-pattern that makes the interface
        # between two classes in the same repo opaque. Replaced with an explicit
        # constructor call (if the signature changes, surfacing it immediately
        # here as a TypeError is the correct failure mode).
        # --------------------------
        quant_cfg = config.get("quantization", {})

        model_kwargs = dict(
            model_name=model_cfg.get("name", "yolov8n"),
            num_classes=num_classes,
            device=str(self.device),
        )
        # Quantization settings (Table X defaults follow the MCAQYOLO-side defaults)
        for key in ("min_bits", "max_bits", "target_bits", "grid_size",
                    "bit_mapping", "normalize_complexity"):
            if key in quant_cfg:
                model_kwargs[key] = quant_cfg[key]

        self.model = MCAQYOLO(**model_kwargs).to(self.device)

        # --------------------------
        # Dataset / Dataloader
        # --------------------------
        self.train_dataset, self.val_dataset = self._init_datasets()
        self.train_loader, self.val_loader = self._init_dataloaders()

        print(
            f"[MCAQ] Datasets initialized: "
            f"train={len(self.train_dataset)} samples, val={len(self.val_dataset)} samples"
        )

        # --------------------------
        # Complexity scores (for the curriculum)
        # --------------------------
        self.complexity_scores: Optional[torch.Tensor] = None
        self.complexity_path = self.output_dir / "complexity_scores.npy"
        self.complexity_scores = self._compute_complexity_scores()

        # --------------------------
        # Optimizer & Scheduler
        # --------------------------
        lr = float(config.get("learning_rate", 1e-3))
        optim_cfg = config.get("optimizer", {})
        opt_type = optim_cfg.get("type", "adamw").lower()

        if opt_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=float(optim_cfg.get("weight_decay", 0.05)),
                betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        sched_cfg = config.get("scheduler", {})
        sched_type = sched_cfg.get("type", "cosine").lower()
        # Paper Table X: LR warmup = 5 epochs (distinct from the curriculum's
        # Twarm=20 — this is the optimizer LR ramp, not the data curriculum)
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))

        if sched_type == "cosine":
            # Main scheduler: Cosine Annealing
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup_epochs,
                eta_min=float(sched_cfg.get("eta_min", 1e-6)),
            )

            if warmup_epochs > 0:
                # Paper Table X: warmup LR 1e-5 with base LR 1e-3 => start_factor = 0.01
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                # Combine warmup + main scheduler
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = main_scheduler
        else:
            self.scheduler = None

        # --------------------------
        # Mixed precision (torch.amp — torch.cuda.amp is deprecated).
        # AMP is enabled on CUDA only: honoring config amp:true on CPU/MPS falls
        # into warning/unsupported code paths (REVIEW FIX — auto-disabled).
        # --------------------------
        self.use_amp = (bool(config.get("training", {}).get("amp", True))
                        and self.device.type == "cuda")
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # mAP evaluation cadence: full-val NMS+AP every N epochs (reviewer
        # note — per-epoch AP can dominate evaluation cost on large val sets).
        self.map_interval = max(1, int(config.get("training", {}).get("map_interval", 1)))

        # distillation, curriculum settings
        self.distill_cfg = config.get("distillation", {})
        self.curriculum_cfg = config.get("curriculum", {})

        # --------------------------
        # Curriculum scheduler — single source of truth for the 3-stage schedule
        # (paper Fig.3 / Algorithm 3 / Table X). Stage boundaries default to the
        # paper's 20/50; temperature anneals 1+9*exp(-5t/T) with T = total epochs.
        # --------------------------
        self.curriculum = CurriculumScheduler(
            warmup_epochs=int(self.curriculum_cfg.get("warmup_epochs", 20)),
            transition_epochs=int(self.curriculum_cfg.get("transition_epochs", 50)),
            total_epochs=self.epochs,
            initial_complexity=float(self.curriculum_cfg.get("initial_complexity", 0.2)),
            initial_temperature=float(self.curriculum_cfg.get("initial_temperature", 10.0)),
            lambda_smooth=float(self.curriculum_cfg.get("lambda_smooth", 0.1)),
        )

        # Raw teacher nn.Module (the YOLO wrapper post-processes outputs; KD needs
        # the raw Detect-head maps)
        self.teacher_module = self.teacher_model.model
        self.teacher_module.eval()
        for p in self.teacher_module.parameters():
            p.requires_grad = False

        # Feature-level KD (paper Sec IV-E: LKD uses logit-level AND feature-level
        # matching): capture the teacher's FP32 C3/C4/C5 features at the same
        # backbone indices the student quantizes (same architecture assumed;
        # shape-checked before use).
        self._teacher_feats: Dict[int, torch.Tensor] = {}
        try:
            t_layers = list(self.teacher_module.model)
            for idx in getattr(self.model, "backbone_out_indices", []):
                if 0 <= idx < len(t_layers):
                    t_layers[idx].register_forward_hook(self._make_teacher_hook(idx))
        except Exception as e:
            print(f"[MCAQ][WARN] teacher feature hooks unavailable: {e}")

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------
    def _build_ultra_cfg(self) -> SimpleNamespace:
        """
        Build the cfg (SimpleNamespace) to pass to Ultralytics build_yolo_dataset.

        - Uses Ultralytics DEFAULT_CFG as the base
        - Overrides the fields we need.
        """
        # Start with DEFAULT_CFG as base (contains all required fields)
        import copy
        base = copy.deepcopy(vars(DEFAULT_CFG))

        # Override with our custom settings
        base["task"] = "detect"
        base["mode"] = "train"
        base["imgsz"] = self.imgsz
        base["batch"] = self.batch_size
        base["workers"] = self.num_workers
        base["device"] = str(self.device)

        # if the YOLO object has overrides, apply some of them
        if hasattr(self.teacher_model, "overrides") and isinstance(self.teacher_model.overrides, dict):
            for key, val in self.teacher_model.overrides.items():
                if key in base:
                    base[key] = val

        return SimpleNamespace(**base)

    def _resolve_data_yaml(self) -> Tuple[str, Dict[str, Any]]:
        """
        Find and return yaml_path (or ppe_yaml, etc.) from config['data'].

        Returns:
            Tuple of (yaml_path, data_dict)
        """
        data_cfg = self.data_cfg
        yaml_path = data_cfg.get("yaml_path", None)

        if yaml_path is None:
            yaml_path = data_cfg.get("ppe_yaml", None) or data_cfg.get("coco_yaml", None)

        if yaml_path is None:
            raise FileNotFoundError(
                "Dataset yaml_path not provided in config['data']. "
                "e.g. config['data']['yaml_path'] = '/path/to/dataset_ppe.yaml'"
            )

        yaml_path = str(Path(yaml_path).resolve())
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset yaml file not found: {yaml_path}")

        # Load yaml as dictionary
        with open(yaml_path, 'r') as f:
            data_dict = yaml.safe_load(f)

        # Ensure 'path' is absolute
        if 'path' in data_dict and not os.path.isabs(data_dict['path']):
            yaml_dir = os.path.dirname(yaml_path)
            data_dict['path'] = os.path.abspath(os.path.join(yaml_dir, data_dict['path']))

        return yaml_path, data_dict

    def _init_datasets(self):
        """
        Create the train/val Dataset using the Ultralytics YOLODataset.
        """
        yaml_path, data_dict = self._resolve_data_yaml()
        cfg_ns = self._build_ultra_cfg()

        data_cfg = self.data_cfg
        train_img_rel = data_cfg.get("train", None)
        val_img_rel = data_cfg.get("val", None)

        if train_img_rel is None or val_img_rel is None:
            raise ValueError(
                "config['data'] must contain 'train' and 'val' paths.\n"
                "e.g. 'train': 'train/images', 'val': 'val/images'"
            )

        # Build absolute paths from data_dict['path']
        dataset_root = data_dict.get("path", "")
        train_img_path = os.path.join(dataset_root, train_img_rel)
        val_img_path = os.path.join(dataset_root, val_img_rel)

        # Kept for the augmentation-free curriculum-scoring dataset (review fix)
        self.train_img_path = train_img_path
        self._data_dict = data_dict

        train_dataset = build_yolo_dataset(
            cfg=cfg_ns,
            img_path=train_img_path,
            batch=self.batch_size,
            data=data_dict,  # Pass dict, not string
            mode="train",
            rect=False,
            stride=32,
        )

        val_dataset = build_yolo_dataset(
            cfg=cfg_ns,
            img_path=val_img_path,
            batch=self.batch_size,
            data=data_dict,  # Pass dict, not string
            mode="val",
            rect=True,
            stride=32,
        )

        return train_dataset, val_dataset

    def _init_dataloaders(self):
        """
        Use Ultralytics' dedicated build_dataloader so that the
        InfiniteDataLoader + its dedicated collate_fn are used as-is.
        """
        train_loader = build_dataloader(
            dataset=self.train_dataset,
            batch=self.batch_size,
            workers=self.num_workers,
            shuffle=True,
            rank=-1,  # single GPU
        )

        val_loader = build_dataloader(
            dataset=self.val_dataset,
            batch=self.batch_size,
            workers=self.num_workers,
            shuffle=False,
            rank=-1,
        )

        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Complexity (for the curriculum)
    # ------------------------------------------------------------------
    def _build_scoring_dataset(self):
        """
        Augmentation-free dataset over the TRAIN images for curriculum scoring.

        REVIEW FIX (measured defect): the previous code scored the train-mode
        dataset directly, whose __getitem__ applies mosaic(=1.0)/flip/HSV —
        fetching the same index twice differed by mean |delta| = 85/255, so
        Algorithm 3's SortByComplexity ordered ONE random mosaic composite per
        index, not the underlying image, and cached that snapshot. Building
        the scoring dataset in mode='val' disables all augmentation; scores
        are then deterministic properties of the raw (letterboxed) images.
        """
        cfg_ns = self._build_ultra_cfg()
        return build_yolo_dataset(
            cfg=cfg_ns,
            img_path=self.train_img_path,
            batch=1,
            data=self._data_dict,
            mode="val",   # mode drives augment=False inside ultralytics
            rect=False,   # keep the square letterbox geometry of training
            stride=32,
        )

    def _compute_complexity_scores(self) -> torch.Tensor:
        """
        Compute the morphological complexity score over the entire train
        dataset (using the augmentation-free scoring dataset; aligned to the
        train order by file path). The cache is stored together with
        (backend, imgsz, file-list md5) metadata and is automatically
        recomputed on a mismatch — the old always-reuse cache kept using
        polluted values even after the scoring method changed.
        """
        import hashlib
        import json

        # Backend policy: 'train' (default) = the analyzer's training-time
        # backend (single source of truth for the curriculum signal);
        # 'cv2'/'gpu' force that backend for scoring only.
        # (Read from config directly: this method runs before the
        # self.curriculum_cfg attribute is assigned in __init__.)
        score_backend = str(
            self.config.get("curriculum", {}).get("score_backend", "train")
        ).lower()
        backend_arg = None if score_backend == "train" else score_backend
        analyzer_backend = getattr(
            getattr(self.model, "complexity_analyzer", None), "metric_backend", "gpu"
        )
        effective_backend = backend_arg or analyzer_backend

        train_files = [str(Path(p).resolve()) for p in self.train_dataset.im_files]
        files_md5 = hashlib.md5("\n".join(train_files).encode()).hexdigest()
        meta = {
            "version": 2,
            "augment": False,
            "backend": effective_backend,
            "imgsz": self.imgsz,
            "n": len(train_files),
            "files_md5": files_md5,
        }
        meta_path = self.complexity_path.with_suffix(".meta.json")

        if self.complexity_path.exists() and meta_path.exists():
            try:
                cached = json.loads(meta_path.read_text())
            except Exception:
                cached = None
            if cached == meta:
                print(f"[MCAQ] Loading complexity scores from {self.complexity_path}")
                scores_np = np.load(self.complexity_path)
                return torch.from_numpy(scores_np).float().to(self.device)
            print("[MCAQ] Cached complexity scores are stale "
                  "(backend/imgsz/file-list changed) — recomputing.")

        print("[MCAQ] Computing complexity scores for curriculum learning "
              f"(augment-free, backend={effective_backend})...")
        scoring_ds = self._build_scoring_dataset()

        # Algorithm 3 line 1: SortByComplexity uses the unified morphological
        # complexity C(x) (Eq.8) via the model's analyzer.
        scores_np = compute_dataset_complexity(
            dataset=scoring_ds,
            model=self.model,
            batch_size=self.batch_size,
            device=str(self.device),
            save_path=None,          # saved below, in train order
            backend=backend_arg,
        )

        # Align scoring order to the TRAIN dataset order by file path.
        score_files = [str(Path(p).resolve()) for p in scoring_ds.im_files]
        by_path = {p: float(s) for p, s in zip(score_files, scores_np)}
        missing = [p for p in train_files if p not in by_path]
        if missing:
            raise RuntimeError(
                f"[MCAQ] {len(missing)} train images missing from the scoring "
                f"dataset (first: {missing[0]}) — path alignment failed."
            )
        scores_np = np.asarray([by_path[p] for p in train_files], dtype=np.float32)

        np.save(self.complexity_path, scores_np)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved complexity scores to {self.complexity_path}")
        return torch.from_numpy(scores_np).float().to(self.device)

    def _make_teacher_hook(self, idx: int):
        """Store the teacher's feature map at backbone layer `idx` (FP32, frozen)."""
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                self._teacher_feats[idx] = output
        return hook

    def _get_curriculum_temperature(self, epoch: int) -> float:
        """
        Algorithm 3 line 10: alpha_t = 1 + 9*exp(-5t/T) — delegated to the
        CurriculumScheduler (single source of truth). Final temperature is 1.0
        (paper Table X), at which point bit allocation is fully adaptive.
        """
        if not self.curriculum_cfg.get("enabled", True):
            return 1.0
        return float(self.curriculum.get_temperature(epoch))

    def _build_curriculum_loader(self, tau_t: float):
        """
        Algorithm 3 line 9: D_t = {(x,y) in D_sorted : C(x) <= tau_t}.

        Builds a DataLoader restricted to samples whose precomputed complexity
        score is below the epoch's threshold. Falls back to the easiest samples
        when the threshold leaves too few. Returns the full train_loader once
        tau_t reaches 1.0 (post warm-up).
        """
        if tau_t >= 1.0 or self.complexity_scores is None:
            return self.train_loader

        scores = self.complexity_scores.detach().cpu().numpy()
        idx = np.where(scores <= tau_t)[0]

        min_needed = max(self.batch_size, 64)
        if len(idx) < min_needed:
            # Algorithm 3 sorts by complexity — fall back to the easiest samples
            idx = np.argsort(scores)[:min_needed]

        collate = getattr(self.train_dataset, "collate_fn", None)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(idx.tolist()),
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # Train / Eval Loop
    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        self.teacher_module.eval()

        # --- 3-stage curriculum state for this epoch (paper Fig.3 / Algorithm 3) ---
        stage = self.curriculum.get_stage(epoch)
        temp = self._get_curriculum_temperature(epoch)
        tau_t = self.curriculum.get_complexity_threshold(epoch)
        loss_weights = self.curriculum.get_loss_weights(epoch)  # lambda1(t) annealed
        # The curriculum's 8->4 bit-target schedule must be wired into Lbit —
        # a fixed target of 4 from epoch 1 exerts maximal collapse pressure
        # exactly when Ldet is absent (Stage 1 bypasses quantization).
        target_bits_t = float(self.curriculum.get_target_bits(epoch))
        # Stage 1: high-precision warm-up — quantization bypassed (FP16/AMP forward)
        quantize = stage >= 2

        kd_enabled = bool(self.distill_cfg.get("enabled", True))

        # Algorithm 3 line 9: restrict this epoch's data to C(x) <= tau_t
        epoch_loader = self._build_curriculum_loader(tau_t)

        # Accumulators for metrics
        total_loss = 0.0
        total_det_loss = 0.0
        total_bit_loss = 0.0
        total_smooth_loss = 0.0
        total_avg_bits = 0.0
        bit_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        n_batches = 0

        # Create progress bar
        pbar = tqdm(
            epoch_loader,
            desc=f"Epoch {epoch}/{self.epochs} [S{stage}]",
            unit="batch",
            ncols=150,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

        for batch_idx, batch in enumerate(pbar):
            imgs = batch["img"].to(self.device, non_blocking=True)
            # Ensure images are float and normalized to 0-1
            if imgs.dtype == torch.uint8:
                imgs = imgs.float() / 255.0
            elif imgs.dtype != torch.float32 and imgs.dtype != torch.float16:
                imgs = imgs.float()
            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                # student model forward (includes complexity + bit allocation;
                # in Stage 1 quantize=False for the high-precision warm-up)
                outputs, aux_info = self.model(imgs, temp, quantize=quantize)

                # FP32 teacher raw outputs for KD (paper Eq.20: L includes
                # lambda3 * LKD — composed inside the loss, not alpha-blended).
                # autocast is explicitly disabled so the teacher really runs in
                # FP32 even when AMP is on (paper: FP32 teacher).
                teacher_out = None
                if kd_enabled:
                    with torch.no_grad(), autocast("cuda", enabled=False):
                        self._teacher_feats.clear()
                        teacher_out = self.teacher_module(imgs.float())

                    # Feature-level KD: student's quantized C3/C4/C5 vs the
                    # teacher's FP32 features at the same layers
                    feat_losses = []
                    for li, fq in zip(
                        aux_info.get('feature_layers', []),
                        aux_info.get('quantized_features', []),
                    ):
                        tf = self._teacher_feats.get(li)
                        if tf is not None and tf.shape == fq.shape:
                            feat_losses.append(
                                F.mse_loss(fq.float(), tf.detach().float())
                            )
                    if feat_losses:
                        aux_info['kd_feature_loss'] = sum(feat_losses) / len(feat_losses)

                # Eq.(20): L = Ldet + lambda1(t) Lbit + lambda2 Lsmooth
                #            + lambda3 LKD + lambda4 Lreg
                # lambda4 applies to the bit-mapping network weights only.
                loss, loss_dict = self.model.loss_fn(
                    outputs,
                    batch,
                    aux_info,
                    teacher_outputs=teacher_out,
                    model_params=self.model.bit_mapper,
                    loss_weights=loss_weights,
                    target_bits=target_bits_t,
                )
                loss_det = loss_dict.get('loss_det', loss)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Paper Table X: gradient clipping = 1.0 (unscale before clipping under AMP)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Paper Table X: gradient clipping = 1.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Paper Eq.(18): Wi <- |Wi| — re-project the bit-mapping network onto
            # non-negative weights after every update to keep f(C) monotone
            self.model.bit_mapper.enforce_weight_constraints()

            # Accumulate metrics
            total_loss += loss.detach().item()
            total_det_loss += loss_dict.get('loss_det', torch.tensor(0.0)).detach().item() if isinstance(loss_dict.get('loss_det'), torch.Tensor) else loss_dict.get('loss_det', 0.0)
            total_bit_loss += loss_dict.get('loss_bit', torch.tensor(0.0)).detach().item() if isinstance(loss_dict.get('loss_bit'), torch.Tensor) else loss_dict.get('loss_bit', 0.0)
            total_smooth_loss += loss_dict.get('loss_smooth', torch.tensor(0.0)).detach().item() if isinstance(loss_dict.get('loss_smooth'), torch.Tensor) else loss_dict.get('loss_smooth', 0.0)

            # Track bit allocation
            avg_bits = aux_info.get('avg_bits', 0.0)
            if isinstance(avg_bits, torch.Tensor):
                avg_bits = avg_bits.item()
            total_avg_bits += avg_bits

            # Count bit distribution from bit_map (single map or per-scale list).
            # Training-time maps are continuous (fractional-bit quantization) —
            # round for the histogram.
            if 'bit_map' in aux_info:
                bit_map = aux_info['bit_map']
                maps = bit_map if isinstance(bit_map, (list, tuple)) else [bit_map]
                for m in maps:
                    if isinstance(m, torch.Tensor):
                        mr = torch.round(m.detach())
                        for bits in range(2, 9):
                            bit_counts[bits] += (mr == bits).sum().item()

            n_batches += 1

            # Update progress bar
            current_avg_bits = total_avg_bits / n_batches if n_batches > 0 else 0
            pbar.set_postfix({
                'loss': f'{total_loss/n_batches:.4f}',
                'det': f'{total_det_loss/n_batches:.4f}',
                'bits': f'{current_avg_bits:.2f}',
                'temp': f'{temp:.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        pbar.close()

        # Calculate averages
        avg_loss = total_loss / max(1, n_batches)
        avg_det_loss = total_det_loss / max(1, n_batches)
        avg_bit_loss = total_bit_loss / max(1, n_batches)
        avg_smooth_loss = total_smooth_loss / max(1, n_batches)
        avg_bits = total_avg_bits / max(1, n_batches)

        # Print epoch summary with bit distribution
        total_tiles = sum(bit_counts.values())
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Total Loss: {avg_loss:.4f} | Det Loss: {avg_det_loss:.4f} | Bit Loss: {avg_bit_loss:.6f} | Smooth Loss: {avg_smooth_loss:.6f}")
        print(f"  Average Bits: {avg_bits:.2f} | Temperature: {temp:.2f}")
        if total_tiles > 0:
            print(f"  Bit Distribution:")
            for bits in range(2, 9):
                count = bit_counts[bits]
                pct = (count / total_tiles) * 100
                bar = '█' * int(pct / 2)
                print(f"    {bits}-bit: {bar} {pct:.1f}% ({count:,})")
        print(f"{'='*80}\n")

        return {
            "loss": avg_loss,
            "det_loss": avg_det_loss,
            "bit_loss": avg_bit_loss,
            "smooth_loss": avg_smooth_loss,
            "avg_bits": avg_bits,
            "temperature": temp
        }

    @torch.no_grad()
    def evaluate(self, quantize: bool = True, temperature: float = 1.0,
                 compute_map_metric: bool = True) -> Dict[str, float]:
        """
        Validation loop: val_loss on every call, plus mAP@0.5 (used for
        best-checkpoint selection) when compute_map_metric is True.

        Args:
            quantize: evaluate under the same quantization state as the training
                stage (False during the Stage 1 warm-up — prevents a
                train/val regime mismatch)
            compute_map_metric: decode + NMS + AP over the full validation set.
                On large val sets this dominates evaluation cost — set
                training.map_interval > 1 in the config to amortize it
                (mAP50 is None on skipped epochs and best-checkpoint
                selection only updates on epochs where it is computed).
            temperature: REVIEW FIX — this used to be fixed at 1.0, so during
                annealing (Stage 2) training (alpha_t>1, bits saturated upward)
                and validation ran under different quantization regimes
                (contradicting the "same regime" comment; measured as an
                epoch-1 val_loss spike). It now takes the epoch's alpha_t
                directly so the train/val regimes match.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_dets, all_tgts = [], []

        for batch in self.val_loader:
            imgs = batch["img"].to(self.device, non_blocking=True)
            # Ensure images are float and normalized to 0-1
            if imgs.dtype == torch.uint8:
                imgs = imgs.float() / 255.0
            elif imgs.dtype != torch.float32 and imgs.dtype != torch.float16:
                imgs = imgs.float()

            with autocast("cuda", enabled=self.use_amp):
                outputs, aux_info = self.model(imgs, temperature=temperature, quantize=quantize)
                loss, loss_dict = self.model.loss_fn(
                    outputs,
                    batch,
                    aux_info,
                    model_params=self.model.bit_mapper,
                )
                loss_det = loss_dict.get('loss_det', loss)

            total_loss += loss_det.detach().item()
            n_batches += 1

            # REVIEW FIX: also measure mAP@0.5 — best-checkpoint selection should
            # be based on detection performance, not val loss (wires the AP
            # implementation in utils.evaluation into the training loop).
            if compute_map_metric:
                B, _, img_h, img_w = imgs.shape
                dets = _decode_outputs(outputs)
                tgts = extract_targets_per_image(batch, B, img_w, img_h)
                for d, t in zip(dets, tgts):
                    all_dets.append(d.float().cpu())
                    all_tgts.append(t)

        avg_loss = total_loss / max(1, n_batches)
        map50 = (compute_map(all_dets, all_tgts, iou_thresholds=[0.5])["mAP@0.5"]
                 if compute_map_metric else None)
        return {"val_loss": avg_loss, "mAP50": map50}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self) -> nn.Module:
        """
        Full training loop.
        """
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Output directory: {self.output_dir}")

        best_map = -1.0
        best_weights_path = self.output_dir / "best.pt"
        last_weights_path = self.output_dir / "last.pt"

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            stage = self.curriculum.get_stage(epoch)
            # Validate under the same quantization regime as training
            # (Stage 1 warm-up: high precision)
            # mAP cadence: every training.map_interval epochs (always on the
            # final epoch so short runs still report a number)
            want_map = (epoch % self.map_interval == 0) or (epoch == self.epochs)
            val_metrics = self.evaluate(
                quantize=stage >= 2,
                temperature=self._get_curriculum_temperature(epoch),
                compute_map_metric=want_map,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            map_str = (f", mAP@0.5: {val_metrics['mAP50']:.4f}"
                       if val_metrics["mAP50"] is not None else "")
            print(
                f"Epoch {epoch}/{self.epochs} [S{stage}] "
                f"- loss: {train_metrics['loss']:.4f}, "
                f"val_loss: {val_metrics['val_loss']:.4f}"
                f"{map_str}"
            )

            torch.save(self.model.state_dict(), last_weights_path)

            # REVIEW FIX (best-checkpoint selection): previously best.pt was
            # chosen by the minimum val_loss regardless of regime — Stage 1's
            # (non-quantized warm-up) loss is structurally lower, so unless the
            # quantized loss after entering Stage 2 dips below it, best.pt stays
            # pinned to the FP warm-up checkpoint forever. Now best is chosen by
            # the peak mAP@0.5 of the 'fully-quantized Stage 3' (a detection
            # model's best should be by AP, not by loss).
            if (stage >= 3 and val_metrics["mAP50"] is not None
                    and val_metrics["mAP50"] > best_map):
                best_map = val_metrics["mAP50"]
                torch.save(self.model.state_dict(), best_weights_path)

        if best_map < 0:
            # short run (Stage 3 not reached): save the final state as best, but announce it explicitly
            torch.save(self.model.state_dict(), best_weights_path)
            print("[MCAQ] Run ended before Stage 3 — best.pt is the FINAL model "
                  "(no quantized-regime mAP selection happened).")
        else:
            print(f"[MCAQ] Training finished. Best quantized mAP@0.5={best_map:.4f}")
        print(f"[MCAQ] Best weights saved to {best_weights_path}")

        return self.model


def main(argv: Optional[list] = None) -> None:
    """
    CLI entry point (`mcaq-yolo-train`): read the YAML config and run the Trainer.
    """
    parser = argparse.ArgumentParser(description="MCAQ-YOLO training")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--device", default=None, help="Override config device (cpu / cuda / mps)")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (review fix: reproducibility)")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed

    Trainer(config).train()


if __name__ == "__main__":
    main()
