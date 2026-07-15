"""
Reproducibility utilities (review fix: the pipeline had no seed control,
so no reported number was re-runnable bit-for-bit).
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Fix RNG state across `random`, NumPy and torch (+ all CUDA devices).

    With deterministic=True, additionally request deterministic kernels:
    cudnn.deterministic on, cudnn.benchmark off,
    torch.use_deterministic_algorithms(warn_only=True), and the CUBLAS
    workspace env var required by deterministic GEMMs.

    NOTES / LIMITS (state plainly, per review):
    - Ultralytics' InfiniteDataLoader seeds its generator and workers from
      torch's global RNG, which this call fixes — but augmentation order is
      only reproducible for identical num_workers and batch_size.
    - Some GPU ops have no deterministic implementation; warn_only=True
      degrades gracefully instead of crashing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # torch < 1.11 signature
            torch.use_deterministic_algorithms(True)
