"""
MCAQ-YOLO: Morphological Complexity-Aware Quantization for YOLO
"""

__version__ = "0.2.1"
__author__ = "MCAQ-YOLO Team"

# REVIEW FIX (packaging): the previous eager `from .models.mcaq_yolo import
# MCAQYOLO` pulled `ultralytics` at package-import time, which (a) broke the
# advertised "CPU smoke tests run without ultralytics" claim — pytest failed
# at COLLECTION with ModuleNotFoundError (measured) — and (b) made even
# `import mcaq_yolo.core.morphology` impossible in minimal environments.
# Core classes stay eager (torch/cv2/scipy only); the model, loss, trainer
# and predictor are lazy via PEP 562 module __getattr__.

from .core.morphology import MorphologicalComplexityAnalyzer
from .core.bit_allocation import ComplexityToBitMappingNetwork, LinearBitMapper
from .core.quantization import SpatialAdaptiveQuantization
from .core.curriculum import CurriculumScheduler

_LAZY = {
    "MCAQYOLO": ".models.mcaq_yolo",
    "MCAQYOLOLoss": ".models.mcaq_yolo",
    "MCQLYOLOLoss": ".models.mcaq_yolo",  # legacy alias (original typo'd name)
    "Trainer": ".train",
    "Predictor": ".inference",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        mod = import_module(_LAZY[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache for subsequent lookups
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + list(_LAZY.keys())))


__all__ = [
    "MCAQYOLO",
    "MCAQYOLOLoss",
    "MCQLYOLOLoss",
    "Trainer",
    "Predictor",
    "MorphologicalComplexityAnalyzer",
    "ComplexityToBitMappingNetwork",
    "LinearBitMapper",
    "SpatialAdaptiveQuantization",
    "CurriculumScheduler",
]
