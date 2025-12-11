"""
MCAQ-YOLO: Morphological Complexity-Aware Quantization for YOLO
"""

__version__ = "0.1.0"
__author__ = "MCAQ-YOLO Team"

from .models.mcaq_yolo import MCAQYOLO
from .core.morphology import MorphologicalComplexityAnalyzer
from .core.bit_allocation import ComplexityToBitMappingNetwork
from .core.quantization import SpatialAdaptiveQuantization
from .core.curriculum import CurriculumScheduler

__all__ = [
    "MCAQYOLO",
    "MorphologicalComplexityAnalyzer",
    "ComplexityToBitMappingNetwork",
    "SpatialAdaptiveQuantization",
    "CurriculumScheduler",
]