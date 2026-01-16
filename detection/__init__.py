"""
Detection module for hallucination detection.

Provides:
- DivergenceMetrics: Compute metrics from tracks
- HallucinationDetector: Binary classifier for hallucination detection
"""

from .divergence_metrics import DivergenceMetrics, compute_divergence_score
from .detector import HallucinationDetector

__all__ = [
    'DivergenceMetrics',
    'compute_divergence_score',
    'HallucinationDetector'
]
