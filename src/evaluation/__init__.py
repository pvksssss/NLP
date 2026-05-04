"""
Evaluation module for multilingual NMT.
"""

from .metrics import compute_bleu, compute_chrf, compute_all_metrics
from .evaluator import NMTEvaluator

__all__ = [
    'compute_bleu',
    'compute_chrf',
    'compute_all_metrics',
    'NMTEvaluator'
]
