"""
Main package initialization for Multilingual NMT.
"""

from .data import load_multilingual_dataset, preprocess_function
from .models import load_mbart_model
from .training import BalancedLossSeq2SeqTrainer, create_trainer
from .evaluation import compute_all_metrics, NMTEvaluator
from .inference import MultilingualTranslator

__version__ = "1.0.0"
__author__ = "NLP Course - AI VIETNAM"

__all__ = [
    # Data
    "load_multilingual_dataset",
    "preprocess_function",
    # Models
    "load_mbart_model",
    # Training
    "BalancedLossSeq2SeqTrainer",
    "create_trainer",
    # Evaluation
    "compute_all_metrics",
    "NMTEvaluator",
    # Inference
    "MultilingualTranslator",
]
