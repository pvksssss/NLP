"""
Models module for multilingual NMT.
"""

from .mbart import load_mbart_model, MBNMTModel

__all__ = [
    'load_mbart_model',
    'MBNMTModel'
]
