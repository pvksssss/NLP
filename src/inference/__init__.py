"""
Inference module for multilingual NMT.
"""

from .translator import MultilingualTranslator, translate_text, translate_batch

__all__ = [
    'MultilingualTranslator',
    'translate_text',
    'translate_batch'
]
