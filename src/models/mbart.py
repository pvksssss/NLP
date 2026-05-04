"""
mBART-50 model wrapper for multilingual NMT.
"""

from typing import Optional, Dict, Tuple, Any
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_mbart_model(
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
    device: str = None,
    cache_dir: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Load mBART-50 model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or local path
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        cache_dir: Cache directory for model files
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=False
    )
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {model.num_parameters():,}")
    
    return model, tokenizer


class MBNMTModel:
    """
    Wrapper class for mBART-50 multilingual NMT.
    
    Provides a clean interface for:
    - Model loading and initialization
    - Translation with different strategies
    - Model info and diagnostics
    
    Example:
        nmt = MBNMTModel("facebook/mbart-large-50-many-to-many-mmt")
        translation = nmt.translate("Hello", src="en", tgt="vi")
    """
    
    # Supported languages with mBART language codes
    LANG_CODES = {
        "en": "en_XX", "vi": "vi_VN", "fr": "fr_XX", "de": "de_DE",
        "es": "es_XX", "ar": "ar_AR", "zh": "zh_CN", "ru": "ru_RU",
        "ja": "ja_XX", "ko": "ko_KR", "it": "it_IT", "pt": "pt_XX",
        "nl": "nl_XX", "pl": "pl_PL", "tr": "tr_TR", "ro": "ro_RO",
        "cs": "cs_CZ", "hi": "hi_IN", "th": "th_TH", "uk": "uk_UA"
    }
    
    def __init__(
        self,
        model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the NMT model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Cache directory
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, self.tokenizer = load_mbart_model(
            model_name,
            self.device,
            cache_dir
        )
        
        # Cache lang code IDs
        self.lang_code_to_id = self.tokenizer.lang_code_to_id
    
    def get_lang_code(self, lang: str) -> str:
        """Get mBART language code for a language."""
        return self.LANG_CODES.get(lang, lang)
    
    def get_lang_id(self, lang: str) -> int:
        """Get the tokenizer ID for a language code."""
        lang_code = self.get_lang_code(lang)
        return self.tokenizer.lang_code_to_id.get(lang_code, 0)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return self.model.num_parameters()
    
    @property
    def max_length(self) -> int:
        """Get model maximum sequence length."""
        return self.model.config.max_position_embeddings
    
    def get_model_info(self) -> Dict:
        """Get model information dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "vocab_size": self.vocab_size,
            "num_parameters": self.num_parameters,
            "max_length": self.max_length,
            "supported_languages": list(self.LANG_CODES.keys()),
            "num_languages": len(self.LANG_CODES)
        }
    
    def summary(self) -> str:
        """Get a summary string of the model."""
        info = self.get_model_info()
        num_params = f"{info['num_parameters']:,}"
        return f"""
╔══════════════════════════════════════════════════════════╗
║             mBART-50 Multilingual NMT Model             ║
╠══════════════════════════════════════════════════════════╣
║ Model: {info['model_name']:<50} ║
║ Device: {info['device']:<50} ║
║ Vocabulary Size: {info['vocab_size']:<39} ║
║ Parameters: {num_params:<44} ║
║ Max Length: {info['max_length']:<47} ║
║ Languages: {info['num_languages']:<47} ║
╚══════════════════════════════════════════════════════════╝
        """.strip()
