"""Multilingual translation inference module."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


class MultilingualTranslator:
    """
    Easy-to-use multilingual translator with mBART-50.
    
    Example:
        translator = MultilingualTranslator("facebook/mbart-large-50-many-to-many-mmt")
        
        # Single translation
        result = translator.translate("Hello, how are you?", src_lang="en", tgt_lang="vi")
        print(result)  # "Xin chào, bạn khỏe không?"
        
        # Batch translation
        results = translator.translate_batch(
            ["Hello", "Goodbye"],
            src_lang="en",
            tgt_lang="fr"
        )
    """
    
    SUPPORTED_LANGS = {
        "en", "vi", "fr", "de", "es", "ar", "zh", "ru", "ja", "ko",
        "it", "pt", "nl", "pl", "tr", "ro", "cs", "hi", "th", "uk"
    }
    
    LANG_CODE_MAP = {
        "en": "en_XX",
        "vi": "vi_VN",
        "fr": "fr_XX",
        "de": "de_DE",
        "es": "es_XX",
        "ar": "ar_AR",
        "zh": "zh_CN",
        "ru": "ru_RU",
        "ja": "ja_XX",
        "ko": "ko_KR",
        "it": "it_IT",
        "pt": "pt_XX",
        "nl": "nl_XX",
        "pl": "pl_PL",
        "tr": "tr_TR",
        "ro": "ro_RO",
    }
    
    def __init__(
        self,
        model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the translator.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Cache directory for model files
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=False
        )
        
        print(f"Model loaded on {self.device}")
    
    def _get_lang_code(self, lang: str) -> str:
        """Get mBART language code."""
        return self.LANG_CODE_MAP.get(lang, lang)
    
    def _validate_langs(self, src_lang: str, tgt_lang: str):
        """Validate language codes."""
        if src_lang not in self.SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported source language: {src_lang}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_LANGS))}"
            )
        if tgt_lang not in self.SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported target language: {tgt_lang}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_LANGS))}"
            )
    
    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 128,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = None,
        top_p: float = None,
        **generate_kwargs
    ) -> str:
        """
        Translate a single text.
        
        Args:
            text: Source text
            src_lang: Source language code
            tgt_lang: Target language code
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (only used if do_sample=True)
            do_sample: Whether to use sampling
            top_k: Top-k for sampling
            top_p: Top-p for sampling
            **generate_kwargs: Additional arguments for generate
        
        Returns:
            Translated text
        """
        self._validate_langs(src_lang, tgt_lang)
        
        # Set tokenizer language
        self.tokenizer.src_lang = self._get_lang_code(src_lang)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get forced bos token for target language
        forced_bos_id = self.tokenizer.lang_code_to_id.get(
            self._get_lang_code(tgt_lang)
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_k=top_k if do_sample and top_k else None,
                top_p=top_p if do_sample and top_p else None,
                **generate_kwargs
            )
        
        # Decode
        translation = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return translation
    
    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        max_length: int = 128,
        batch_size: int = 8,
        num_beams: int = 4,
        show_progress: bool = True,
        **generate_kwargs
    ) -> List[str]:
        """
        Translate multiple texts in batches.
        
        Args:
            texts: List of source texts
            src_lang: Source language code
            tgt_lang: Target language code
            max_length: Maximum generation length
            batch_size: Batch size
            num_beams: Number of beams for beam search
            show_progress: Show progress bar
            **generate_kwargs: Additional arguments for generate
        
        Returns:
            List of translated texts
        """
        self._validate_langs(src_lang, tgt_lang)
        
        all_translations = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Translating")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Set tokenizer language
            self.tokenizer.src_lang = self._get_lang_code(src_lang)
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get forced bos token
            forced_bos_id = self.tokenizer.lang_code_to_id.get(
                self._get_lang_code(tgt_lang)
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    **generate_kwargs
                )
            
            # Decode batch
            batch_translations = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            all_translations.extend(batch_translations)
        
        return all_translations
    
    def translate_with_strategy(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        strategy: str = "beam",
        **kwargs
    ) -> str:
        """
        Translate with a specific decoding strategy.
        
        Args:
            text: Source text
            src_lang: Source language code
            tgt_lang: Target language code
            strategy: Decoding strategy ('greedy', 'beam', 'top_k', 'top_p', 'nucleus')
            **kwargs: Additional arguments
        
        Returns:
            Translated text
        """
        strategies = {
            "greedy": {"num_beams": 1, "do_sample": False},
            "beam": {"num_beams": 4, "do_sample": False},
            "top_k": {"num_beams": 1, "do_sample": True, "top_k": 50, "temperature": 0.7},
            "top_p": {"num_beams": 1, "do_sample": True, "top_p": 0.9, "temperature": 0.7},
            "top_k_p": {"num_beams": 1, "do_sample": True, "top_k": 50, "top_p": 0.9, "temperature": 0.7},
            "nucleus": {"num_beams": 1, "do_sample": True, "top_p": 0.95, "temperature": 0.7},
        }
        
        params = strategies.get(strategy.lower())
        if params is None:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Choose from: {', '.join(strategies.keys())}"
            )
        
        params.update(kwargs)
        return self.translate(text, src_lang, tgt_lang, **params)


class OpusMTTranslator:
    """Pairwise translator backed by separate Helsinki-NLP/opus-mt-* models.

    This is useful when you don't have a single multilingual checkpoint and instead
    want to route by (src_lang, tgt_lang) pair.
    """

    DEFAULT_MODEL_MAP: Dict[Tuple[str, str], str] = {
        ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    }

    def __init__(
        self,
        model_map: Optional[Dict[Tuple[str, str], str]] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.cache_dir = cache_dir
        self.model_map: Dict[Tuple[str, str], str] = model_map or dict(self.DEFAULT_MODEL_MAP)

        self._models: Dict[Tuple[str, str], PreTrainedModel] = {}
        self._tokenizers: Dict[Tuple[str, str], PreTrainedTokenizer] = {}

    def supported_pairs(self) -> List[Tuple[str, str]]:
        return sorted(self.model_map.keys())

    def preload_all(self) -> None:
        """Eagerly load all configured (tokenizer, model) pairs into memory."""
        for src_lang, tgt_lang in self.supported_pairs():
            self._load_pair(src_lang, tgt_lang)

    def _resolve_pair(self, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        return (src_lang, tgt_lang)

    def _load_pair(self, src_lang: str, tgt_lang: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        pair = self._resolve_pair(src_lang, tgt_lang)
        model_name = self.model_map.get(pair)
        if model_name is None:
            supported = ", ".join([f"{s}->{t}" for s, t in self.supported_pairs()])
            raise ValueError(f"Unsupported language pair: {src_lang}->{tgt_lang}. Supported: {supported}")

        if pair not in self._models:
            print(f"Loading model for {src_lang}->{tgt_lang}: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.cache_dir)
            model.to(self.device)
            model.eval()
            self._tokenizers[pair] = tokenizer
            self._models[pair] = model

        return self._tokenizers[pair], self._models[pair]

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 128,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **generate_kwargs,
    ) -> str:
        tokenizer, model = self._load_pair(src_lang, tgt_lang)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_k=top_k if do_sample and top_k else None,
                top_p=top_p if do_sample and top_p else None,
                **generate_kwargs,
            )

        return tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def translate_with_strategy(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        strategy: str = "beam",
        **kwargs,
    ) -> str:
        strategies = {
            "greedy": {"num_beams": 1, "do_sample": False},
            "beam": {"num_beams": 4, "do_sample": False},
            "top_k": {"num_beams": 1, "do_sample": True, "top_k": 50, "temperature": 0.7},
            "top_p": {"num_beams": 1, "do_sample": True, "top_p": 0.9, "temperature": 0.7},
            "top_k_p": {"num_beams": 1, "do_sample": True, "top_k": 50, "top_p": 0.9, "temperature": 0.7},
            "nucleus": {"num_beams": 1, "do_sample": True, "top_p": 0.95, "temperature": 0.7},
        }

        params = strategies.get(strategy.lower())
        if params is None:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Choose from: {', '.join(strategies.keys())}"
            )

        params.update(kwargs)
        return self.translate(text, src_lang, tgt_lang, **params)


# Convenience functions
def translate_text(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
    device: str = None
) -> str:
    """
    Translate a single text (convenience function).
    
    Args:
        text: Source text
        src_lang: Source language code
        tgt_lang: Target language code
        model_name: Model name
        device: Device to use
    
    Returns:
        Translated text
    """
    translator = MultilingualTranslator(model_name, device)
    return translator.translate(text, src_lang, tgt_lang)


def translate_batch(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
    device: str = None,
    batch_size: int = 8,
    show_progress: bool = True
) -> List[str]:
    """
    Translate multiple texts (convenience function).
    
    Args:
        texts: List of source texts
        src_lang: Source language code
        tgt_lang: Target language code
        model_name: Model name
        device: Device to use
        batch_size: Batch size
        show_progress: Show progress bar
    
    Returns:
        List of translated texts
    """
    translator = MultilingualTranslator(model_name, device)
    return translator.translate_batch(
        texts,
        src_lang,
        tgt_lang,
        batch_size=batch_size,
        show_progress=show_progress,
    )
