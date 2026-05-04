"""
Data preprocessing for multilingual NMT.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
from transformers import PreTrainedTokenizer

from .loader import LANG_CONFIGS


# Language code mapping for mBART-50
LANG_TO_MBART = {
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

# Default max sequence length
DEFAULT_MAX_LEN = 128


def preprocess_function(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = DEFAULT_MAX_LEN,
    lang_to_mbart: Dict[str, str] = None
) -> Dict[str, List[Any]]:
    """
    Tokenize and preprocess examples for NMT model.
    
    Args:
        examples: Dictionary containing 'src', 'tgt', 'pair' lists
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        lang_to_mbart: Mapping from language codes to mBART language codes
    
    Returns:
        Dictionary with tokenized inputs and labels
    """
    if lang_to_mbart is None:
        lang_to_mbart = LANG_TO_MBART
    
    src_texts = examples["src"]
    tgt_texts = examples["tgt"]
    pairs = examples["pair"]
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    pair_list = []
    
    for src, tgt, pair in zip(src_texts, tgt_texts, pairs):
        src_lang, tgt_lang = pair.split("-")
        
        # Set tokenizer language
        tokenizer.src_lang = lang_to_mbart.get(src_lang, src_lang)
        tokenizer.tgt_lang = lang_to_mbart.get(tgt_lang, tgt_lang)
        
        # Get target language token ID for forced BOS
        target_lang_token_id = tokenizer.lang_code_to_id[lang_to_mbart.get(tgt_lang, tgt_lang)]
        
        # Tokenize source
        enc = tokenizer(
            src,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        
        # Tokenize target (labels) with text_target parameter
        dec = tokenizer(
            text_target=tgt,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        
        dec_input_ids = dec["input_ids"]
        if dec_input_ids:
            dec_input_ids[0] = target_lang_token_id
        
        # Replace padding tokens in labels with -100 for loss computation
        labels = [-100 if t == tokenizer.pad_token_id else t for t in dec_input_ids]
        
        input_ids_list.append(enc["input_ids"])
        attention_mask_list.append(enc["attention_mask"])
        labels_list.append(labels)
        pair_list.append(pair)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "pair": pair_list
    }


def preprocess_function_notebook_style(examples: Dict[str, List[Any]], tokenizer: PreTrainedTokenizer, max_length: int = 128) -> Dict[str, List[Any]]:
    """
    Preprocess function matching the notebook style exactly.
    
    This version uses text_target parameter and sets target_lang_token_id.
    """
    src_tgt = list(zip(examples["src"], examples["tgt"], examples["pair"]))
    input_ids_list, attention_mask_list, labels_list, pair_list = [], [], [], []
    
    for src, tgt, pair in src_tgt:
        src_lang, tgt_lang = pair.split("-")
        tokenizer.src_lang = LANG_TO_MBART[src_lang]
        tokenizer.tgt_lang = LANG_TO_MBART[tgt_lang]
        target_lang_token_id = tokenizer.lang_code_to_id[LANG_TO_MBART[tgt_lang]]
        
        enc = tokenizer(src, padding="max_length", truncation=True, max_length=max_length)
        dec = tokenizer(text_target=tgt, padding="max_length", truncation=True, max_length=max_length)
        
        dec_input_ids = dec["input_ids"]
        if dec_input_ids:
            dec_input_ids[0] = target_lang_token_id
        
        labels = [-100 if t == tokenizer.pad_token_id else t for t in dec_input_ids]
        
        input_ids_list.append(enc["input_ids"])
        attention_mask_list.append(enc["attention_mask"])
        labels_list.append(labels)
        pair_list.append(pair)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "pair": pair_list
    }


@dataclass
class DataCollatorForNMT:
    """
    Data collator for NMT that handles padding and batch creation.
    """
    tokenizer: PreTrainedTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.
        
        Args:
            examples: List of examples from dataset
        
        Returns:
            Dictionary of batched tensors
        """
        # Extract input_ids, attention_mask, labels
        input_ids = torch.tensor([e["input_ids"] for e in examples])
        attention_mask = torch.tensor([e["attention_mask"] for e in examples])
        labels = torch.tensor([e["labels"] for e in examples])
        
        # Prepare batch dictionary
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Add language pair information if available
        if examples and "pair" in examples[0]:
            batch["pair"] = [e["pair"] for e in examples]
        
        return batch


def visualize_data_statistics(dataset, tokenizer, max_samples: int = 5000) -> Dict[str, Any]:
    """
    Calculate and visualize data statistics.
    
    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer for calculating token lengths
        max_samples: Maximum samples to analyze
    
    Returns:
        Dictionary containing statistics
    """
    from collections import Counter
    
    # Analyze a subset for efficiency
    samples = min(max_samples, len(dataset))
    indices = range(0, len(dataset), len(dataset) // samples)
    subset = dataset.select(indices)
    
    # Calculate lengths
    src_lengths = []
    tgt_lengths = []
    pair_counts = Counter()
    
    for example in subset:
        src_lengths.append(len(example["src"].split()))
        tgt_lengths.append(len(example["tgt"].split()))
        pair_counts[example["pair"]] += 1
    
    stats = {
        "num_samples": len(dataset),
        "num_pairs": len(pair_counts),
        "pair_distribution": dict(pair_counts),
        "avg_src_length": sum(src_lengths) / len(src_lengths),
        "avg_tgt_length": sum(tgt_lengths) / len(tgt_lengths),
        "max_src_length": max(src_lengths),
        "max_tgt_length": max(tgt_lengths),
    }
    
    return stats
