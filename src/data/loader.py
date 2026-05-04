"""
Data loader for multilingual NMT using OPUS-100 dataset.
"""

from collections import Counter
from typing import List, Tuple, Dict, Optional
import math

from datasets import load_dataset, concatenate_datasets, DatasetDict


# Supported language configurations: (dataset_config, src_lang, tgt_lang)
LANG_CONFIGS = [
    ("en-vi", "en", "vi"),  # English -> Vietnamese
    ("en-fr", "en", "fr"),  # English -> French
    ("de-en", "de", "en"),  # German -> English
]

# Maximum samples per language pair (to control training time)
MAX_SAMPLES_PER_PAIR = 50000


def load_multilingual_dataset(
    lang_configs: List[Tuple[str, str, str]] = None,
    max_samples_per_pair: int = MAX_SAMPLES_PER_PAIR,
    val_size: int = 1000,
    test_size: int = 2000
) -> DatasetDict:
    """
    Load and concatenate multilingual dataset from OPUS-100.
    
    Args:
        lang_configs: List of (config_name, src_lang, tgt_lang) tuples
        max_samples_per_pair: Maximum samples to use per language pair
        val_size: Number of validation samples per pair
        test_size: Number of test samples per pair
    
    Returns:
        DatasetDict with 'train', 'validation', 'test' splits
    """
    if lang_configs is None:
        lang_configs = LANG_CONFIGS
    
    all_train, all_val, all_test = [], [], []
    
    for config_name, src, tgt in lang_configs:
        pair = f"{src}-{tgt}"
        print(f"Loading {pair}...")
        
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", config_name)
        except Exception as e:
            print(f"Warning: Could not load {config_name}: {e}")
            continue
        
        # Select training samples
        train = ds["train"].select(
            range(min(len(ds["train"]), max_samples_per_pair))
        )
        
        # Use validation set if available, otherwise use part of test
        if "validation" in ds:
            val = ds["validation"]
        else:
            val = ds["test"].select(range(min(val_size, len(ds["test"]) // 2)))
        
        # Select test samples
        test = ds["test"].select(
            range(min(test_size, len(ds["test"])))
        )
        
        # Rename and restructure the data
        train = train.map(
            lambda x: {
                "pair": pair,
                "src": x["translation"][src],
                "tgt": x["translation"][tgt]
            },
            remove_columns=["translation"]
        )
        
        val = val.map(
            lambda x: {
                "pair": pair,
                "src": x["translation"][src],
                "tgt": x["translation"][tgt]
            },
            remove_columns=["translation"]
        )
        
        test = test.map(
            lambda x: {
                "pair": pair,
                "src": x["translation"][src],
                "tgt": x["translation"][tgt]
            },
            remove_columns=["translation"]
        )
        
        all_train.append(train)
        all_val.append(val)
        all_test.append(test)
        print(f"  {pair}: train={len(train)}, val={len(val)}, test={len(test)}")
    
    if not all_train:
        raise ValueError("No language pairs could be loaded!")
    
    return DatasetDict({
        "train": concatenate_datasets(all_train),
        "validation": concatenate_datasets(all_val),
        "test": concatenate_datasets(all_test)
    })


def get_lang_pair_statistics(dataset) -> Dict[str, int]:
    """
    Calculate statistics for each language pair in the dataset.
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict
    
    Returns:
        Dictionary mapping pair names to sample counts
    """
    if isinstance(dataset, DatasetDict):
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset
    
    pair_counts = Counter(train_dataset["pair"])
    return dict(pair_counts)


def calculate_pair_weights(
    pair_counts: Dict[str, int],
    method: str = "sqrt"
) -> Dict[str, float]:
    """
    Calculate balanced loss weights for each language pair.
    
    Args:
        pair_counts: Dictionary mapping pair names to sample counts
        method: Weighting method ('sqrt' or 'inverse')
    
    Returns:
        Dictionary mapping pair names to weights
    """
    total = sum(pair_counts.values())
    
    if method == "sqrt":
        # Square root method: w = sqrt(N / c)
        # Less extreme than inverse, good for moderate imbalance
        weights = {
            pair: math.sqrt(total / count) 
            for pair, count in pair_counts.items()
        }
    elif method == "inverse":
        # Inverse method: w = N / c
        # More aggressive balancing
        weights = {
            pair: total / count 
            for pair, count in pair_counts.items()
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights


def get_sample_weights(
    dataset,
    pair_weights: Dict[str, float]
) -> List[float]:
    """
    Get sample-level weights based on language pair weights.
    
    Args:
        dataset: HuggingFace Dataset
        pair_weights: Dictionary mapping pair names to weights
    
    Returns:
        List of weights for each sample
    """
    return [pair_weights.get(pair, 1.0) for pair in dataset["pair"]]
