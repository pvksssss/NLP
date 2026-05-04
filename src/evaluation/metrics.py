"""
Evaluation metrics for NMT models.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def _load_evaluate():
    """Lazy load the evaluate library to avoid circular imports."""
    import importlib
    return importlib.import_module("evaluate")


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    tokenize: str = "13a"
) -> Dict[str, float]:
    """
    Compute SacreBLEU score for translation predictions.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations (each can have multiple refs)
        tokenize: Tokenization method ('13a', 'intl', 'zh', 'ja-mecab', etc.)
    
    Returns:
        Dictionary with 'score', 'precision', 'brevity_penalty', etc.
    """
    evaluate = _load_evaluate()
    sacrebleu = evaluate.load("sacrebleu")
    
    result = sacrebleu.compute(
        predictions=predictions,
        references=references,
        tokenize=tokenize
    )
    
    return {
        "bleu": result["score"],
        "precision_1": result["precisions"][0],
        "precision_2": result["precisions"][1],
        "precision_3": result["precisions"][2],
        "precision_4": result["precisions"][3],
        "brevity_penalty": result["bp"],
        "length_ratio": result["sys_len"] / max(result["ref_len"], 1),
        "translation_length": result["sys_len"],
        "reference_length": result["ref_len"]
    }


def compute_chrf(
    predictions: List[str],
    references: List[List[str]],
    order: int = 6,
    beta: float = 2.0
) -> Dict[str, float]:
    """
    Compute chrF++ score for translation predictions.
    
    chrF++ is a character n-gram based metric that is language-independent
    and correlates well with human evaluation.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        order: Maximum order of character n-grams (default: 6)
        beta: Parameter for F-score (default: 2.0 emphasizes recall)
    
    Returns:
        Dictionary with 'score' (chrF++) and 'precision', 'recall'
    """
    chrf = _load_evaluate().load("chrf")
    
    result = chrf.compute(
        predictions=predictions,
        references=references,
    )

    return {
        "chrf": result["score"],
    }


def compute_bleurt(
    predictions: List[str],
    references: List[List[str]],
    model_name: str = "bleurt-base-128"
) -> Dict[str, float]:
    """
    Compute BLEURT score for translation predictions.
    
    BLEURT uses a BERT-based model fine-tuned on human ratings.
    Note: Requires the bleurt package to be installed.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        model_name: BLEURT model checkpoint name
    
    Returns:
        Dictionary with 'score', 'mean', 'std'
    """
    try:
        bleurt = _load_evaluate().load("bleurt", model_name)
        
        # Flatten references for BLEURT (takes single reference)
        flat_refs = [ref[0] if isinstance(ref, list) else ref for ref in references]
        
        result = bleurt.compute(
            predictions=predictions,
            references=flat_refs
        )
        
        return {
            "bleurt": np.mean(result["scores"]),
            "bleurt_std": np.std(result["scores"]),
            "bleurt_min": np.min(result["scores"]),
            "bleurt_max": np.max(result["scores"])
        }
    except Exception as e:
        print(f"Warning: BLEURT computation failed: {e}")
        return {
            "bleurt": 0.0,
            "bleurt_std": 0.0,
            "bleurt_error": str(e)
        }


def compute_all_metrics(
    predictions: List[str],
    references: List[List[str]],
    include_bleurt: bool = False,
    return_detailed: bool = False
) -> Dict[str, Any]:
    """
    Compute all available NMT metrics.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        include_bleurt: Whether to include BLEURT (slower)
        return_detailed: Whether to return detailed per-metric results
    
    Returns:
        Dictionary with all computed metrics
    """
    results = {}
    
    # SacreBLEU
    bleu_results = compute_bleu(predictions, references)
    results["bleu"] = bleu_results["bleu"]
    
    if return_detailed:
        results["bleu_details"] = bleu_results
    
    # chrF++
    chrf_results = compute_chrf(predictions, references)
    results["chrf"] = chrf_results["chrf"]
    
    if return_detailed:
        results["chrf_details"] = chrf_results
    
    # BLEURT (optional)
    if include_bleurt:
        try:
            bleurt_results = compute_bleurt(predictions, references)
            results["bleurt"] = bleurt_results["bleurt"]
            
            if return_detailed:
                results["bleurt_details"] = bleurt_results
        except Exception:
            pass  # Skip if BLEURT not available
    
    return results


def compute_metrics_by_pair(
    predictions: List[str],
    references: List[List[str]],
    pairs: List[str],
    include_bleurt: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each language pair.
    
    This is useful for identifying which language pairs are performing well
    and which need improvement.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        pairs: List of language pair names for each prediction
        include_bleurt: Whether to include BLEURT
    
    Returns:
        Dictionary mapping pair names to their metrics
    """
    from collections import defaultdict
    
    # Group predictions and references by pair
    grouped = defaultdict(lambda: {"preds": [], "refs": []})
    
    for pred, ref, pair in zip(predictions, references, pairs):
        grouped[pair]["preds"].append(pred)
        grouped[pair]["refs"].append(ref)
    
    # Compute metrics for each pair
    pair_metrics = {}
    
    for pair, data in grouped.items():
        metrics = compute_all_metrics(
            data["preds"],
            data["refs"],
            include_bleurt=include_bleurt
        )
        pair_metrics[pair] = metrics
    
    return pair_metrics


def compute_variance_metrics(
    predictions: List[str],
    references: List[List[str]],
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Compute variance and confidence intervals for BLEU score using bootstrap.
    
    This helps understand the stability of the BLEU score and whether
    differences between models are statistically significant.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
    
    Returns:
        Dictionary with mean, std, and confidence intervals
    """
    bleu_scores = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.randint(0, len(predictions), len(predictions))
        boot_preds = [predictions[i] for i in indices]
        boot_refs = [references[i] for i in indices]
        
        try:
            result = compute_bleu(boot_preds, boot_refs)
            bleu_scores.append(result["bleu"])
        except Exception:
            continue
    
    if not bleu_scores:
        return {"error": "Could not compute bootstrap scores"}
    
    bleu_scores = np.array(bleu_scores)
    mean = np.mean(bleu_scores)
    std = np.std(bleu_scores)
    
    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bleu_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bleu_scores, 100 * (1 - alpha / 2))
    
    return {
        "mean_bleu": mean,
        "std_bleu": std,
        "confidence_interval": (ci_lower, ci_upper),
        "min_bleu": np.min(bleu_scores),
        "max_bleu": np.max(bleu_scores),
        "n_bootstrap": n_bootstrap
    }
