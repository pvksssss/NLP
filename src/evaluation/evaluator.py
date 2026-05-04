"""
NMT Evaluator class for comprehensive model evaluation.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import json
import os
from collections import defaultdict

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .metrics import (
    compute_all_metrics, 
    compute_metrics_by_pair,
    compute_variance_metrics
)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    overall_metrics: Dict[str, float]
    metrics_by_pair: Dict[str, Dict[str, float]]
    variance_metrics: Optional[Dict[str, Any]] = None
    predictions: Optional[List[str]] = None
    references: Optional[List[str]] = None
    pairs: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_metrics": self.overall_metrics,
            "metrics_by_pair": self.metrics_by_pair,
            "variance_metrics": self.variance_metrics,
            "num_samples": len(self.predictions) if self.predictions else 0
        }
    
    def save(self, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 60]
        lines.append("EVALUATION RESULTS")
        lines.append("=" * 60)
        
        lines.append("\nOverall Metrics:")
        for metric, value in self.overall_metrics.items():
            lines.append(f"  {metric.upper()}: {value:.2f}")

        lines.append("\nMetrics by Language Pair:")
        for pair, metrics in sorted(self.metrics_by_pair.items()):
            lines.append(f"  [{pair}]")
            for metric, value in metrics.items():
                lines.append(f"    {metric}: {value:.2f}")

        if self.variance_metrics:
            lines.append("\nVariance Analysis:")
            lines.append(f"  Mean BLEU: {self.variance_metrics.get('mean_bleu', 0):.2f}")
            lines.append(f"  Std BLEU: {self.variance_metrics.get('std_bleu', 0):.2f}")
            ci = self.variance_metrics.get('confidence_interval', (0, 0))
            lines.append(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class NMTEvaluator:
    """
    Comprehensive evaluator for Neural Machine Translation models.
    
    This class provides:
    - Batch inference on datasets
    - Multiple metrics computation (BLEU, chrF++, BLEURT)
    - Per-pair evaluation
    - Variance analysis with bootstrap
    - Result export and visualization
    
    Example:
        evaluator = NMTEvaluator(model, tokenizer)
        results = evaluator.evaluate(test_dataset)
        print(results.summary())
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = None,
        max_length: int = 128,
        use_fast_tokenizer: bool = False
    ):
        """
        Initialize the NMT evaluator.
        
        Args:
            model: The translation model to evaluate
            tokenizer: Tokenizer for preprocessing
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            max_length: Maximum generation length
            use_fast_tokenizer: Whether to use fast tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = max_length
        
        # Language code mapping for mBART
        self.lang_to_mbart = {
            "en": "en_XX", "vi": "vi_VN", "fr": "fr_XX",
            "de": "de_DE", "es": "es_XX", "ar": "ar_AR",
            "zh": "zh_CN", "ru": "ru_RU"
        }
    
    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = None,
        top_p: float = None,
        **generate_kwargs
    ) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of source texts
            src_lang: Source language code
            tgt_lang: Target language code
            num_beams: Number of beams for beam search (use > 1 for beam search)
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k for sampling
            top_p: Top-p (nucleus) for sampling
            **generate_kwargs: Additional arguments for model.generate
        
        Returns:
            List of translated texts
        """
        # Set tokenizer language
        self.tokenizer.src_lang = self.lang_to_mbart.get(src_lang, src_lang)
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set forced bos token for target language
        forced_bos_id = self.tokenizer.lang_code_to_id.get(
            self.lang_to_mbart.get(tgt_lang, tgt_lang)
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=self.max_length,
                num_beams=num_beams,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_k=top_k if do_sample and top_k else None,
                top_p=top_p if do_sample and top_p else None,
                **generate_kwargs
            )
        
        # Decode
        translations = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return translations
    
    def evaluate(
        self,
        dataset,
        src_lang: str = "en",
        tgt_lang: str = "vi",
        batch_size: int = 16,
        num_beams: int = 4,
        compute_variance: bool = False,
        include_bleurt: bool = False,
        show_progress: bool = True
    ) -> EvaluationResult:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: HuggingFace Dataset with 'src', 'tgt', 'pair' columns
            src_lang: Default source language
            tgt_lang: Default target language
            batch_size: Batch size for inference
            num_beams: Number of beams for generation
            compute_variance: Whether to compute variance with bootstrap
            include_bleurt: Whether to include BLEURT metric
            show_progress: Whether to show progress bar
        
        Returns:
            EvaluationResult object with all metrics
        """
        all_predictions = []
        all_references = []
        all_pairs = []
        
        # Process in batches
        n_batches = (len(dataset) + batch_size - 1) // batch_size
        
        iterator = range(0, len(dataset), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Evaluating")
        
        for i in iterator:
            batch = dataset[i:i + batch_size]
            
            # Get source texts and language pairs
            batch_texts = batch["src"]
            batch_pairs = batch.get("pair", [f"{src_lang}-{tgt_lang}"] * len(batch_texts))
            batch_refs = batch["tgt"]

            # Translate in true batches, grouped by language pair
            batch_predictions: List[Optional[str]] = [None] * len(batch_texts)
            grouped_indices: Dict[tuple, List[int]] = defaultdict(list)

            for idx, pair in enumerate(batch_pairs):
                pair_str = str(pair)
                if "-" in pair_str:
                    src, tgt = pair_str.split("-", 1)
                else:
                    src, tgt = src_lang, tgt_lang
                grouped_indices[(src, tgt, pair_str)].append(idx)

            for (src, tgt, _pair_str), indices in grouped_indices.items():
                texts = [batch_texts[idx] for idx in indices]
                preds = self.translate_batch(
                    texts,
                    src,
                    tgt,
                    num_beams=num_beams,
                )
                for pred, idx in zip(preds, indices):
                    batch_predictions[idx] = pred

            for pred, ref, pair in zip(batch_predictions, batch_refs, batch_pairs):
                if pred is None:
                    pred = ""
                all_predictions.append(pred)
                all_references.append([ref])  # BLEU expects list of references
                all_pairs.append(str(pair))
        
        # Compute overall metrics
        overall_metrics = compute_all_metrics(
            all_predictions,
            all_references,
            include_bleurt=include_bleurt
        )
        
        # Compute metrics by pair
        metrics_by_pair = compute_metrics_by_pair(
            all_predictions,
            all_references,
            all_pairs,
            include_bleurt=include_bleurt
        )
        
        # Compute variance metrics if requested
        variance_metrics = None
        if compute_variance:
            variance_metrics = compute_variance_metrics(
                all_predictions,
                all_references,
                n_bootstrap=100
            )
        
        return EvaluationResult(
            overall_metrics=overall_metrics,
            metrics_by_pair=metrics_by_pair,
            variance_metrics=variance_metrics,
            predictions=all_predictions,
            references=all_references,
            pairs=all_pairs
        )
    
    def compare_strategies(
        self,
        dataset,
        src_lang: str = "en",
        tgt_lang: str = "vi",
        batch_size: int = 16,
        sample_size: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different decoding strategies.
        
        Strategies compared:
        - Greedy decoding
        - Beam search
        - Top-k sampling
        - Top-p (nucleus) sampling
        
        Args:
            dataset: HuggingFace Dataset
            src_lang: Source language
            tgt_lang: Target language
            batch_size: Batch size
            sample_size: Number of samples to evaluate
        
        Returns:
            Dictionary mapping strategy names to metrics
        """
        # Use subset for faster comparison
        eval_dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        strategies = {
            "greedy": {"num_beams": 1, "do_sample": False},
            "beam_search": {"num_beams": 4, "do_sample": False},
            "top_k": {"num_beams": 1, "do_sample": True, "top_k": 50, "temperature": 0.7},
            "top_p": {"num_beams": 1, "do_sample": True, "top_p": 0.9, "temperature": 0.7},
            "top_k_p": {"num_beams": 1, "do_sample": True, "top_k": 50, "top_p": 0.9, "temperature": 0.7},
        }
        
        results = {}
        
        for strategy_name, params in strategies.items():
            print(f"Evaluating {strategy_name}...")
            
            # Translate with this strategy
            preds = []
            refs = []
            
            for i in range(0, len(eval_dataset), batch_size):
                batch = eval_dataset[i:i + batch_size]
                
                batch_preds = self.translate_batch(
                    batch["src"],
                    src_lang,
                    tgt_lang,
                    **params
                )
                
                preds.extend(batch_preds)
                refs.extend([[r] for r in batch["tgt"]])
            
            # Compute metrics
            metrics = compute_all_metrics(preds, refs)
            results[strategy_name] = metrics
        
        return results
