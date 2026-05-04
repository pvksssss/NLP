"""
Custom Trainer with Balanced Loss for multilingual NMT.
"""

from typing import Dict, Optional, Any, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, PreTrainedModel
from transformers.trainer_utils import PredictionOutput


class BalancedLossSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2Seq Trainer that implements balanced loss weighting by language pair.
    
    This trainer addresses data imbalance in multilingual NMT by applying higher weights
    to language pairs with fewer training samples.
    
    The balanced loss formula:
        L_balanced = Σ(w_pair * L_sample) / Σ(w_pair)
    
    where:
        - L_sample is the loss for each sample (averaged over valid tokens)
        - w_pair is the weight for the language pair (computed as sqrt(N_total / N_pair))
    """
    
    def __init__(
        self,
        pair_weights: Optional[Dict[str, float]] = None,
        *args,
        **kwargs
    ):
        """
        Initialize the trainer with balanced loss support.
        
        Args:
            pair_weights: Dictionary mapping language pairs to their weights
            *args: Additional positional arguments for Seq2SeqTrainer
            **kwargs: Additional keyword arguments for Seq2SeqTrainer
        """
        super().__init__(*args, **kwargs)
        self.pair_weights = pair_weights or {}
    
    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs
    ) -> PredictionOutput:
        """
        Override prediction_step to remove non-tensor fields.
        
        The 'pair' field is removed because the model doesn't accept it as input,
        but it's needed for loss computation.
        """
        # Remove non-tensor fields that shouldn't go to the model
        inputs = {k: v for k, v in inputs.items() if k != "pair"}
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
    
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute balanced loss weighted by language pair.
        
        The loss is computed per sample and then weighted by the language pair weight.
        This ensures that language pairs with fewer samples contribute equally to training.
        """
        # Remove 'pair' from inputs as it's not a model parameter
        if "pair" not in inputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        
        pairs = inputs.pop("pair")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        # Compute cross-entropy loss per token
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction='none'
        )
        
        # Reshape logits and labels for loss computation
        # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # labels: (batch_size, seq_len) -> (batch_size * seq_len)
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Compute per-token loss
        loss_per_token = loss_fct(logits_flat, labels_flat)
        
        # Reshape to (batch_size, seq_len) for sample-level averaging
        loss_per_token = loss_per_token.view(labels.size(0), labels.size(1))
        
        # Create mask for valid tokens (non-padding, non-ignored)
        mask = (labels != -100).float()
        
        # Compute loss per sample (average over valid tokens)
        # Using clamp(min=1) to avoid division by zero
        loss_per_sample = (loss_per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Apply pair weights
        device = loss_per_sample.device
        weights = torch.tensor(
            [self.pair_weights.get(p, 1.0) for p in pairs],
            device=device,
            dtype=torch.float
        )
        
        # Compute balanced loss
        weighted_loss = loss_per_sample * weights
        loss = weighted_loss.sum() / weights.sum()
        
        # Return loss and outputs if requested
        return (loss, outputs) if return_outputs else loss


def create_trainer(
    model,
    args: Seq2SeqTrainingArguments,
    train_dataset,
    eval_dataset,
    tokenizer,
    compute_metrics,
    pair_weights: Optional[Dict[str, float]] = None,
    data_collator=None,
    **kwargs
) -> BalancedLossSeq2SeqTrainer:
    """
    Factory function to create a BalancedLossSeq2SeqTrainer.
    
    Args:
        model: The model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer for processing
        compute_metrics: Function to compute metrics
        pair_weights: Language pair weights for balanced loss
        data_collator: Custom data collator (optional)
        **kwargs: Additional arguments for the trainer
    
    Returns:
        Configured BalancedLossSeq2SeqTrainer
    """
    trainer = BalancedLossSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        pair_weights=pair_weights,
        data_collator=data_collator,
        **kwargs
    )
    
    return trainer
