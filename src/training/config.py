"""
Training configuration for multilingual NMT.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TrainingConfig:
    """
    Complete training configuration for multilingual NMT.
    """
    # Model configuration
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"
    
    # Data configuration
    lang_configs: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        ("en-vi", "en", "vi"),
        ("en-fr", "en", "fr"),
        ("de-en", "de", "en"),
    ])
    max_samples_per_pair: int = 50000
    max_length: int = 128
    
    # Training hyperparameters
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Optimizer settings
    optim: str = "adamw_torch"
    
    # Logging and saving
    output_dir: str = "./mbart50-multilingual"
    logging_steps: int = 500
    save_total_limit: int = 1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    gradient_accumulation_steps: int = 1

    # Generation settings for evaluation
    predict_with_generate: bool = True
    generation_max_length: int = 128
    
    # Miscellaneous
    seed: int = 42
    remove_unused_columns: bool = False
    report_to: str = "none"  # Set to "wandb" for wandb logging
    
    # Device settings
    fp16: bool = True  # Use mixed precision training
    dataloader_num_workers: int = 2
    
    def to_dict(self):
        """Convert to dictionary for Seq2SeqTrainingArguments."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class EvaluationConfig:
    """
    Configuration for model evaluation.
    """
    # Metrics to compute
    use_bleu: bool = True
    use_comet: bool = False  # Requires comet installed
    use_chrf: bool = True
    
    # Sampling strategies for inference
    use_greedy: bool = True
    use_beam_search: bool = True
    use_top_k: bool = True
    use_top_p: bool = True
    
    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    num_beams: int = 4
    
    # Output settings
    output_predictions: bool = True
    save_translations: bool = True


@dataclass  
class InferenceConfig:
    """
    Configuration for inference/prediction.
    """
    # Model settings
    model_path: str = "./mbart50-multilingual"
    checkpoint_path: Optional[str] = None
    
    # Generation parameters
    max_length: int = 128
    min_length: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    num_beams: int = 4
    do_sample: bool = False  # Greedy decoding if False
    
    # Device settings
    device: str = "cuda"  # or "cpu"
    
    # Batch settings
    batch_size: int = 16
