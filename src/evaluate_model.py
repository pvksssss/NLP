#!/usr/bin/env python3
"""
Evaluate a translation model on OPUS-100 test/validation split.
Run from project root: python -m src.evaluate_model

Examples:
  python -m src.evaluate_model --model_path outputs/best_model --split test
  python -m src.evaluate_model --model_path outputs/best_model --split test --num_beams 1
  python -m src.evaluate_model --model_path facebook/mbart-large-50-many-to-many-mmt --split test --test_size 200

Outputs:
  - JSON metrics report under outputs/eval/
  - CSV with (pair, src, ref, pred) under outputs/eval/
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.loader import load_multilingual_dataset
from src.evaluation.evaluator import NMTEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an NMT model with BLEU/chrF++")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/best_model",
        help="Local path or Hugging Face model id (default: outputs/best_model)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "test"],
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_beams", type=int, default=4, help="num_beams for generation")
    parser.add_argument("--max_length", type=int, default=128, help="Max generation length")
    parser.add_argument("--max_samples_per_pair", type=int, default=50000, help="Train cap (for loading only)")
    parser.add_argument("--val_size", type=int, default=1000, help="Validation size per pair")
    parser.add_argument("--test_size", type=int, default=2000, help="Test size per pair")
    parser.add_argument(
        "--variance",
        action="store_true",
        help="Compute BLEU bootstrap variance + confidence interval",
    )
    parser.add_argument(
        "--bleurt",
        action="store_true",
        help="Include BLEURT (slow; requires extra deps)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval",
        help="Where to write JSON/CSV outputs (default: outputs/eval)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NMT Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"num_beams: {args.num_beams}")

    print("\n[1/3] Loading dataset...")
    dataset = load_multilingual_dataset(
        max_samples_per_pair=args.max_samples_per_pair,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    eval_ds = dataset[args.split]
    print(f"Eval samples: {len(eval_ds)}")

    print("\n[2/3] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    print("\n[3/3] Running evaluation...")
    evaluator = NMTEvaluator(model=model, tokenizer=tokenizer, max_length=args.max_length)
    results = evaluator.evaluate(
        eval_ds,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        compute_variance=args.variance,
        include_bleurt=args.bleurt,
        show_progress=True,
    )

    print(results.summary())

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    safe_model_name = (
        Path(args.model_path).name if Path(args.model_path).exists() else args.model_path.replace("/", "_")
    )

    json_path = output_dir / f"{ts}_metrics_{safe_model_name}_{args.split}.json"
    results.save(str(json_path))

    csv_path = output_dir / f"{ts}_preds_{safe_model_name}_{args.split}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair", "src", "ref", "pred"])
        writer.writeheader()
        for pair, src, ref, pred in zip(results.pairs, eval_ds["src"], eval_ds["tgt"], results.predictions):
            writer.writerow({"pair": pair, "src": src, "ref": ref, "pred": pred})

    print(f"\nSaved metrics: {json_path}")
    print(f"Saved predictions: {csv_path}")


if __name__ == "__main__":
    main()
