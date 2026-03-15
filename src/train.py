"""
Full fine-tuning entry point for the DeBERTa prompt injection detection pipeline.

Three-stage sequential fine-tuning (ADR-003):
  Stage 1 — xTRam1/safe-guard-prompt-injection
  Stage 2 — reshabhs/SPML_Chatbot_Prompt_Injection   (continues in-memory from stage 1)
  Stage 3 — nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (reloads from deberta-pi-full-final)

Run from the repository root:
    python src/train.py

Confirmed results from this pipeline (docs/evidence-ledger.md §8):
  Stage 2 model on Aegis test set: 41.60% accuracy, F1 0.3053
  Stage 3 model on Aegis test set: 81.16% accuracy, F1 0.8255

For the LoRA pipeline, see src/train_lora.py (Phase 4).
"""

import os
import sys

# Ensure src/ is on the path so that config, data, utils are importable
# when this script is run as: python src/train.py
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import config
from data import load_stage1, load_stage2, load_stage3
from utils import (
    compute_metrics,
    plot_training_metrics,
    set_global_seed,
    write_run_config,
)


def train_stage1(model, tokenizer):
    """Train on Safe-Guard dataset. Returns (model, test_metrics)."""
    print("\n" + "=" * 60)
    print("STAGE 1: Training on safe-guard-prompt-injection dataset")
    print("=" * 60 + "\n")

    train_tok, val_tok, test_tok = load_stage1(tokenizer)

    args = TrainingArguments(
        output_dir=config.STAGE1_OUTPUT_DIR,
        per_device_train_batch_size=config.STAGE1_TRAIN_BATCH,
        per_device_eval_batch_size=config.STAGE1_EVAL_BATCH,
        learning_rate=config.STAGE1_LR,
        num_train_epochs=config.STAGE1_EPOCHS,
        fp16=torch.cuda.is_available(),
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        report_to=config.REPORT_TO,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.STAGE1_PATIENCE)
        ],
    )

    trainer.train()
    plot_training_metrics(trainer, "safe-guard-prompt-injection", 1, config.PLOTS_DIR)

    test_metrics = trainer.evaluate(test_tok)
    print("\nStage 1 Test metrics:", test_metrics)

    model.save_pretrained(config.STAGE1_FINAL_DIR)
    tokenizer.save_pretrained(config.STAGE1_FINAL_DIR)
    print(f"\nStage 1 model saved to: {config.STAGE1_FINAL_DIR}")

    return model, test_metrics


def train_stage2(model, tokenizer):
    """
    Continue training on SPML dataset using the in-memory model from stage 1.
    Returns test_metrics.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Training on SPML_Chatbot_Prompt_Injection dataset")
    print("=" * 60 + "\n")

    train_tok, val_tok, test_tok = load_stage2(tokenizer)

    args = TrainingArguments(
        output_dir=config.STAGE2_OUTPUT_DIR,
        per_device_train_batch_size=config.STAGE2_TRAIN_BATCH,
        per_device_eval_batch_size=config.STAGE2_EVAL_BATCH,
        learning_rate=config.STAGE2_LR,
        num_train_epochs=config.STAGE2_EPOCHS,
        fp16=torch.cuda.is_available(),
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        report_to=config.REPORT_TO,
    )

    trainer = Trainer(
        model=model,  # Continue training with the same model from stage 1
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.STAGE2_PATIENCE)
        ],
    )

    trainer.train()
    plot_training_metrics(trainer, "SPML_Chatbot_Prompt_Injection", 2, config.PLOTS_DIR)

    test_metrics = trainer.evaluate(test_tok)
    print("\nStage 2 Test metrics:", test_metrics)

    # Save the combined stages 1+2 model; stage 3 reloads from this path
    print("\n" + "=" * 60)
    print("Saving final model after both training stages")
    print("=" * 60 + "\n")

    model.save_pretrained(config.STAGE2_FINAL_DIR)
    tokenizer.save_pretrained(config.STAGE2_FINAL_DIR)
    print(f"Final model saved to: {config.STAGE2_FINAL_DIR}")
    print("Best model checkpoint from stage 2:", trainer.state.best_model_checkpoint)

    return test_metrics


def train_stage3():
    """
    Load the stage 2 output and continue training on the Aegis dataset.

    Stage 3 reloads from disk (config.STAGE2_FINAL_DIR) rather than continuing
    in-memory. This matches src/finetune_2.py lines 88-91.

    Returns test_metrics.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Continue training on Aegis-AI-Content-Safety dataset")
    print("=" * 60 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(config.STAGE2_FINAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.STAGE2_FINAL_DIR, num_labels=2
    )

    train_tok, val_tok, test_tok = load_stage3(tokenizer)

    print(f"Dataset splits: {config.STAGE3_DATASET}")
    print(f"  train={len(train_tok)}, val={len(val_tok)}, test={len(test_tok)}")

    args = TrainingArguments(
        output_dir=config.STAGE3_OUTPUT_DIR,
        per_device_train_batch_size=config.STAGE3_TRAIN_BATCH,
        per_device_eval_batch_size=config.STAGE3_EVAL_BATCH,
        learning_rate=config.STAGE3_LR,
        num_train_epochs=config.STAGE3_EPOCHS,
        fp16=torch.cuda.is_available(),
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        report_to=config.REPORT_TO,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.STAGE3_PATIENCE)
        ],
    )

    trainer.train()
    plot_training_metrics(
        trainer, "Aegis-AI-Content-Safety-Dataset-2.0", 3, config.PLOTS_DIR
    )

    test_metrics = trainer.evaluate(test_tok)
    print("\nStage 3 Test metrics:", test_metrics)

    model.save_pretrained(config.STAGE3_FINAL_DIR)
    tokenizer.save_pretrained(config.STAGE3_FINAL_DIR)
    print(f"\nStage 3 model saved to: {config.STAGE3_FINAL_DIR}")
    print("Best model checkpoint from stage 3:", trainer.state.best_model_checkpoint)

    return test_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Full fine-tuning pipeline for DeBERTa prompt injection detection."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.GLOBAL_SEED,
        help=f"Random seed (default: {config.GLOBAL_SEED})",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    write_run_config(
        config.PLOTS_DIR,
        {
            "pipeline": "full_ft",
            "seed": args.seed,
            "base_model": config.BASE_MODEL,
            "stage1": {
                "dataset": config.STAGE1_DATASET,
                "lr": config.STAGE1_LR,
                "train_batch": config.STAGE1_TRAIN_BATCH,
                "epochs": config.STAGE1_EPOCHS,
                "patience": config.STAGE1_PATIENCE,
            },
            "stage2": {
                "dataset": config.STAGE2_DATASET,
                "lr": config.STAGE2_LR,
                "train_batch": config.STAGE2_TRAIN_BATCH,
                "epochs": config.STAGE2_EPOCHS,
                "patience": config.STAGE2_PATIENCE,
            },
            "stage3": {
                "dataset": config.STAGE3_DATASET,
                "lr": config.STAGE3_LR,
                "train_batch": config.STAGE3_TRAIN_BATCH,
                "epochs": config.STAGE3_EPOCHS,
                "patience": config.STAGE3_PATIENCE,
            },
        },
    )

    print(f"Loading base model: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL, num_labels=2
    )

    # Stages 1 and 2 share the in-memory model object
    model, test_metrics1 = train_stage1(model, tokenizer)
    test_metrics2 = train_stage2(model, tokenizer)

    # Stage 3 reloads the model from deberta-pi-full-final
    test_metrics3 = train_stage3()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Stage 1 (safe-guard) test metrics: {test_metrics1}")
    print(f"Stage 2 (SPML) test metrics:       {test_metrics2}")
    print(f"Stage 3 (Aegis) test metrics:      {test_metrics3}")
    print(f"\nTraining metric plots saved in: {config.PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
