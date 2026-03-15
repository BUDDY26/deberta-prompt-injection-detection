"""
LoRA fine-tuning entry point for the DeBERTa prompt injection detection pipeline.

Two-stage sequential LoRA training (ADR-002, ADR-006):
  Stage 1 — xTRam1/safe-guard-prompt-injection
  Stage 2 — reshabhs/SPML_Chatbot_Prompt_Injection   (continues in-memory from stage 1)

Stage 3 (Aegis) LoRA was never written in the original notebook and is explicitly
out of scope for this pipeline. See ADR-002 (D2 resolution) and ADR-006.

Run from the repository root:
    python src/train_lora.py

Source: extracted from notebooks/Module2Project (1).ipynb cells 9–10 (ADR-006).
The notebook is retained at its original path as a supplementary reference.

Correction applied on extraction (ADR-006):
    num_train_epochs is set to 10 for both stages.
    The notebook currently shows 20, but trainer_state.json artifacts and the
    notebook's own execution output confirm the original training used 10 epochs.

Within-distribution results from these artifacts (models/deberta-pi-lora-*):
  Stage 1 best val accuracy: 98.67% at epoch 2
  Stage 2 best val accuracy: 96.07% at epoch 8

For the full fine-tuning pipeline (including the Aegis cross-dataset result of
81.16% accuracy / F1 0.8255), see src/train.py (Phase 3).
"""

import os
import sys

# Ensure src/ is on the path so that config, data, utils are importable
# when this script is run as: python src/train_lora.py
sys.path.insert(0, os.path.dirname(__file__))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import config
from data import load_stage1, load_stage2
from utils import (
    compute_metrics,
    plot_training_metrics,
    set_global_seed,
    write_run_config,
)

# ---------------------------------------------------------------------------
# LoRA-specific constants
# Source: docs/evidence-ledger.md §3 (trainer_state.json + notebook cells 9–10)
#         and §7 (adapter_config.json — all four artifacts agree)
# ---------------------------------------------------------------------------

# Adapter hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "o_proj"]
LORA_MODULES_TO_SAVE = ["classifier", "score"]
LORA_BIAS = "none"

# Training hyperparameters
LORA_TRAIN_BATCH = 16
LORA_EVAL_BATCH = 32
LORA_LR = 2e-4  # differs from full FT (2e-5); higher LR is standard for LoRA
LORA_STAGE1_EPOCHS = (
    10  # confirmed by trainer_state.json (notebook shows 20 — see ADR-006)
)
LORA_STAGE2_EPOCHS = 10  # confirmed by trainer_state.json
LORA_PATIENCE = 3

# Output directory names — match models/ directory names exactly
LORA_STAGE1_OUTPUT_DIR = "deberta-pi-lora-stage1"
LORA_STAGE1_FINAL_DIR = "deberta-pi-lora-stage1-final"
LORA_STAGE2_OUTPUT_DIR = "deberta-pi-lora-stage2"
LORA_FINAL_ADAPTER_DIR = "deberta-pi-lora-final-adapter"  # model.save_pretrained
LORA_FINAL_FULL_DIR = "deberta-pi-lora-final-full"  # trainer.save_model


def _make_lora_config():
    """
    Build the LoraConfig from confirmed artifact values.

    All fields sourced from adapter_config.json files in models/deberta-pi-lora-*
    (evidence-ledger §7). All four adapter configs agree on every value.
    """
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=LORA_MODULES_TO_SAVE,
        bias=LORA_BIAS,
        task_type=TaskType.SEQ_CLS,
    )


def train_lora_stage1(model, tokenizer):
    """
    Wrap the base model in LoRA adapters and train on the Safe-Guard dataset.

    Returns (peft_model, test_metrics). The returned peft_model is passed
    directly into train_lora_stage2 to continue training in-memory.
    """
    print("\n" + "=" * 60)
    print("LORA STAGE 1: Training on safe-guard-prompt-injection dataset")
    print("=" * 60 + "\n")

    lora_config = _make_lora_config()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    train_tok, val_tok, test_tok = load_stage1(tokenizer)

    args = TrainingArguments(
        output_dir=LORA_STAGE1_OUTPUT_DIR,
        per_device_train_batch_size=LORA_TRAIN_BATCH,
        per_device_eval_batch_size=LORA_EVAL_BATCH,
        learning_rate=LORA_LR,
        num_train_epochs=LORA_STAGE1_EPOCHS,
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
        model=peft_model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=LORA_PATIENCE)],
    )

    trainer.train()
    plot_training_metrics(
        trainer, "safe-guard-prompt-injection (LoRA)", 1, config.PLOTS_DIR
    )

    test_metrics = trainer.evaluate(test_tok)
    print("\nLoRA Stage 1 Test metrics:", test_metrics)

    peft_model.save_pretrained(LORA_STAGE1_FINAL_DIR)
    tokenizer.save_pretrained(LORA_STAGE1_FINAL_DIR)
    print(f"\nLoRA Stage 1 adapter saved to: {LORA_STAGE1_FINAL_DIR}")

    return peft_model, test_metrics


def train_lora_stage2(peft_model, tokenizer):
    """
    Continue LoRA training on the SPML dataset using the in-memory model from stage 1.

    Two saves are performed at the end, matching notebook cell 10 exactly (ADR-006):
      - model.save_pretrained  → LORA_FINAL_ADAPTER_DIR  (adapter weights only)
      - trainer.save_model     → LORA_FINAL_FULL_DIR      (adapter + training_args.bin)

    Returns test_metrics.
    """
    print("\n" + "=" * 60)
    print("LORA STAGE 2: Training on SPML_Chatbot_Prompt_Injection dataset")
    print("=" * 60 + "\n")

    train_tok, val_tok, test_tok = load_stage2(tokenizer)

    args = TrainingArguments(
        output_dir=LORA_STAGE2_OUTPUT_DIR,
        per_device_train_batch_size=LORA_TRAIN_BATCH,
        per_device_eval_batch_size=LORA_EVAL_BATCH,
        learning_rate=LORA_LR,
        num_train_epochs=LORA_STAGE2_EPOCHS,
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
        model=peft_model,  # Continue training with the same PEFT model from stage 1
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=LORA_PATIENCE)],
    )

    trainer.train()
    plot_training_metrics(
        trainer, "SPML_Chatbot_Prompt_Injection (LoRA)", 2, config.PLOTS_DIR
    )

    test_metrics = trainer.evaluate(test_tok)
    print("\nLoRA Stage 2 Test metrics:", test_metrics)

    # Two saves — matches notebook cell 10 save calls exactly (ADR-006, evidence-ledger §7)
    print("\n" + "=" * 60)
    print("Saving final LoRA models after both training stages")
    print("=" * 60 + "\n")

    # Save 1: adapter weights only (model.save_pretrained)
    peft_model.save_pretrained(LORA_FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_FINAL_ADAPTER_DIR)
    print(f"Final adapter saved to: {LORA_FINAL_ADAPTER_DIR}")

    # Save 2: adapter + training_args.bin (trainer.save_model)
    trainer.save_model(LORA_FINAL_FULL_DIR)
    tokenizer.save_pretrained(LORA_FINAL_FULL_DIR)
    print(f"Final full model saved to: {LORA_FINAL_FULL_DIR}")

    print(
        "Best model checkpoint from LoRA stage 2:", trainer.state.best_model_checkpoint
    )

    return test_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning pipeline for DeBERTa prompt injection detection."
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
            "pipeline": "lora",
            "seed": args.seed,
            "base_model": config.BASE_MODEL,
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
                "bias": LORA_BIAS,
            },
            "stage1": {
                "dataset": config.STAGE1_DATASET,
                "lr": LORA_LR,
                "train_batch": LORA_TRAIN_BATCH,
                "epochs": LORA_STAGE1_EPOCHS,
                "patience": LORA_PATIENCE,
            },
            "stage2": {
                "dataset": config.STAGE2_DATASET,
                "lr": LORA_LR,
                "train_batch": LORA_TRAIN_BATCH,
                "epochs": LORA_STAGE2_EPOCHS,
                "patience": LORA_PATIENCE,
            },
        },
    )

    print(f"Loading base model: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL, num_labels=2
    )

    # Stages 1 and 2 share the in-memory PEFT model object
    peft_model, test_metrics1 = train_lora_stage1(base_model, tokenizer)
    test_metrics2 = train_lora_stage2(peft_model, tokenizer)

    print("\n" + "=" * 60)
    print("LoRA training complete!")
    print(f"Stage 1 (safe-guard) test metrics: {test_metrics1}")
    print(f"Stage 2 (SPML) test metrics:       {test_metrics2}")
    print(f"\nTraining metric plots saved in: {config.PLOTS_DIR}/")
    print()
    print("Stage 3 LoRA (Aegis) was not part of the original training run.")
    print("See ADR-002 (D2) and ADR-006 for scope rationale.")
    print("=" * 60)


if __name__ == "__main__":
    main()
