"""
Shared utilities for the training pipelines.

Functions extracted from src/finetune.py and src/finetune_2.py, where they
existed as identical copies. Centralizing them here eliminates the duplication
confirmed in docs/evidence-ledger.md §7.

    compute_metrics        -- accuracy metric callback for HuggingFace Trainer
    plot_training_metrics  -- save loss/accuracy plots after each training stage
    set_global_seed        -- set random seeds across Python, NumPy, and PyTorch
"""

import os
import random

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch

_accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """
    Compute accuracy from Trainer eval_pred.

    Sourced from src/finetune.py lines 19-22 and src/finetune_2.py lines 19-22
    (identical implementations).
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return _accuracy_metric.compute(predictions=preds, references=labels)


def plot_training_metrics(trainer, dataset_name, stage_num, output_dir):
    """
    Plot training loss, evaluation loss, and evaluation accuracy for one stage.

    Saves a three-panel PNG to output_dir/stage{stage_num}_metrics.png.

    Sourced from src/finetune.py lines 25-85 and src/finetune_2.py lines 25-85
    (identical implementations).

    Args:
        trainer:       A completed HuggingFace Trainer instance.
        dataset_name:  Human-readable dataset name for the plot title.
        stage_num:     Integer stage number (1, 2, or 3).
        output_dir:    Directory path where the PNG is saved.
    """
    log_history = trainer.state.log_history

    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    eval_accuracy = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_steps.append(entry.get("step", 0))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_loss.append(entry["eval_loss"])
            eval_accuracy.append(entry.get("eval_accuracy", 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Stage {stage_num}: {dataset_name}", fontsize=16, fontweight="bold")

    if train_steps and train_loss:
        axes[0].plot(train_steps, train_loss, "b-", linewidth=2, label="Training Loss")
        axes[0].set_xlabel("Steps", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Training Loss", fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    if eval_steps and eval_loss:
        axes[1].plot(eval_steps, eval_loss, "r-", linewidth=2, marker="o", label="Eval Loss")
        axes[1].set_xlabel("Steps", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].set_title("Evaluation Loss", fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    if eval_steps and eval_accuracy:
        axes[2].plot(eval_steps, eval_accuracy, "g-", linewidth=2, marker="s", label="Eval Accuracy")
        axes[2].set_xlabel("Steps", fontsize=12)
        axes[2].set_ylabel("Accuracy", fontsize=12)
        axes[2].set_title("Evaluation Accuracy", fontsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"stage{stage_num}_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved training metrics plot to: {plot_path}")
    plt.close()


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    The original training scripts set seed=42 only for dataset splits
    (train_test_split calls). This function extends that to cover all
    sources of randomness, improving reproducibility of the full pipeline.

    Args:
        seed: Integer seed value. Default matches the split seed used in
              src/finetune.py lines 98 and 175.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
