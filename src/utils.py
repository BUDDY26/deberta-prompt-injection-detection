"""
Shared utilities for the training pipelines.

Functions extracted from src/finetune.py and src/finetune_2.py, where they
existed as identical copies. Centralizing them here eliminates the duplication
confirmed in docs/evidence-ledger.md §7.

    compute_metrics        -- accuracy metric callback for HuggingFace Trainer
    plot_training_metrics  -- save loss/accuracy plots after each training stage
    set_global_seed        -- set random seeds across Python, NumPy, and PyTorch
    write_run_config       -- write run_config.json into a training output directory
"""

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_metrics(eval_pred):
    """
    Compute accuracy from Trainer eval_pred.

    Sourced from src/finetune.py lines 19-22 and src/finetune_2.py lines 19-22
    (identical implementations).

    NumPy-based accuracy avoids a sys.path collision between the HuggingFace
    evaluate package and src/evaluate.py when pytest prepends src/ to sys.path
    via conftest.py.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": float((preds == labels).mean())}


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
        axes[1].plot(
            eval_steps, eval_loss, "r-", linewidth=2, marker="o", label="Eval Loss"
        )
        axes[1].set_xlabel("Steps", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].set_title("Evaluation Loss", fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    if eval_steps and eval_accuracy:
        axes[2].plot(
            eval_steps,
            eval_accuracy,
            "g-",
            linewidth=2,
            marker="s",
            label="Eval Accuracy",
        )
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


def write_run_config(output_dir: str, config_snapshot: dict) -> None:
    """
    Write a run_config.json file into output_dir capturing run metadata.

    Always-included fields (added by this function):
        timestamp   -- ISO 8601 UTC timestamp of the run
        git_commit  -- short commit hash from git HEAD (empty string if unavailable)
        python      -- Python version string
        torch       -- PyTorch version string

    Caller-provided fields (via config_snapshot):
        Any key-value pairs relevant to the run, e.g. seed, hyperparameters.

    Args:
        output_dir:      Directory where run_config.json will be written.
                         Created if it does not exist.
        config_snapshot: Dict of caller-supplied fields to include in the record.
    """
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = ""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "python": sys.version,
        "torch": torch.__version__,
    }
    record.update(config_snapshot)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Run config written to: {path}")
