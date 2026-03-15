# # Part 1:

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
import matplotlib.pyplot as plt
import os

# 0) Metric
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)


# Function to plot training metrics
def plot_training_metrics(trainer, dataset_name, stage_num, output_dir):
    """
    Plot training and evaluation loss/accuracy for a training stage.
    """
    log_history = trainer.state.log_history

    # Extract metrics
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

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Stage {stage_num}: {dataset_name}", fontsize=16, fontweight="bold")

    # Plot 1: Training Loss
    if train_steps and train_loss:
        axes[0].plot(train_steps, train_loss, "b-", linewidth=2, label="Training Loss")
        axes[0].set_xlabel("Steps", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Training Loss", fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # Plot 2: Evaluation Loss
    if eval_steps and eval_loss:
        axes[1].plot(
            eval_steps, eval_loss, "r-", linewidth=2, marker="o", label="Eval Loss"
        )
        axes[1].set_xlabel("Steps", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].set_title("Evaluation Loss", fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    # Plot 3: Evaluation Accuracy
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

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"stage{stage_num}_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved training metrics plot to: {plot_path}")
    plt.close()


# 1) Load tokenizer + model from finetune.py output (already trained on 2 datasets)
tokenizer = AutoTokenizer.from_pretrained("deberta-pi-full-final")
model = AutoModelForSequenceClassification.from_pretrained(
    "deberta-pi-full-final", num_labels=2
)

# 2) Load nvidia/Aegis-AI-Content-Safety-Dataset-2.0
ds1 = load_dataset(
    "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
)  # continuing training on this dataset

# 2a) Check the dataset structure first
print(f"Dataset splits: {ds1.keys()}")
print(
    f"Dataset columns: {ds1['train'].column_names if 'train' in ds1 else list(ds1.values())[0].column_names}"
)

# Use the existing splits from the Aegis dataset
train_ds1 = ds1["train"]
val_ds1 = ds1["validation"]
test_ds1 = ds1["test"]


# 3) Tokenize - Aegis dataset has 'prompt' column and 'prompt_label' for labels
def preprocess(batch):
    # Aegis dataset has 'prompt' column for text and 'prompt_label' for labels
    # prompt_label: "safe" = 0, "unsafe" = 1
    tokenized = tokenizer(
        batch["prompt"], truncation=True, padding="max_length", max_length=256
    )
    # Convert string labels to integers
    tokenized["labels"] = [
        1 if label == "unsafe" else 0 for label in batch["prompt_label"]
    ]
    return tokenized


# Get all column names to remove (everything including prompt_label since we're adding labels in preprocess)
cols_to_remove_train = train_ds1.column_names
cols_to_remove_val = val_ds1.column_names
cols_to_remove_test = test_ds1.column_names

train_tok1 = train_ds1.map(
    preprocess, batched=True, remove_columns=cols_to_remove_train
)
val_tok1 = val_ds1.map(preprocess, batched=True, remove_columns=cols_to_remove_val)
test_tok1 = test_ds1.map(preprocess, batched=True, remove_columns=cols_to_remove_test)

# Debug: Check what columns are available and verify labels are numeric
print(f"train_tok1 columns: {train_tok1.column_names}")
print(f"Sample item: {train_tok1[0]}")
print(
    f"Label type: {type(train_tok1[0]['labels'])}, Label value: {train_tok1[0]['labels']}"
)

# Set format for PyTorch - similar to finetune.py
train_tok1.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_tok1.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_tok1.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# 4) Continue training on Aegis dataset (Stage 3) - Full fine-tuning
print("\n" + "=" * 60)
print("STAGE 3: Continue training on Aegis-AI-Content-Safety dataset")
print("=" * 60 + "\n")

args1 = TrainingArguments(
    output_dir="deberta-pi-full-stage3",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=25,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",  # or evaluation_strategy if your version supports it
    save_strategy="epoch",
    load_best_model_at_end=True,  # automatically reload the best checkpoint
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=1,  # keep only the best model
    logging_steps=50,
    report_to="none",
)

trainer1 = Trainer(
    model=model,
    args=args1,
    train_dataset=train_tok1,
    eval_dataset=val_tok1,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )  # stop if no improvement for 3 evals
    ],
)

trainer1.train()

# Plot Stage 3 metrics
plot_training_metrics(
    trainer1, "Aegis-AI-Content-Safety-Dataset-2.0", 3, "training_plots"
)

# Evaluate on the held-out test set for stage 3
test_metrics1 = trainer1.evaluate(test_tok1)
print("\nStage 3 Test metrics:", test_metrics1)

# Save stage 3 model as final model
model.save_pretrained("deberta-pi-full-stage3-final")
tokenizer.save_pretrained("deberta-pi-full-stage3-final")
print("\nStage 3 model saved to: deberta-pi-full-stage3-final")

best_model_path = trainer1.state.best_model_checkpoint
print("Best model checkpoint from stage 3:", best_model_path)

print("\n" + "=" * 60)
print("Training complete!")
print(f"Stage 3 (Aegis) test metrics: {test_metrics1}")
print("\nTraining metric plots saved in: training_plots/")
print("=" * 60)
