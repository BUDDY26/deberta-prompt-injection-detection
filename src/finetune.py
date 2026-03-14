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
        if 'loss' in entry and 'epoch' in entry:
            train_steps.append(entry.get('step', 0))
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry.get('step', 0))
            eval_loss.append(entry['eval_loss'])
            eval_accuracy.append(entry.get('eval_accuracy', 0))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Stage {stage_num}: {dataset_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    if train_steps and train_loss:
        axes[0].plot(train_steps, train_loss, 'b-', linewidth=2, label='Training Loss')
        axes[0].set_xlabel('Steps', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Plot 2: Evaluation Loss
    if eval_steps and eval_loss:
        axes[1].plot(eval_steps, eval_loss, 'r-', linewidth=2, marker='o', label='Eval Loss')
        axes[1].set_xlabel('Steps', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Evaluation Loss', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # Plot 3: Evaluation Accuracy
    if eval_steps and eval_accuracy:
        axes[2].plot(eval_steps, eval_accuracy, 'g-', linewidth=2, marker='s', label='Eval Accuracy')
        axes[2].set_xlabel('Steps', fontsize=12)
        axes[2].set_ylabel('Accuracy', fontsize=12)
        axes[2].set_title('Evaluation Accuracy', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'stage{stage_num}_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved training metrics plot to: {plot_path}")
    plt.close()

# 1) Load tokenizer + base model (binary classifier: 0 = safe, 1 = injection)
tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection")
model = AutoModelForSequenceClassification.from_pretrained(
    "ProtectAI/deberta-v3-base-prompt-injection", num_labels=2
)

# 2) Load first dataset
ds1 = load_dataset("xTRam1/safe-guard-prompt-injection")  # columns: "text", "label"

# 2a) Create validation split from the training set (e.g., 10%)
# If the dataset already has 'validation', you can skip this split and use ds1["validation"].
splits1 = ds1["train"].train_test_split(test_size=0.1, seed=42)
train_ds1 = splits1["train"]
val_ds1 = splits1["test"]
test_ds1 = ds1["test"]

# 3) Tokenize
def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_tok1 = train_ds1.map(preprocess, batched=True, remove_columns=["text"])
val_tok1   = val_ds1.map(preprocess,   batched=True, remove_columns=["text"])
test_tok1  = test_ds1.map(preprocess,  batched=True, remove_columns=["text"])

for d in (train_tok1, val_tok1, test_tok1):
    d = d.rename_column("label", "labels") if "label" in d.column_names else d
    d.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4) Train on first dataset (safe-guard-prompt-injection) - Full fine-tuning
print("\n" + "="*60)
print("STAGE 1: Training on safe-guard-prompt-injection dataset")
print("="*60 + "\n")

args1 = TrainingArguments(
    output_dir="deberta-pi-full-stage1",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=10, #5
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",          # or evaluation_strategy if your version supports it
    save_strategy="epoch",
    load_best_model_at_end=True,    # automatically reload the best checkpoint
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=1,             # keep only the best model
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
        EarlyStoppingCallback(early_stopping_patience=3)  # stop if no improvement for 3 evals
    ],
)

trainer1.train()

# Plot Stage 1 metrics
plot_training_metrics(trainer1, "safe-guard-prompt-injection", 1, "training_plots")

# Evaluate on the held-out test set for stage 1
test_metrics1 = trainer1.evaluate(test_tok1)
print("\nStage 1 Test metrics:", test_metrics1)

# Save stage 1 model
model.save_pretrained("deberta-pi-full-stage1-final")
tokenizer.save_pretrained("deberta-pi-full-stage1-final")
print("\nStage 1 model saved to: deberta-pi-full-stage1-final")

# 6) Load and prepare second dataset (SPML_Chatbot_Prompt_Injection)
print("\n" + "="*60)
print("STAGE 2: Training on SPML_Chatbot_Prompt_Injection dataset")
print("="*60 + "\n")

ds2 = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")

# Prepare second dataset - create validation split if needed
if "validation" in ds2:
    train_ds2 = ds2["train"]
    val_ds2 = ds2["validation"]
    test_ds2 = ds2["test"] if "test" in ds2 else val_ds2
else:
    splits2 = ds2["train"].train_test_split(test_size=0.1, seed=42)
    train_ds2 = splits2["train"]
    val_ds2 = splits2["test"]
    test_ds2 = ds2["test"] if "test" in ds2 else val_ds2

# Check dataset columns and create appropriate preprocessing function
print(f"Dataset 2 columns: {train_ds2.column_names}")

# Determine text and label columns for second dataset
# SPML dataset has: 'System Prompt', 'User Prompt', 'Prompt injection' (label), 'Degree', 'Source'
# We'll combine System Prompt and User Prompt as the text input
def preprocess_ds2(batch):
    # Combine system and user prompts
    if 'System Prompt' in batch and 'User Prompt' in batch:
        combined_text = [f"{sys} {usr}" for sys, usr in zip(batch['System Prompt'], batch['User Prompt'])]
    elif 'User Prompt' in batch:
        combined_text = batch['User Prompt']
    elif 'System Prompt' in batch:
        combined_text = batch['System Prompt']
    else:
        # Fallback to any text-like column
        text_col = None
        for possible_col in ['text', 'prompt', 'input', 'sentence', 'content']:
            if possible_col in batch:
                text_col = possible_col
                break
        if text_col is None:
            text_col = [col for col in batch.keys() if col not in ['label', 'Prompt injection']][0]
        combined_text = batch[text_col]

    return tokenizer(combined_text, truncation=True, padding="max_length", max_length=256)

# First rename the label column, then tokenize
# 'Prompt injection' column contains the labels
if 'Prompt injection' in train_ds2.column_names:
    train_ds2 = train_ds2.rename_column("Prompt injection", "label")
    val_ds2 = val_ds2.rename_column("Prompt injection", "label")
    test_ds2 = test_ds2.rename_column("Prompt injection", "label")

# Tokenize second dataset - keep the label column
train_tok2 = train_ds2.map(preprocess_ds2, batched=True, remove_columns=[col for col in train_ds2.column_names if col != "label"])
val_tok2   = val_ds2.map(preprocess_ds2,   batched=True, remove_columns=[col for col in val_ds2.column_names if col != "label"])
test_tok2  = test_ds2.map(preprocess_ds2,  batched=True, remove_columns=[col for col in test_ds2.column_names if col != "label"])

# Rename label to labels and set format
train_tok2 = train_tok2.rename_column("label", "labels")
val_tok2 = val_tok2.rename_column("label", "labels")
test_tok2 = test_tok2.rename_column("label", "labels")

train_tok2.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_tok2.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_tok2.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 7) Continue training on second dataset
args2 = TrainingArguments(
    output_dir="deberta-pi-full-stage2",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=15, #5
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
)

trainer2 = Trainer(
    model=model,  # Continue training with the same model from stage 1
    args=args2,
    train_dataset=train_tok2,
    eval_dataset=val_tok2,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3)
    ],
)

trainer2.train()

# Plot Stage 2 metrics
plot_training_metrics(trainer2, "SPML_Chatbot_Prompt_Injection", 2, "training_plots")

# Evaluate on the held-out test set for stage 2
test_metrics2 = trainer2.evaluate(test_tok2)
print("\nStage 2 Test metrics:", test_metrics2)

# 8) Save final model after both stages
print("\n" + "="*60)
print("Saving final model after both training stages")
print("="*60 + "\n")

# Save final model
model.save_pretrained("deberta-pi-full-final")
tokenizer.save_pretrained("deberta-pi-full-final")
print("Final model saved to: deberta-pi-full-final")

best_model_path = trainer2.state.best_model_checkpoint
print("Best model checkpoint from stage 2:", best_model_path)

print("\n" + "="*60)
print("Training complete!")
print(f"Stage 1 (safe-guard) test metrics: {test_metrics1}")
print(f"Stage 2 (SPML) test metrics: {test_metrics2}")
print("\nTraining metric plots saved in: training_plots/")
print("="*60)