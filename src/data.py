"""
Dataset loading and preprocessing for the full fine-tuning pipeline.

Each function loads one stage's dataset, applies the confirmed preprocessing,
and returns (train_dataset, val_dataset, test_dataset) as HuggingFace Dataset
objects with torch format set and "labels" column present.

All split logic, text concatenation, and label mapping are sourced directly
from src/finetune.py and src/finetune_2.py. No new behavior is added.

    load_stage1  -- xTRam1/safe-guard-prompt-injection
    load_stage2  -- reshabhs/SPML_Chatbot_Prompt_Injection
    load_stage3  -- nvidia/Aegis-AI-Content-Safety-Dataset-2.0
"""

from datasets import load_dataset

import config


def load_stage1(tokenizer):
    """
    Load and tokenize the Safe-Guard Prompt Injection dataset.

    Behavior sourced from src/finetune.py lines 94-113:
    - Text column: "text"
    - Label column: "label" (integer, 0=safe 1=injection)
    - Validation: 10% split from training set, seed=42
    - Test: native dataset test split

    Args:
        tokenizer: A loaded HuggingFace tokenizer.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) with torch format.
    """
    ds = load_dataset(config.STAGE1_DATASET)

    splits = ds["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = splits["train"]
    val_ds = splits["test"]
    test_ds = ds["test"]

    def preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=config.TOKENIZER_TRUNCATION,
            padding=config.TOKENIZER_PADDING,
            max_length=config.MAX_LENGTH,
        )

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=["text"])
    test_tok = test_ds.map(preprocess, batched=True, remove_columns=["text"])

    # Rename "label" → "labels" for HuggingFace Trainer compatibility.
    # The original finetune.py attempted this in a loop but the assignment was
    # lost (d = d.rename_column(...) only updated the loop variable). Training
    # still worked because the Trainer handles "label" columns. The rename is
    # done correctly here so that set_format can target "labels" explicitly.
    train_tok = train_tok.rename_column("label", "labels") if "label" in train_tok.column_names else train_tok
    val_tok = val_tok.rename_column("label", "labels") if "label" in val_tok.column_names else val_tok
    test_tok = test_tok.rename_column("label", "labels") if "label" in test_tok.column_names else test_tok

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_tok, val_tok, test_tok


def _preprocess_spml_batch(tokenizer, batch):
    """
    Tokenize one SPML batch.

    Text construction sourced from src/finetune.py lines 186-205:
    - If both "System Prompt" and "User Prompt" present: concatenate with " "
    - Else fall back to whichever single column is present
    - Else search for generic text column names
    """
    if "System Prompt" in batch and "User Prompt" in batch:
        combined_text = [
            f"{sys_p} {usr_p}"
            for sys_p, usr_p in zip(batch["System Prompt"], batch["User Prompt"])
        ]
    elif "User Prompt" in batch:
        combined_text = batch["User Prompt"]
    elif "System Prompt" in batch:
        combined_text = batch["System Prompt"]
    else:
        text_col = None
        for possible_col in ["text", "prompt", "input", "sentence", "content"]:
            if possible_col in batch:
                text_col = possible_col
                break
        if text_col is None:
            text_col = [
                col for col in batch.keys()
                if col not in ["label", "Prompt injection"]
            ][0]
        combined_text = batch[text_col]

    return tokenizer(
        combined_text,
        truncation=config.TOKENIZER_TRUNCATION,
        padding=config.TOKENIZER_PADDING,
        max_length=config.MAX_LENGTH,
    )


def load_stage2(tokenizer):
    """
    Load and tokenize the SPML Chatbot Prompt Injection dataset.

    Behavior sourced from src/finetune.py lines 167-226:
    - Text: "System Prompt" + " " + "User Prompt" (concatenated; see _preprocess_spml_batch)
    - Label column: "Prompt injection" (integer), renamed to "labels"
    - Validation: native split if present; else 10% from training set, seed=42
    - Test: native split if present; else reuse validation split

    Args:
        tokenizer: A loaded HuggingFace tokenizer.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) with torch format.
    """
    ds = load_dataset(config.STAGE2_DATASET)

    if "validation" in ds:
        train_ds = ds["train"]
        val_ds = ds["validation"]
        test_ds = ds["test"] if "test" in ds else val_ds
    else:
        splits = ds["train"].train_test_split(test_size=0.1, seed=42)
        train_ds = splits["train"]
        val_ds = splits["test"]
        test_ds = ds["test"] if "test" in ds else val_ds

    print(f"Dataset 2 columns: {train_ds.column_names}")

    # Rename label column before tokenization so it survives remove_columns
    if "Prompt injection" in train_ds.column_names:
        train_ds = train_ds.rename_column("Prompt injection", "label")
        val_ds = val_ds.rename_column("Prompt injection", "label")
        test_ds = test_ds.rename_column("Prompt injection", "label")

    def preprocess(batch):
        return _preprocess_spml_batch(tokenizer, batch)

    # Keep only "label"; remove all other non-tokenizer columns
    non_label_train = [c for c in train_ds.column_names if c != "label"]
    non_label_val = [c for c in val_ds.column_names if c != "label"]
    non_label_test = [c for c in test_ds.column_names if c != "label"]

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=non_label_train)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=non_label_val)
    test_tok = test_ds.map(preprocess, batched=True, remove_columns=non_label_test)

    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")
    test_tok = test_tok.rename_column("label", "labels")

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_tok, val_tok, test_tok


def load_stage3(tokenizer):
    """
    Load and tokenize the NVIDIA Aegis AI Content Safety 2.0 dataset.

    Behavior sourced from src/finetune_2.py lines 94-131:
    - Text column: "prompt"
    - Label column: "prompt_label" (string) — "unsafe" maps to 1, all other values to 0
    - Splits: native train / validation / test (all three confirmed present)

    Note: the label mapping here ("unsafe" → 1, else → 0) is the training-time
    mapping from finetune_2.py line 111. It differs from the broader keyword
    matching in test_model.py, which is evaluation-only logic.

    Args:
        tokenizer: A loaded HuggingFace tokenizer.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) with torch format.
    """
    ds = load_dataset(config.STAGE3_DATASET)

    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    cols_train = train_ds.column_names
    cols_val = val_ds.column_names
    cols_test = test_ds.column_names

    def preprocess(batch):
        tokenized = tokenizer(
            batch["prompt"],
            truncation=config.TOKENIZER_TRUNCATION,
            padding=config.TOKENIZER_PADDING,
            max_length=config.MAX_LENGTH,
        )
        tokenized["labels"] = [
            1 if label == "unsafe" else 0
            for label in batch["prompt_label"]
        ]
        return tokenized

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=cols_train)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=cols_val)
    test_tok = test_ds.map(preprocess, batched=True, remove_columns=cols_test)

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_tok, val_tok, test_tok
