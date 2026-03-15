"""
Post-training evaluation script for the DeBERTa prompt injection detection pipeline.

Parameterized replacement for src/test_model.py (ADR-005, D5 = Option A).

Evaluation scope (D5):
  One model vs. the Aegis test set per invocation.
  Only the Aegis dataset is supported under the approved scope.

Run from the repository root:
    python src/evaluate.py --model-path deberta-pi-full-stage3-final --dataset aegis
    python src/evaluate.py --model-path deberta-pi-full-stage2-final --dataset aegis

Output file:
    results/evaluation/{model_name}_{dataset_slug}_results.txt

Improvements over src/test_model.py:
  - Model path and dataset accepted as CLI arguments (no hardcoded values)
  - Output written to results/evaluation/ with a deterministic filename
    (fixes the path mismatch documented in evidence-ledger §10)
  - Label mapping uses the strict training-time rule: "unsafe" → 1, else → 0
    (matches finetune_2.py line 111; produces identical results on the Aegis dataset
    because its prompt_label field contains only "safe" and "unsafe" values)
  - Random-sample and misclassification debug sections omitted from the saved file
    (they were interactive debugging aids in test_model.py, not part of the
    confirmed result-file format)

Preserved from src/test_model.py (evidence-ledger §5):
  - Batch size: 16
  - Max length: 256 (from config.MAX_LENGTH)
  - Decision rule: torch.argmax(logits, dim=-1)
  - Device: CUDA if available, else CPU
  - Metrics: accuracy_score, precision_recall_fscore_support(average='binary',
    pos_label=1, zero_division=0), confusion_matrix,
    classification_report(target_names=['Safe','Unsafe'], digits=4, zero_division=0)
  - Output format: matches the confirmed result files exactly
    (results/test_results_2dataset.txt, results/test_results_3datasets.txt)
"""

import argparse
import os
import sys

# Ensure src/ is on the path so that config is importable
# when this script is run as: python src/evaluate.py
sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

# ---------------------------------------------------------------------------
# Dataset registry — D5 = Aegis only (ADR-005)
# Extend this table if a future decision expands the evaluation scope.
# ---------------------------------------------------------------------------
_DATASET_REGISTRY = {
    "aegis": config.STAGE3_DATASET,  # "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
}

_INFERENCE_BATCH_SIZE = 16  # evidence-ledger §5 (test_model.py line 16)


def _load_aegis():
    """
    Load the Aegis test split and return (texts, labels).

    Text column: "prompt"
    Label mapping: "unsafe" → 1, all other values → 0
    Sourced from finetune_2.py lines 101–112 and evidence-ledger §4.

    The Aegis dataset has a native "test" split (evidence-ledger §4).
    The split-selection guard is included for safety but is not expected to fire.
    """
    ds = load_dataset(config.STAGE3_DATASET)

    if "test" in ds:
        test_ds = ds["test"]
    elif "validation" in ds:
        test_ds = ds["validation"]
    else:
        test_ds = ds["train"].train_test_split(test_size=0.1, seed=42)["test"]

    print(f"Test set size: {len(test_ds)}")
    print(f"Dataset columns: {test_ds.column_names}\n")

    texts = [example["prompt"] for example in test_ds]
    labels = [1 if example["prompt_label"] == "unsafe" else 0 for example in test_ds]

    print(
        f"Label distribution: Safe (0): {labels.count(0)}, Unsafe (1): {labels.count(1)}\n"
    )
    return texts, labels


def _run_inference(model, tokenizer, texts, device):
    """
    Run batched inference and return a list of integer predictions.

    Decision rule: torch.argmax(logits, dim=-1)
    Batch size: _INFERENCE_BATCH_SIZE (16)
    Sourced from test_model.py lines 165–185 (evidence-ledger §5).
    """
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), _INFERENCE_BATCH_SIZE), desc="Inference"):
            batch_texts = texts[i : i + _INFERENCE_BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                truncation=config.TOKENIZER_TRUNCATION,
                padding=config.TOKENIZER_PADDING,
                max_length=config.MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)
    return predictions


def _format_report(model_name, dataset_id, labels, predictions, device):
    """
    Build the evaluation report string in the confirmed result-file format.

    Format sourced from results/test_results_2dataset.txt and
    results/test_results_3datasets.txt (evidence-ledger §8).
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        target_names=["Safe", "Unsafe"],
        digits=4,
        zero_division=0,
    )

    lines = [
        "=" * 80,
        "Model Evaluation Results",
        "=" * 80,
        "",
        f"Model: {model_name}",
        f"Dataset: {dataset_id}",
        f"Test Set Size: {len(labels)}",
        f"Device: {device}",
        "",
        f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)",
        "",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"F1-Score:  {f1:.4f}",
        "",
        "Confusion Matrix:",
        "                Predicted",
        "              Safe  Unsafe",
        f"Actual Safe   {cm[0][0]:5d}  {cm[0][1]:5d}",
        f"      Unsafe  {cm[1][0]:5d}  {cm[1][1]:5d}",
        "",
        "Detailed Classification Report:",
        report,
    ]
    return "\n".join(lines)


def evaluate(model_path, dataset_slug):
    """
    Evaluate one model against one dataset and write results to disk.

    Args:
        model_path:   Path to a saved HuggingFace model directory.
        dataset_slug: Short name for the dataset. Currently only "aegis" is
                      supported (D5 = Option A, ADR-005).

    Output file:
        results/evaluation/{model_name}_{dataset_slug}_results.txt
    """
    if dataset_slug not in _DATASET_REGISTRY:
        supported = ", ".join(f'"{k}"' for k in _DATASET_REGISTRY)
        raise ValueError(
            f"Unsupported dataset '{dataset_slug}'. "
            f"Supported values under D5 scope: {supported}."
        )

    dataset_id = _DATASET_REGISTRY[dataset_slug]
    model_name = os.path.basename(os.path.normpath(model_path))

    print("=" * 80)
    print(f"Evaluating: {model_name}")
    print(f"Dataset:    {dataset_id}")
    print("=" * 80 + "\n")

    # Load model
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}\n")

    # Load dataset
    print(f"Loading {dataset_id}...")
    texts, labels = _load_aegis()

    # Inference
    print("Running inference...")
    predictions = _run_inference(model, tokenizer, texts, device)

    # Build report
    report_text = _format_report(model_name, dataset_id, labels, predictions, device)

    # Print to stdout
    print("\n" + report_text)

    # Write to file
    output_dir = os.path.join("results", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_{dataset_slug}_results.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"\nResults saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine-tuned DeBERTa model against the Aegis test set. "
            "Results are written to results/evaluation/. "
            "See ADR-005 for scope rationale."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help=(
            "Path to a saved HuggingFace model directory. "
            "Example: deberta-pi-full-stage3-final"
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(_DATASET_REGISTRY.keys()),
        help=(
            "Dataset to evaluate against. "
            f"Supported: {', '.join(_DATASET_REGISTRY.keys())}. "
            "(D5 = Option A: Aegis only — ADR-005)"
        ),
    )

    args = parser.parse_args()
    evaluate(args.model_path, args.dataset)


if __name__ == "__main__":
    main()
