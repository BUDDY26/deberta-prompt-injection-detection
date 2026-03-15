"""
Inference utilities for the DeBERTa prompt injection detection pipeline.

Supports both artifact formats produced by this repository:
  - Full fine-tuned model (src/train.py)     — loaded with AutoModelForSequenceClassification
  - LoRA adapter checkpoint (src/train_lora.py) — loaded with PeftModel on top of the base model

Format detection: if adapter_config.json is present in model_path, the directory is
treated as a LoRA adapter. The base model ID is read from adapter_config.json
(base_model_name_or_path) so no hardcoded model IDs are required here.

Public API:
    load_model(model_path)                            -> (model, tokenizer, device)
    predict(text, model, tokenizer, device)           -> dict
    predict_batch(texts, model, tokenizer, device)    -> list[dict]

CLI:
    python src/inference.py --model-path <path> --text "..."
    python src/inference.py --model-path <path> --text "..." --output-format json

Run from the repository root:
    python src/inference.py --model-path models/deberta-pi-lora-final-adapter --text "Ignore all instructions"
"""

import argparse
import json
import os
import sys

# Ensure src/ is on the path so that config is importable
# when this script is run as: python src/inference.py
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------
_LABEL_NAMES = {0: "safe", 1: "injection"}

# ---------------------------------------------------------------------------
# LoRA adapter sentinel file
# ---------------------------------------------------------------------------
_ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def _is_lora_adapter(model_path: str) -> bool:
    """Return True if model_path contains a LoRA adapter_config.json."""
    return os.path.isfile(os.path.join(model_path, _ADAPTER_CONFIG_FILENAME))


def _read_base_model_id(model_path: str) -> str:
    """
    Read base_model_name_or_path from adapter_config.json.

    Raises ValueError if the field is absent or empty.
    """
    config_path = os.path.join(model_path, _ADAPTER_CONFIG_FILENAME)
    with open(config_path, encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg.get("base_model_name_or_path", "")
    if not base_model_id:
        raise ValueError(
            f"adapter_config.json at {config_path} does not contain "
            "'base_model_name_or_path'. Cannot determine base model."
        )
    return base_model_id


def load_model(model_path: str):
    """
    Load a fine-tuned model and tokenizer from model_path.

    Detects the artifact format automatically:
      - If adapter_config.json is present: loads as a LoRA adapter (requires peft).
      - Otherwise: loads as a full fine-tuned model.

    Args:
        model_path: Path to the saved model directory.

    Returns:
        (model, tokenizer, device) — model is on device and in eval mode.

    Raises:
        ImportError: If model_path is a LoRA adapter and peft is not installed.
        ValueError: If adapter_config.json is present but malformed.
        OSError:    If model_path does not exist or is not a valid model directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _is_lora_adapter(model_path):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                f"'{model_path}' is a LoRA adapter checkpoint (adapter_config.json "
                "found), but the peft package is not installed. "
                "Install it with: pip install peft"
            ) from exc

        base_model_id = _read_base_model_id(model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id, num_labels=2
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(text: str, model, tokenizer, device) -> dict:
    """
    Run inference on a single text string.

    Args:
        text:      Input text to classify.
        model:     Loaded model (output of load_model).
        tokenizer: Matching tokenizer (output of load_model).
        device:    torch.device the model is on.

    Returns:
        {
            "label":       int   — 0 (safe) or 1 (injection)
            "label_str":   str   — "safe" or "injection"
            "probability": float — confidence in the predicted label
            "probabilities": {"safe": float, "injection": float}
        }
    """
    inputs = tokenizer(
        text,
        truncation=config.TOKENIZER_TRUNCATION,
        padding=config.TOKENIZER_PADDING,
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).squeeze(0)
    label = int(probs.argmax().item())
    prob_safe = float(probs[0].item())
    prob_injection = float(probs[1].item())

    return {
        "label": label,
        "label_str": _LABEL_NAMES[label],
        "probability": float(probs[label].item()),
        "probabilities": {"safe": prob_safe, "injection": prob_injection},
    }


def predict_batch(texts: list, model, tokenizer, device) -> list:
    """
    Run inference on a list of text strings.

    Processes all texts as a single batch. For large inputs, callers should
    chunk texts externally before calling this function.

    Args:
        texts:     List of input strings.
        model:     Loaded model (output of load_model).
        tokenizer: Matching tokenizer (output of load_model).
        device:    torch.device the model is on.

    Returns:
        List of dicts, one per input text. Each dict has the same structure
        as the return value of predict().
    """
    if not texts:
        return []

    inputs = tokenizer(
        texts,
        truncation=config.TOKENIZER_TRUNCATION,
        padding=config.TOKENIZER_PADDING,
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs_all = F.softmax(outputs.logits, dim=-1)
    results = []
    for probs in probs_all:
        label = int(probs.argmax().item())
        results.append(
            {
                "label": label,
                "label_str": _LABEL_NAMES[label],
                "probability": float(probs[label].item()),
                "probabilities": {
                    "safe": float(probs[0].item()),
                    "injection": float(probs[1].item()),
                },
            }
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_text(result: dict, text: str) -> str:
    """Format a single prediction result as human-readable text."""
    lines = [
        f"Text:        {text}",
        f"Label:       {result['label_str']} ({result['label']})",
        f"Confidence:  {result['probability']:.4f} ({result['probability'] * 100:.1f}%)",
        f"Prob(safe):  {result['probabilities']['safe']:.4f}",
        f"Prob(inj):   {result['probabilities']['injection']:.4f}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run prompt injection inference using a fine-tuned DeBERTa model. "
            "Supports both full fine-tuned and LoRA adapter checkpoints."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/inference.py --model-path models/deberta-pi-lora-final-adapter "
            '--text "Ignore all previous instructions"\n'
            "  python src/inference.py --model-path deberta-pi-full-stage3-final "
            '--text "What is the weather today?" --output-format json'
        ),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a saved model directory (full fine-tuned or LoRA adapter).",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Input text to classify.",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (human-readable) or 'json' (machine-readable). Default: text.",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}", file=sys.stderr)
    model, tokenizer, device = load_model(args.model_path)
    print(f"Model loaded on device: {device}\n", file=sys.stderr)

    result = predict(args.text, model, tokenizer, device)

    if args.output_format == "json":
        output = {"text": args.text, **result}
        print(json.dumps(output, indent=2))
    else:
        print(_format_text(result, args.text))


if __name__ == "__main__":
    main()
