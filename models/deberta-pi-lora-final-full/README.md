---
base_model: ProtectAI/deberta-v3-base-prompt-injection
library_name: peft
tags:
- base_model:adapter:ProtectAI/deberta-v3-base-prompt-injection
- lora
- transformers
---

# deberta-pi-lora-final-full

LoRA adapter checkpoint for binary prompt injection detection. Adapter weights plus training
configuration — load using PEFT on top of `ProtectAI/deberta-v3-base-prompt-injection`.

This is the final full-output save from a two-stage LoRA fine-tuning pipeline (stages 1 and 2),
including `training_args.bin` for training configuration reference. Stage 3 (Aegis) was not
executed for this LoRA run. For the lightweight adapter-only version, see
`models/deberta-pi-lora-final-adapter/`.

---

## Model Details

### Model Description

- **Task:** Binary sequence classification — prompt injection detection
  - Label `0`: Safe prompt
  - Label `1`: Prompt injection attack
- **Model type:** LoRA adapter (`peft_type: LORA`, `task_type: SEQ_CLS`)
- **Base model:** `ProtectAI/deberta-v3-base-prompt-injection`
- **Fine-tuning approach:** Parameter-efficient fine-tuning (PEFT) with LoRA adapters
- **Language(s):** English
- **License:** MIT
- **Developed by:** BUDDY26
- **PEFT version:** 0.17.1

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| `r` (rank) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.1 |
| `target_modules` | `query_proj`, `key_proj`, `value_proj`, `o_proj` |
| `modules_to_save` | `classifier`, `score` |
| `bias` | `none` |
| `use_dora` | false |
| `use_rslora` | false |

### Training Configuration

| Setting | Value |
|---------|-------|
| Tokenization `max_length` | 256 |
| Tokenization `padding` | `max_length` |
| `per_device_train_batch_size` | 16 |
| `num_train_epochs` (configured) | 10 |
| `metric_for_best_model` | accuracy |
| `load_best_model_at_end` | true |
| Early stopping patience | 3 epochs |

---

## Training Data

Three-stage sequential fine-tuning (stages 1–2 completed for this LoRA pipeline):

| Stage | Dataset | HuggingFace ID | Text column | Label |
|-------|---------|----------------|------------|-------|
| 1 | Safe-Guard Prompt Injection | `xTRam1/safe-guard-prompt-injection` | `text` | integer (0/1) |
| 2 | SPML Chatbot Prompt Injection | `reshabhs/SPML_Chatbot_Prompt_Injection` | `System Prompt` + `User Prompt` | `Prompt injection` column |

Stage 3 (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`) was not executed for this LoRA pipeline.

---

## Evaluation Results

Within-distribution validation accuracy (held-out split of the stage dataset):

| Stage | Dataset | Best validation accuracy | Epoch reached |
|-------|---------|------------------------|---------------|
| Stage 1 | Safe-Guard (held-out) | **98.67%** | Epoch 2 of 10 (early stopping) |
| Stage 2 | SPML (held-out) | **96.07%** | Epoch 8 of 10 |

No cross-dataset evaluation (e.g., on Aegis) is available for this LoRA pipeline. Cross-dataset
results (81.16% accuracy on Aegis) are from the full fine-tuning pipeline; see `results/`.

---

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base_model_id = "ProtectAI/deberta-v3-base-prompt-injection"
adapter_path = "models/deberta-pi-lora-final-full"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=2)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

inputs = tokenizer("Ignore previous instructions and...", return_tensors="pt",
                   truncation=True, padding="max_length", max_length=256)
outputs = model(**inputs)
predicted_label = outputs.logits.argmax(dim=-1).item()
# 0 = safe, 1 = prompt injection
```

---

## Differences from `deberta-pi-lora-final-adapter`

Both directories contain the same LoRA adapter weights (`adapter_config.json`,
`adapter_model.safetensors`) trained on the same stages. This directory additionally includes
`training_args.bin`, which records the full `TrainingArguments` configuration used during
training. Use either directory to load the adapter; use this one if you need access to the
original training configuration.

---

## Limitations and Honest Caveats

- **No Aegis cross-dataset evaluation.** The confirmed 81.16% accuracy result (Aegis test set)
  belongs to the full fine-tuning pipeline, not this LoRA adapter. This model's generalisation
  to the Aegis distribution is untested.
- **Stage 2 run may be incomplete.** The stage 2 training log records 8 of 10 epochs completed
  with `should_training_stop: false`, suggesting the run may have been interrupted before
  graceful early stopping.
- **Within-distribution accuracy may be inflated by pretraining overlap.** The base model
  was pretrained on prompt injection data by ProtectAI; the training sets for stages 1 and 2
  may overlap with that pretraining data.

---

## Repository

`https://github.com/BUDDY26/deberta-prompt-injection-detection`

See `docs/evidence-ledger.md` for the authoritative source of all confirmed training facts.
