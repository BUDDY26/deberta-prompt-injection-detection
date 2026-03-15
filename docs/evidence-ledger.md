# Evidence Ledger

**Project:** deberta-prompt-injection-detection
**Created:** 2026-03-14
**Author:** Entry protocol scan — read-only repository inspection

> This document records every confirmed technical detail discovered during the initial
> repository scan. Every item is sourced directly from a repository file.
> No inferences or assumptions appear here.
> This ledger is the authoritative technical reference for project reconstruction.

---

## 1. Project Objective

- Binary sequence classification: detect prompt injection attacks in text input.
  Source: `src/finetune.py` line 87 comment — "binary classifier: 0 = safe, 1 = injection"

- Label encoding: `0 = safe`, `1 = unsafe/injection`.
  Source: `src/finetune.py` line 87; `src/test_model.py` lines 210–211

- Task type declared in all PEFT adapter configs: `SEQ_CLS`.
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/adapter_config.json` field `task_type`

- Project title stated in `README.md`: "Multi-stage fine-tuning pipeline for DeBERTa-v3 to detect
  prompt injection attacks using Safe-Guard, SPML, and NVIDIA Aegis datasets."
  Source: `README.md` line 3

- GitHub repository URL: `https://github.com/BUDDY26/deberta-prompt-injection-detection.git`
  Source: `docs/runbooks/operations.md` line 22

---

## 2. Base Model and Framework

### Base model

- `ProtectAI/deberta-v3-base-prompt-injection`
  Source: `src/finetune.py` lines 88–91 (`AutoTokenizer.from_pretrained`, `AutoModelForSequenceClassification.from_pretrained`)

- `ProtectAI/deberta-v3-base-prompt-injection` confirmed as `base_model_name_or_path` in all four
  LoRA adapter configs.
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/adapter_config.json` field `base_model_name_or_path`;
  `models/deberta-pi-lora-stage2/checkpoint-7208/adapter_config.json` field `base_model_name_or_path`;
  `models/deberta-pi-lora-final-adapter/adapter_config.json` field `base_model_name_or_path`;
  `models/deberta-pi-lora-final-full/adapter_config.json` field `base_model_name_or_path`

### Framework and libraries (full fine-tuning pipeline)

- HuggingFace `transformers`: `AutoTokenizer`, `AutoModelForSequenceClassification`,
  `TrainingArguments`, `Trainer`, `EarlyStoppingCallback`
  Source: `src/finetune.py` lines 4–11; `src/finetune_2.py` lines 4–11

- HuggingFace `datasets`: `load_dataset`
  Source: `src/finetune.py` line 3; `src/finetune_2.py` line 3

- HuggingFace `evaluate`: `evaluate.load("accuracy")`
  Source: `src/finetune.py` line 17; `src/finetune_2.py` line 17

- `torch` (PyTorch)
  Source: `src/finetune.py` line 2; `src/finetune_2.py` line 2; `src/test_model.py` line 5

- `matplotlib.pyplot`
  Source: `src/finetune.py` line 13; `src/finetune_2.py` line 13

- `sklearn.metrics`: `accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`,
  `classification_report`
  Source: `src/test_model.py` line 8

- `numpy`
  Source: `src/test_model.py` line 9

- `tqdm`
  Source: `src/test_model.py` line 10

### Framework and libraries (LoRA pipeline)

- PEFT version: `0.17.1`
  Source: `models/deberta-pi-lora-stage1-final/README.md` line 205;
  `models/deberta-pi-lora-final-full/README.md` line 205

- PEFT type: `LORA`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/adapter_config.json` field `peft_type`

- Library name declared in model card frontmatter: `peft`; tags: `lora`, `transformers`
  Source: `models/deberta-pi-lora-stage1-final/README.md` lines 2–7;
  `models/deberta-pi-lora-final-full/README.md` lines 2–7

### Python version

- Python `3.11` (primary); `3.12` (CI test matrix secondary)
  Source: `.github/workflows/ci.yml` lines 36, 70

### Linting and formatting tools

- `ruff` (linter) and `black` (formatter)
  Source: `.github/workflows/ci.yml` lines 41, 44, 47

### Security scanning

- `bandit` — Python security linter, run against `src/` with `-ll` (low-severity minimum),
  `--exit-zero` (non-blocking)
  Source: `.github/workflows/ci.yml` lines 138–139

---

## 3. Training Pipeline Stages

Two distinct training pipelines coexist in this repository.

**Pipeline A — Full fine-tuning** (`src/finetune.py`, `src/finetune_2.py`)
Outputs: `deberta-pi-full-stage1-final`, `deberta-pi-full-final`, `deberta-pi-full-stage3-final`

**Pipeline B — LoRA fine-tuning** (training script not present in `src/`)
Outputs: `models/deberta-pi-lora-stage1/`, `models/deberta-pi-lora-stage2/`, etc.

### Stage 1 (both pipelines)

- Dataset: `xTRam1/safe-guard-prompt-injection`
  Source: `src/finetune.py` line 94

- Text column: `"text"`
  Source: `src/finetune.py` line 105

- Label column: `"label"` (integer)
  Source: `src/finetune.py` line 94 comment; line 112 rename check

- Validation split: 10% drawn from `ds1["train"]` with `seed=42`
  Source: `src/finetune.py` lines 98–100

- Test split: native `ds1["test"]`
  Source: `src/finetune.py` line 101

- Output directory (full fine-tuning): `deberta-pi-full-stage1`
  Source: `src/finetune.py` line 121

- Saved model path (full fine-tuning): `deberta-pi-full-stage1-final`
  Source: `src/finetune.py` lines 158–159

#### Stage 1 hyperparameters — full fine-tuning

- `learning_rate`: `2e-5`
  Source: `src/finetune.py` line 122

- `per_device_train_batch_size`: `8`
  Source: `src/finetune.py` line 120

- `per_device_eval_batch_size`: `16`
  Source: `src/finetune.py` line 121 (second argument)

- `num_train_epochs`: `10`
  Source: `src/finetune.py` line 125

- `fp16`: `torch.cuda.is_available()` (enabled only when CUDA present)
  Source: `src/finetune.py` line 126

- `eval_strategy`: `"epoch"`
  Source: `src/finetune.py` line 127

- `save_strategy`: `"epoch"`
  Source: `src/finetune.py` line 128

- `load_best_model_at_end`: `True`
  Source: `src/finetune.py` line 129

- `metric_for_best_model`: `"accuracy"`
  Source: `src/finetune.py` line 130

- `greater_is_better`: `True`
  Source: `src/finetune.py` line 131

- `save_total_limit`: `1`
  Source: `src/finetune.py` line 133

- `logging_steps`: `50`
  Source: `src/finetune.py` line 134

- `report_to`: `"none"`
  Source: `src/finetune.py` line 135

- `early_stopping_patience`: `3`
  Source: `src/finetune.py` line 144

#### Stage 1 hyperparameters — LoRA run (from trainer_state.json and notebook)

- `train_batch_size`: `16`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `train_batch_size`;
  confirmed by `notebooks/Module2Project (1).ipynb` cell 9 `per_device_train_batch_size=16`

- `per_device_eval_batch_size`: `32`
  Source: `notebooks/Module2Project (1).ipynb` cell 9 `per_device_eval_batch_size=32`

- `learning_rate`: `2e-4`
  Source: `notebooks/Module2Project (1).ipynb` cell 9 `learning_rate=2e-4`
  Note: differs from full fine-tuning pipeline (`2e-5`); higher LR is standard for LoRA
  (fewer parameters updated).

- `logging_steps`: `50`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `logging_steps`

- `save_steps`: `500`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `save_steps`

- `num_train_epochs`: `10`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `num_train_epochs`
  Note: the notebook currently shows `num_train_epochs=20`, but the artifact and notebook
  execution output (`epoch: 5.0` at test evaluation, early stopped) confirm the run used 10.
  The notebook was edited after training.

- `max_steps`: `4640`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `max_steps`

- `early_stopping_patience`: `3`, `early_stopping_threshold`: `0.0`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json`
  field `stateful_callbacks.EarlyStoppingCallback.args`

### Stage 2

- Dataset: `reshabhs/SPML_Chatbot_Prompt_Injection`
  Source: `src/finetune.py` line 167

- Text input: concatenation of `"System Prompt"` + `" "` + `"User Prompt"` when both columns
  present; falls back to `"User Prompt"` alone, then `"System Prompt"` alone, then searches for
  columns named `text`, `prompt`, `input`, `sentence`, `content`
  Source: `src/finetune.py` lines 188–203

- Label column: `"Prompt injection"` renamed to `"label"` before tokenization
  Source: `src/finetune.py` lines 209–212

- Validation split: uses native `"validation"` split if present; otherwise 10% from train with
  `seed=42`
  Source: `src/finetune.py` lines 170–178

- Test split: native `"test"` if present; otherwise reuses validation split
  Source: `src/finetune.py` lines 172–173, 177–178

- Model input for stage 2: the same model object in memory after stage 1 training (no reload)
  Source: `src/finetune.py` line 247 comment — "Continue training with the same model from stage 1"

- Output directory (full fine-tuning): `deberta-pi-full-stage2`
  Source: `src/finetune.py` line 230

- Saved model path after both stages (full fine-tuning): `deberta-pi-full-final`
  Source: `src/finetune.py` lines 272–273

#### Stage 2 hyperparameters — full fine-tuning

- `learning_rate`: `2e-5`
  Source: `src/finetune.py` line 232

- `per_device_train_batch_size`: `8`
  Source: `src/finetune.py` line 232 (positional; same pattern as stage 1)

- `num_train_epochs`: `15`
  Source: `src/finetune.py` line 234

- `fp16`, `eval_strategy`, `save_strategy`, `load_best_model_at_end`, `metric_for_best_model`,
  `greater_is_better`, `save_total_limit`, `logging_steps`, `report_to`: identical to stage 1
  Source: `src/finetune.py` lines 235–244

- `early_stopping_patience`: `3`
  Source: `src/finetune.py` line 253

#### Stage 2 hyperparameters — LoRA run (from trainer_state.json and notebook)

- `train_batch_size`: `16`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` field `train_batch_size`;
  confirmed by `notebooks/Module2Project (1).ipynb` cell 10 `per_device_train_batch_size=16`

- `per_device_eval_batch_size`: `32`
  Source: `notebooks/Module2Project (1).ipynb` cell 10 `per_device_eval_batch_size=32`

- `learning_rate`: `2e-4`
  Source: `notebooks/Module2Project (1).ipynb` cell 10 `learning_rate=2e-4`

- `num_train_epochs`: `10`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` field `num_train_epochs`
  Note: the notebook currently shows `num_train_epochs=20`, but the artifact and notebook
  execution output (`epoch: 10.0` at test evaluation) confirm the run used 10.

- `max_steps`: `9010`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` field `max_steps`

- `logging_steps`: `50`, `save_steps`: `500`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json`

- `early_stopping_patience`: `3`, `early_stopping_threshold`: `0.0`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json`
  field `stateful_callbacks.EarlyStoppingCallback.args`

### Stage 3

- Dataset: `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`
  Source: `src/finetune_2.py` line 94

- Text column: `"prompt"`
  Source: `src/finetune_2.py` line 109

- Label column: `"prompt_label"` (string); mapped `"unsafe"` → `1`, all other values → `0`
  Source: `src/finetune_2.py` lines 111–112

- Splits: uses native `train`, `validation`, `test` splits from the dataset
  Source: `src/finetune_2.py` lines 101–103

- Model input for stage 3: loaded from `"deberta-pi-full-final"` (the output of stages 1+2)
  Source: `src/finetune_2.py` lines 88–91

- Output directory: `deberta-pi-full-stage3`
  Source: `src/finetune_2.py` line 140

- Saved model path: `deberta-pi-full-stage3-final`
  Source: `src/finetune_2.py` lines 177–178

#### Stage 3 hyperparameters — full fine-tuning

- `learning_rate`: `2e-5`
  Source: `src/finetune_2.py` line 141

- `per_device_train_batch_size`: `8`; `per_device_eval_batch_size`: `16`
  Source: `src/finetune_2.py` lines 139–140

- `num_train_epochs`: `25`
  Source: `src/finetune_2.py` line 144

- `fp16`: `torch.cuda.is_available()`
  Source: `src/finetune_2.py` line 145

- `eval_strategy`: `"epoch"`, `save_strategy`: `"epoch"`
  Source: `src/finetune_2.py` lines 146–147

- `load_best_model_at_end`: `True`; `metric_for_best_model`: `"accuracy"`;
  `greater_is_better`: `True`; `save_total_limit`: `1`
  Source: `src/finetune_2.py` lines 148–152

- `logging_steps`: `50`; `report_to`: `"none"`
  Source: `src/finetune_2.py` lines 153–154

- `early_stopping_patience`: `3`
  Source: `src/finetune_2.py` line 163

---

## 4. Datasets

### Dataset 1 — Safe-Guard Prompt Injection

- HuggingFace dataset ID: `xTRam1/safe-guard-prompt-injection`
  Source: `src/finetune.py` line 94

- Confirmed column names used: `"text"` (input), `"label"` (integer target)
  Source: `src/finetune.py` line 94 comment; lines 105, 112

- Dataset has a native `"test"` split
  Source: `src/finetune.py` line 101 (`ds1["test"]` accessed without guard)

### Dataset 2 — SPML Chatbot Prompt Injection

- HuggingFace dataset ID: `reshabhs/SPML_Chatbot_Prompt_Injection`
  Source: `src/finetune.py` line 167

- Confirmed column names: `"System Prompt"`, `"User Prompt"`, `"Prompt injection"` (label),
  `"Degree"`, `"Source"` — documented in source comment
  Source: `src/finetune.py` lines 183–185

- Validation split presence is conditional (code checks for it at runtime)
  Source: `src/finetune.py` lines 170–178

### Dataset 3 — NVIDIA Aegis AI Content Safety 2.0

- HuggingFace dataset ID: `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`
  Source: `src/finetune_2.py` line 94; `src/test_model.py` line 45;
  `results/test_results_2dataset.txt` line 3; `results/test_results_3datasets.txt` line 3

- Confirmed column names: `"prompt"` (input text), `"prompt_label"` (string label)
  Source: `src/finetune_2.py` lines 109–112; `src/test_model.py` lines 83–84

- Native `train`, `validation`, and `test` splits confirmed present
  Source: `src/finetune_2.py` lines 101–103 (accessed directly without guard)

- Test split size: `1964` examples
  Source: `results/test_results_2dataset.txt` line 4; `results/test_results_3datasets.txt` line 4

- Test split label distribution: `Safe (0): 905`, `Unsafe (1): 1059`
  Source: `results/test_results_3datasets.txt` confusion matrix lines 18–20
  (row sums: Safe=905, Unsafe=1059)

---

## 5. Evaluation Metrics

### During training

- Metric: `accuracy` loaded via `evaluate.load("accuracy")`
  Source: `src/finetune.py` line 17; `src/finetune_2.py` line 17

- Computation: `logits.argmax(axis=-1)` compared to labels
  Source: `src/finetune.py` lines 20–22; `src/finetune_2.py` lines 20–22

- Logged fields per evaluation step: `eval_loss`, `eval_accuracy`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` log_history entries

### Post-training test evaluation (`src/test_model.py`)

- Overall accuracy (`accuracy_score`)
  Source: `src/test_model.py` lines 193–194

- Binary precision, recall, F1-score — positive class: `1` (Unsafe); `zero_division=0`
  Source: `src/test_model.py` lines 197–203

- Per-class precision, recall, F1-score, support — classes `[0, 1]`; `zero_division=0`
  Source: `src/test_model.py` lines 206–211

- Confusion matrix (`confusion_matrix`)
  Source: `src/test_model.py` line 214

- Full `classification_report` with `target_names=['Safe', 'Unsafe']`, `digits=4`
  Source: `src/test_model.py` lines 222–229

### Inference configuration

- Batch size for inference: `16`
  Source: `src/test_model.py` line 16 (`BATCH_SIZE = 16`)

- Classification decision rule: `torch.argmax(logits, dim=-1)` — no custom threshold
  Source: `src/test_model.py` line 184

- Device: CUDA if available, else CPU
  Source: `src/test_model.py` line 27

---

## 6. Checkpointing and Early Stopping

### Confirmed settings (all stages, full fine-tuning)

- `load_best_model_at_end`: `True`
  Source: `src/finetune.py` line 129; `src/finetune_2.py` line 148

- `metric_for_best_model`: `"accuracy"`
  Source: `src/finetune.py` line 130; `src/finetune_2.py` line 149

- `greater_is_better`: `True`
  Source: `src/finetune.py` line 131; `src/finetune_2.py` line 150

- `save_total_limit`: `1` (only the single best checkpoint is retained)
  Source: `src/finetune.py` line 133; `src/finetune_2.py` line 151

- `early_stopping_patience`: `3` epochs (stop if no improvement for 3 consecutive evaluations)
  Source: `src/finetune.py` lines 143–145; `src/finetune_2.py` lines 162–164

### Confirmed settings (LoRA runs, from trainer_state.json)

- `early_stopping_threshold`: `0.0`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json`
  `stateful_callbacks.EarlyStoppingCallback.args.early_stopping_threshold`;
  `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` same field

### Stage 1 LoRA — best checkpoint

- `best_global_step`: `928`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `best_global_step`

- `best_metric` (eval accuracy): `0.9866504854368932`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `best_metric`

- `best_model_checkpoint`: `deberta-pi-lora-stage1/checkpoint-928`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json`
  field `best_model_checkpoint`

- Epoch at best checkpoint: `2.0`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `epoch`

- `total_flos`: `1970394277724160.0`
  Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `total_flos`

### Stage 1 LoRA — full eval history

| Epoch | Step | eval_accuracy | eval_loss |
|-------|------|--------------|-----------|
| 1.0 | 464 | 0.9793689320388349 | 0.09352463483810425 |
| 2.0 | 928 | 0.9866504854368932 | 0.08859251439571380 |

Source: `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` `log_history`

### Stage 2 LoRA — best checkpoint

- `best_global_step`: `7208`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` field `best_global_step`

- `best_metric` (eval accuracy): `0.9606741573033708`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` field `best_metric`

- `best_model_checkpoint`: `deberta-pi-lora-stage2/checkpoint-7208`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json`
  field `best_model_checkpoint`

- Epoch at best checkpoint: `8.0`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` field `epoch`

- `total_flos`: `1.53229258186752e+16`
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` field `total_flos`

### Stage 2 LoRA — full eval history

| Epoch | Step | eval_accuracy | eval_loss |
|-------|------|--------------|-----------|
| 1.0 | 901 | 0.9419475655430711 | 0.20562943816184998 |
| 2.0 | 1802 | 0.9531835205992509 | 0.21720524132251740 |
| 3.0 | 2703 | 0.9556803995006242 | 0.20184718072414398 |
| 4.0 | 3604 | 0.9588014981273408 | 0.15827576816082000 |
| 5.0 | 4505 | 0.9575530586766542 | 0.15521152317523956 |
| 6.0 | 5406 | 0.9588014981273408 | 0.15732221305370330 |
| 7.0 | 6307 | 0.9594257178526842 | 0.17076650261878967 |
| 8.0 | 7208 | 0.9606741573033708 | 0.13901996612548828 |

Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` `log_history`

---

## 7. Repository Artifacts

### Source files

| File | Content |
|------|---------|
| `src/finetune.py` | Stages 1 and 2 — full fine-tuning pipeline + metric plotting |
| `src/finetune_2.py` | Stage 3 — full fine-tuning pipeline + metric plotting |
| `src/test_model.py` | Model evaluation script — inference + full metrics report |
| `src/.gitkeep` | Empty placeholder |

Source: repository file listing

### Notebooks

| File | Content |
|------|---------|
| `notebooks/Module2Project (1).ipynb` | Jupyter notebook — LoRA training pipeline, stages 1 and 2 only |

Source: repository file listing; notebook inspection (read-only)

**Notebook structure confirmed by inspection:**

- Cells 0–7: preliminary exploration (distilbert, IMDB, SST2); unrelated to LoRA pipeline
- Cell 9: Stage 1 LoRA training code — executed, with full output
- Cell 10: Stage 2 LoRA training code — executed, with full output; saves both final model artifacts
- Cell 11: Colab zip command (`!zip -r trained-models.zip trained-models`) — Colab-specific
- Cell 16: Planning comment for "Part #3 Nvidia / Aegis" — **no code written**
- Cell 17: Commented-out, unexecuted block using `nvidia/llama-3.1-nemoguard-8b-content-safety`
  wrapping `meta-llama/Meta-Llama-3.1-8B-Instruct` — different model family, not executed

**Stage 3 LoRA was never written in the notebook.** No DeBERTa LoRA training code exists for
the Aegis dataset in any notebook cell.

**Save methods confirmed:**

- `deberta-pi-lora-final-adapter`: saved via `model.save_pretrained("deberta-pi-lora-final-adapter")`
  Source: `notebooks/Module2Project (1).ipynb` cell 10

- `deberta-pi-lora-final-full`: saved via `trainer2.save_model("deberta-pi-lora-final-full")`
  Source: `notebooks/Module2Project (1).ipynb` cell 10

**Stage 2 LoRA in-memory continuation confirmed:**

The notebook continues training the same model object from stage 1 into stage 2 without
reloading, matching the full fine-tuning pipeline's stage 2 behaviour.
Source: `notebooks/Module2Project (1).ipynb` cell 10 — `model=model` passed to `Trainer`

### Model artifacts — LoRA checkpoints

| Directory | Contents |
|-----------|---------|
| `models/deberta-pi-lora-stage1/checkpoint-928/` | `adapter_config.json`, `adapter_model.safetensors`, `optimizer.pt`, `rng_state.pth`, `scaler.pt`, `scheduler.pt`, `trainer_state.json`, `training_args.bin` |
| `models/deberta-pi-lora-stage2/checkpoint-7208/` | Same set of files |

Source: repository file listing

### Model artifacts — LoRA final models

| Directory | Contents |
|-----------|---------|
| `models/deberta-pi-lora-stage1-final/` | `adapter_config.json`, `adapter_model.safetensors`, `README.md` |
| `models/deberta-pi-lora-final-adapter/` | `adapter_config.json`, `adapter_model.safetensors`, `README.md` |
| `models/deberta-pi-lora-final-full/` | `adapter_config.json`, `adapter_model.safetensors`, `README.md`, `training_args.bin` |

Source: repository file listing

### Result files

| File | Model evaluated | Dataset |
|------|----------------|---------|
| `results/test_results_2dataset.txt` | `deberta-pi-full-stage2-final` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` |
| `results/test_results_3datasets.txt` | `deberta-pi-full-stage3-final` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` |

Source: `results/test_results_2dataset.txt` lines 5–6; `results/test_results_3datasets.txt` lines 5–6

### LoRA adapter configuration (all four saved adapters — identical values)

| Parameter | Value |
|-----------|-------|
| `peft_type` | `LORA` |
| `task_type` | `SEQ_CLS` |
| `r` | `16` |
| `lora_alpha` | `32` |
| `lora_dropout` | `0.1` |
| `target_modules` | `["value_proj", "query_proj", "key_proj", "o_proj"]` |
| `modules_to_save` | `["classifier", "score"]` |
| `bias` | `"none"` |
| `use_dora` | `false` |
| `use_rslora` | `false` |
| `use_qalora` | `false` |
| `init_lora_weights` | `true` |
| `inference_mode` | `true` |

Source: `models/deberta-pi-lora-stage1/checkpoint-928/adapter_config.json`;
`models/deberta-pi-lora-stage2/checkpoint-7208/adapter_config.json`;
`models/deberta-pi-lora-final-adapter/adapter_config.json`;
`models/deberta-pi-lora-final-full/adapter_config.json`

### Tokenization — confirmed for all three stages

| Parameter | Value |
|-----------|-------|
| `truncation` | `True` |
| `padding` | `"max_length"` |
| `max_length` | `256` |

Source: `src/finetune.py` line 105; `src/finetune_2.py` line 109; `src/test_model.py` line 16

### CI/CD pipeline

- Trigger: push and pull request to `main` and `develop` branches
  Source: `.github/workflows/ci.yml` lines 11–14

- Jobs: `lint`, `test` (depends on lint), `validate-structure`, `security`
  Source: `.github/workflows/ci.yml` lines 22, 62, 105, 120

- Test matrix: Python `3.11` and `3.12`, runs on `ubuntu-latest`
  Source: `.github/workflows/ci.yml` lines 63, 70

- Test command: `pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing`
  Source: `.github/workflows/ci.yml` lines 88–89

- Coverage report uploaded as artifact (retained 7 days, Python 3.11 run only)
  Source: `.github/workflows/ci.yml` lines 91–97

- Security job: runs `bandit -r src/ -ll --exit-zero` on push to `main` only
  Source: `.github/workflows/ci.yml` lines 125, 138–139

- Structure validation command: `bash scripts/validate-structure.sh --strict`
  Source: `.github/workflows/ci.yml` line 118

---

## 8. Experiment Results

All results are from `results/` directory. Both runs evaluated on
`nvidia/Aegis-AI-Content-Safety-Dataset-2.0` test split (`N = 1964`, device: `cuda`).

### Result 1 — Stage 2 final model on Aegis test set

- Model: `deberta-pi-full-stage2-final`
  Source: `results/test_results_2dataset.txt` line 5

- Overall accuracy: `0.4160` (41.60%)
  Source: `results/test_results_2dataset.txt` line 9

- Precision (Unsafe class, binary): `0.4257`
  Source: `results/test_results_2dataset.txt` line 11

- Recall (Unsafe class, binary): `0.2380`
  Source: `results/test_results_2dataset.txt` line 12

- F1-score (Unsafe class, binary): `0.3053`
  Source: `results/test_results_2dataset.txt` line 13

- Confusion matrix:

  |  | Predicted Safe | Predicted Unsafe |
  |--|---------------|-----------------|
  | Actual Safe | 565 | 340 |
  | Actual Unsafe | 807 | 252 |

  Source: `results/test_results_2dataset.txt` lines 17–19

- Per-class detailed metrics:

  | Class | Precision | Recall | F1 | Support |
  |-------|-----------|--------|----|---------|
  | Safe | 0.4118 | 0.6243 | 0.4963 | 905 |
  | Unsafe | 0.4257 | 0.2380 | 0.3053 | 1059 |

  Source: `results/test_results_2dataset.txt` lines 23–24

### Result 2 — Stage 3 final model on Aegis test set

- Model: `deberta-pi-full-stage3-final`
  Source: `results/test_results_3datasets.txt` line 5

- Overall accuracy: `0.8116` (81.16%)
  Source: `results/test_results_3datasets.txt` line 9

- Precision (Unsafe class, binary): `0.8247`
  Source: `results/test_results_3datasets.txt` line 11

- Recall (Unsafe class, binary): `0.8263`
  Source: `results/test_results_3datasets.txt` line 12

- F1-score (Unsafe class, binary): `0.8255`
  Source: `results/test_results_3datasets.txt` line 13

- Confusion matrix:

  |  | Predicted Safe | Predicted Unsafe |
  |--|---------------|-----------------|
  | Actual Safe | 719 | 186 |
  | Actual Unsafe | 184 | 875 |

  Source: `results/test_results_3datasets.txt` lines 17–19

- Per-class detailed metrics:

  | Class | Precision | Recall | F1 | Support |
  |-------|-----------|--------|----|---------|
  | Safe | 0.7962 | 0.7945 | 0.7954 | 905 |
  | Unsafe | 0.8247 | 0.8263 | 0.8255 | 1059 |

  Source: `results/test_results_3datasets.txt` lines 23–24

- Macro avg: precision `0.8105`, recall `0.8104`, F1 `0.8104`
  Source: `results/test_results_3datasets.txt` line 25

- Weighted avg: precision `0.8116`, recall `0.8116`, F1 `0.8116`
  Source: `results/test_results_3datasets.txt` line 26

---

## 9. Confirmed Limitations

- No `requirements.txt` exists in the repository. The CI pipeline references
  `pip install -r requirements.txt` but no such file is present.
  Source: repository file listing (no `.txt` file found); `.github/workflows/ci.yml` line 84

- No `src/train.py` exists. Both `README.md` and `docs/runbooks/operations.md` list
  `python src/train.py` as the run command, but no such file is present in `src/`.
  Source: repository file listing; `README.md` line 37; `docs/runbooks/operations.md` line 42

- `tests/unit/` and `tests/integration/` contain only `.gitkeep` — zero test files.
  Source: repository file listing

- `docs/architecture.md` contains only unfilled template placeholders.
  Source: `docs/architecture.md` lines 13, 21, 35, 43, 51

- `docs/adr/ADR-001-template.md` is an unfilled template file, not a recorded decision.
  Source: `docs/adr/ADR-001-template.md` lines 1–8 (instruction header present)

- `docs/qa/qa-plan.md` has unfilled sections: Test Strategy, Test File Inventory, Known Gaps.
  Source: `docs/qa/qa-plan.md` lines 13, 57, 72

- `docs/runbooks/operations.md` has unfilled Prerequisites section and generic environment
  variables (`APP_ENV`, `APP_PORT`, `SECRET_KEY`, `DATABASE_URL`) unrelated to an ML project.
  Source: `docs/runbooks/operations.md` lines 12–13, 76–80

- `.env.example` contains generic web-application variables with no ML-specific content.
  Source: `.env.example` lines 6–10

- All model `README.md` files (four) are auto-generated HuggingFace model card templates with
  no fields filled in except `base_model`, `library_name`, `tags`, and PEFT version.
  Source: `models/deberta-pi-lora-stage1-final/README.md` lines 24–204 ("More Information Needed");
  `models/deberta-pi-lora-final-full/README.md` lines 24–204

- The LoRA training script that produced artifacts in `models/` is not present in `src/`.
  Source: `src/` file listing (three files only: `finetune.py`, `finetune_2.py`, `test_model.py`,
  `.gitkeep`); `models/` contains LoRA adapter artifacts confirming a LoRA run occurred

- `README.md` contains two unfilled `{{LINT_COMMAND}}` placeholder tokens.
  Source: `README.md` line 50

- `docs/runbooks/operations.md` contains the same unfilled `{{LINT_COMMAND}}{{LINT_COMMAND}}`
  placeholder in the linting command.
  Source: `docs/runbooks/operations.md` line 58

---

## 10. Reproducibility Risks

- No `requirements.txt`: all library versions (except PEFT 0.17.1) are unknown.
  Source: repository file listing; `models/deberta-pi-lora-stage1-final/README.md` line 205

- No `transformers.set_seed()` or `torch.manual_seed()` call in any training script.
  Source: `src/finetune.py` full text; `src/finetune_2.py` full text

- `seed=42` is set only in `train_test_split` calls for datasets 1 and 2.
  Source: `src/finetune.py` lines 98, 175

- Full fine-tuning scripts output to paths (`deberta-pi-full-*`) that are absent from the
  repository; LoRA models (`deberta-pi-lora-*`) are present in `models/` but their training
  script is absent from `src/`. Both pipelines cannot be fully reproduced from `src/` alone.
  Source: `src/finetune.py` lines 158, 272; repository file listing

- The evaluation script (`src/test_model.py`) saves results to `test_results_2.txt` (line 291)
  but the actual file in the repository is named `results/test_results_2dataset.txt`. The path
  in the script does not match the file that exists.
  Source: `src/test_model.py` line 291; repository file listing

- Stage 2 LoRA run: `trainer_state.json` records `num_train_epochs=10`, best checkpoint at
  `epoch=8.0`, and `should_training_stop: false`. Notebook inspection resolves this: training
  ran to the `num_train_epochs=10` epoch limit (`epoch: 10.0` in notebook execution output).
  Early stopping patience=3 would have required epoch 11 to trigger (best at epoch 8 + 3
  more), but the epoch limit was reached first. **The run completed normally — it was not
  interrupted.** The `should_training_stop: false` at epoch 8 correctly reflects that early
  stopping had not yet fired; termination occurred at epoch 10 due to the epoch budget.
  Source: `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json`;
  `notebooks/Module2Project (1).ipynb` cell 10 execution output `epoch: 10.0`

- No hardware specification is recorded anywhere in the repository. GPU type, VRAM, and
  training duration are unknown.
  Source: all model `README.md` files — "Hardware Type: More Information Needed"

---

*Evidence ledger complete. Contents are read-only confirmed facts. Do not add inferences to this document.*
*Last updated: 2026-03-14*
