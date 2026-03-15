# Operations Runbook

**Project:** deberta-prompt-injection-detection
**Last updated:** 2026-03-14

> Update this runbook whenever setup steps, commands, or environment variables change.

---

## Prerequisites

- Python 3.11 (tested; 3.12 also passes CI)
- pip
- CUDA-capable GPU (strongly recommended; CPU training is supported but impractical at this scale)
- HuggingFace account (required if any dataset requires authentication at download time)
- At least 20 GB free disk space for model checkpoints across three training stages

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/BUDDY26/deberta-prompt-injection-detection.git
cd deberta-prompt-injection-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and fill in real values before proceeding
```

### 4. Run training

Full fine-tuning (all three stages, sequential):

```bash
python src/train.py
```

LoRA adapter training (stages 1–2 only; stage 3 out of scope per ADR-002):

```bash
python src/train_lora.py
```

The original per-stage scripts (`src/finetune.py`, `src/finetune_2.py`) remain available and
produce equivalent results for stages 1–3 of the full fine-tuning pipeline.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Linting and Formatting

```bash
ruff check src/ && black --check src/
```

---

## Structure Validation

```bash
bash scripts/validate-structure.sh
```

---

## Environment Variables

See `.env.example` for the full list of required variables.

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_HOME` | HuggingFace cache directory (model and dataset downloads) | No (defaults to `~/.cache/huggingface`) |
| `CUDA_VISIBLE_DEVICES` | GPU device index or indices for training (e.g., `0`, `0,1`) | No (uses all available GPUs if unset) |
| `MODEL_OUTPUT_DIR` | Root directory for saved model checkpoints | No (scripts default to working directory) |
| `RESULTS_DIR` | Root directory for evaluation output files | No (scripts default to `results/`) |

---

## Troubleshooting

### GPU out of memory (OOM) during training

Reduce `per_device_train_batch_size` in the relevant training script. The confirmed batch size is
`8` for full fine-tuning and `16` for the LoRA runs. Try `4` if OOM errors persist. You can also
enable gradient accumulation to maintain effective batch size.

### Dataset download fails

1. Confirm you have a HuggingFace account and are logged in: `huggingface-cli login`
2. Set `HF_HOME` in `.env` to a directory with sufficient space.
3. The NVIDIA Aegis dataset (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`) may require accepting terms of service on the HuggingFace Hub before it can be downloaded.

### Stage 3 training fails with `FileNotFoundError: deberta-pi-full-final`

Stage 3 (`src/finetune_2.py`) loads the model saved after stages 1 and 2. You must run
`src/finetune.py` first and confirm `deberta-pi-full-final/` was saved before starting stage 3.

### Tests fail on first run

1. Confirm dependencies are installed: `pip install -r requirements-dev.txt`
2. Confirm the exact test command: `pytest tests/ -v`

### Structure validation fails

Run `bash scripts/validate-structure.sh` to see a categorized report.
Common causes: missing required files, unfilled template tokens.

### CI is failing

Check the Actions tab on GitHub. The pipeline runs four jobs: lint, test, validate-structure, and security.

- Lint or test failures indicate source code issues
- validate-structure failures indicate missing required files

---

## Deployment

This repository is a portfolio and research artifact, not a deployed service. No deployment steps
apply. To use a trained checkpoint for inference, load it with `AutoModelForSequenceClassification`
from the HuggingFace `transformers` library, passing the saved model directory as the model path.
