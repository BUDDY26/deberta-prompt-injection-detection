# Model Artifacts

This directory contains trained model artifacts from the DeBERTa prompt injection detection pipeline.

Binary weights and checkpoint files are **not tracked by git** (large files). Only documentation
files (`README.md`, `MANIFEST.md`) and LoRA adapter documentation are tracked.

See `models/MANIFEST.md` for a complete inventory of artifact directories and their contents.

---

## Pipelines

Two training pipelines produce artifacts in this directory:

| Pipeline | Entry point | Artifact directories |
|----------|------------|---------------------|
| Full fine-tuning | `src/train.py` | `deberta-pi-full-*` (gitignored binaries) |
| LoRA adapter | `src/train_lora.py` | `deberta-pi-lora-*` |

---

## LoRA Adapter Artifacts (git-tracked documentation)

| Directory | Description |
|-----------|-------------|
| `deberta-pi-lora-stage1/` | LoRA stage 1 best checkpoint (Safe-Guard dataset) |
| `deberta-pi-lora-stage1-final/` | LoRA stage 1 final adapter save |
| `deberta-pi-lora-stage2/` | LoRA stage 2 best checkpoint (SPML dataset) |
| `deberta-pi-lora-final-adapter/` | Final adapter-only save (`model.save_pretrained`) |
| `deberta-pi-lora-final-full/` | Final full save (`trainer.save_model` — includes `training_args.bin`) |

---

## Reproducing Artifacts

```bash
# Full fine-tuning pipeline (stages 1–3)
python src/train.py

# LoRA adapter pipeline (stages 1–2)
python src/train_lora.py
```

Both scripts accept a `--seed` argument (default: 42) for reproducibility.

---

## Evaluation Results

- **Full fine-tuning (Aegis cross-dataset):** 81.16% accuracy, F1 0.8255
- **LoRA Stage 1 (Safe-Guard held-out):** 98.67% at epoch 2
- **LoRA Stage 2 (SPML held-out):** 96.07% at epoch 8

Canonical evaluation outputs from `src/evaluate.py` are written to `results/evaluation/`.
Legacy result files are in `results/`.
