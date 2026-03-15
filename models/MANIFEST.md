# Model Artifact Manifest

Inventory of all artifact directories produced by the training pipelines.
Binary files are gitignored; this manifest documents what each directory contains
so the artifact set can be verified after a training run.

Last updated: 2026-03-15

---

## Full Fine-Tuning Pipeline (`src/train.py`)

These directories contain full model weights (`config.json`, `model.safetensors` or
`pytorch_model.bin`, `tokenizer.*`). Binaries are gitignored.

| Directory | Stage | Dataset | Notes |
|-----------|-------|---------|-------|
| `deberta-pi-stage1/` | Stage 1 checkpoints | Safe-Guard | Trainer checkpoint dir; best checkpoint selected by `load_best_model_at_end` |
| `deberta-pi-stage1-final/` | Stage 1 final | Safe-Guard | `model.save_pretrained` after stage 1 |
| `deberta-pi-stage2/` | Stage 2 checkpoints | SPML | Trainer checkpoint dir |
| `deberta-pi-full-final/` | Stages 1+2 final | Safe-Guard → SPML | `model.save_pretrained`; stage 3 reloads from this path |
| `deberta-pi-stage3/` | Stage 3 checkpoints | Aegis | Trainer checkpoint dir |
| `deberta-pi-stage3-final/` | Stage 3 final | Aegis | `model.save_pretrained`; canonical cross-dataset model |

**Confirmed result:** Stage 3 model on Aegis test set — 81.16% accuracy, F1 0.8255
(source: `docs/evidence-ledger.md` §8)

---

## LoRA Adapter Pipeline (`src/train_lora.py`)

These directories contain adapter weights only (`adapter_config.json`,
`adapter_model.safetensors`). Documentation files (`README.md`) are git-tracked.

| Directory | Stage | Dataset | Notes |
|-----------|-------|---------|-------|
| `deberta-pi-lora-stage1/` | Stage 1 checkpoints | Safe-Guard | Trainer checkpoint dir |
| `deberta-pi-lora-stage1-final/` | Stage 1 final | Safe-Guard | `peft_model.save_pretrained` after stage 1 |
| `deberta-pi-lora-stage2/` | Stage 2 checkpoints | SPML | Trainer checkpoint dir |
| `deberta-pi-lora-final-adapter/` | Stages 1+2 final (adapter) | Safe-Guard → SPML | `model.save_pretrained` — adapter weights only |
| `deberta-pi-lora-final-full/` | Stages 1+2 final (full) | Safe-Guard → SPML | `trainer.save_model` — adapter + `training_args.bin` |

**Confirmed results (within-distribution):**
- Stage 1 best validation accuracy: 98.67% at epoch 2 of 10
- Stage 2 best validation accuracy: 96.07% at epoch 8 of 10

(source: `docs/evidence-ledger.md` §3)

Stage 3 (Aegis) was not executed for the LoRA pipeline. See ADR-002 (D2) and ADR-006.

---

## Git Tracking Policy

```
models/*                          # ignore all binaries by default
!models/README.md                 # track directory README
!models/MANIFEST.md               # track this manifest
!models/deberta-pi-lora-*/        # allow lora subdirs to be entered
models/deberta-pi-lora-*/*        # ignore binaries within lora subdirs
!models/deberta-pi-lora-*/README.md  # track lora subdirectory READMEs
```

Full fine-tuning model weights (`deberta-pi-full-*`, `deberta-pi-stage*`) are never tracked.
LoRA adapter weights (`.safetensors`, `training_args.bin`) are never tracked.
Only Markdown documentation files are tracked.
