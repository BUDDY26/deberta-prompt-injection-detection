# ADR-002: Canonical Training Strategy — Full Fine-Tuning, LoRA, or Both

**Date:** 2026-03-14
**Status:** Accepted
**Author:** BUDDY26

---

## Context

The repository contains evidence of two distinct training pipelines that were both
executed against the same base model and dataset sequence. They differ in whether
the base model's weights are fully updated or whether only a small set of adapter
layers are trained.

### Pipeline A — Full Fine-Tuning (source present in `src/`)

All model weights are updated during training.
`AutoModelForSequenceClassification` is used directly with the HuggingFace `Trainer`.

| Property | Value | Source |
|----------|-------|--------|
| Training scripts | `src/finetune.py` (stages 1+2), `src/finetune_2.py` (stage 3) | evidence-ledger §7 |
| Stage 1 hyperparameters | LR=2e-5, batch=8, max_epochs=10, patience=3 | evidence-ledger §3 |
| Stage 2 hyperparameters | LR=2e-5, batch=8, max_epochs=15, patience=3 | evidence-ledger §3 |
| Stage 3 hyperparameters | LR=2e-5, batch=8, max_epochs=25, patience=3 | evidence-ledger §3 |
| Checkpoint format | Full model weights + tokenizer | evidence-ledger §7 |
| Stage 3 Aegis test accuracy | **81.16%** | evidence-ledger §8 |
| Stage 3 Aegis F1 (Unsafe) | **0.8255** | evidence-ledger §8 |
| Stage 2 Aegis test accuracy | **41.60%** (cross-dataset baseline) | evidence-ledger §8 |
| Source in `src/` | Yes — reproducible from current scripts | evidence-ledger §7 |
| Full pipeline artifacts in repo | No — `deberta-pi-full-*` paths not committed | evidence-ledger §10 |

### Pipeline B — LoRA Fine-Tuning (source NOT present in `src/`)

Only the LoRA adapter layers and the classifier/score heads are updated.
The base model weights remain frozen.
PEFT `get_peft_model()` wraps the base model before training.

| Property | Value | Source |
|----------|-------|--------|
| Training script | Not in `src/` — presumed in `notebooks/Module2Project (1).ipynb` | evidence-ledger §7, §9 |
| LoRA rank | r=16 | evidence-ledger §7 |
| LoRA alpha | 32 | evidence-ledger §7 |
| LoRA dropout | 0.1 | evidence-ledger §7 |
| Target modules | `query_proj`, `key_proj`, `value_proj`, `o_proj` | evidence-ledger §7 |
| Modules saved fully | `classifier`, `score` | evidence-ledger §7 |
| Stage 1 batch size | 16 (from trainer_state.json) | evidence-ledger §3 |
| Stage 2 batch size | 16 (from trainer_state.json) | evidence-ledger §3 |
| Stage 1 best val accuracy | **98.67%** at epoch 2 | evidence-ledger §6 |
| Stage 2 best val accuracy | **96.07%** at epoch 8 | evidence-ledger §6 |
| Stage 3 LoRA | Not executed — no artifacts present | evidence-ledger §7 |
| Cross-dataset evaluation | None recorded — no Aegis result file for LoRA | evidence-ledger §8 |
| Checkpoint format | Adapter weights only (`adapter_model.safetensors`) | evidence-ledger §7 |
| PEFT version | 0.17.1 | evidence-ledger §2 |

### What both pipelines share

- Base model: `ProtectAI/deberta-v3-base-prompt-injection` (ADR-001)
- Dataset sequence: Safe-Guard → SPML → Aegis (ADR-003)
- Tokenisation: `max_length=256`, `padding="max_length"`, `truncation=True`
- Training metric: accuracy; best model selection via `load_best_model_at_end=True`
- Early stopping patience: 3 epochs

---

## Decision

Maintain both pipelines side-by-side (Option C). The reconstructed `src/` directory will
contain two named entry points and a shared module layer:

```
src/
├── config.py       # Shared — hyperparameters and paths for both pipelines
├── data.py         # Shared — dataset loading and preprocessing for all three stages
├── utils.py        # Shared — metric computation, plotting, seed setting
├── train.py        # Full fine-tuning entry point — orchestrates three-stage pipeline
├── train_lora.py   # LoRA entry point — PEFT adapter training (see open decisions below)
└── evaluate.py     # Post-training evaluation and results output
```

`src/finetune.py` and `src/finetune_2.py` are deprecated after Phase 3 completes and the
reconstructed modules are verified. They are not deleted until the new entry points are
confirmed correct.

### Rationale

The two pipelines are not redundant — they demonstrate different parameter-efficient
fine-tuning paradigms and their artifacts coexist in the repository:

- `models/deberta-pi-lora-*` directories contain confirmed LoRA adapter artifacts with no
  corresponding source script in `src/`. Maintaining `src/train_lora.py` resolves this
  mismatch: the LoRA artifacts become explicable and reproducible.
- The full fine-tuning pipeline is the only one with a confirmed cross-dataset evaluation
  result (81.16% on Aegis). `src/train.py` is the entry point that produced and can
  reproduce that result.
- The shared module layer (`config.py`, `data.py`, `utils.py`) demonstrates clean abstraction
  across two fine-tuning approaches — a stronger portfolio signal than either pipeline alone.
- Choosing this option forecloses the artifact mismatch that Option A would have left open:
  a reviewer examining `models/` under Option A would find LoRA adapter configs with no
  source code to explain them.

### Resolved decisions — LoRA pipeline

All three decisions gated on this ADR have been resolved following a read-only inspection
of `notebooks/Module2Project (1).ipynb`. `src/train_lora.py` is now fully unblocked for
Phase 4.

| Decision | Resolution |
|----------|-----------|
| **D2 — Stage 3 LoRA** | **2-stage only.** Stage 3 LoRA (Aegis) was never written in the notebook. Cell 16 is a planning comment with no code. Cell 17 is a commented-out, unexecuted block using `nvidia/llama-3.1-nemoguard-8b-content-safety` — a Llama-family generative model unrelated to this project's DeBERTa pipeline. Stage 3 LoRA is documented as absent from the original work and is **out of scope** for the reconstructed project unless explicitly added as new, separately identified work. See ADR-006. |
| **D3 — LoRA batch size** | **16.** Confirmed by both the notebook source (`per_device_train_batch_size=16` in cells 9 and 10) and the `trainer_state.json` artifacts (`train_batch_size: 16` in both stage 1 and stage 2 checkpoints). No ambiguity remains. |
| **D6 — Notebook disposition** | **Extract stages 1–2, retain notebook.** The LoRA training code in cells 9–10 is extracted into `src/train_lora.py`. The notebook is retained at its current path as a supplementary reference and execution-provenance artifact. One correction is required on extraction: `num_train_epochs` must be set to `10` (per `trainer_state.json` and notebook execution output), not `20` as the current notebook source shows. See ADR-006. |

**LoRA pipeline scope:** `src/train_lora.py` covers stages 1 and 2 only. This matches the
original notebook. The Aegis cross-dataset evaluation result (81.16% accuracy, 0.8255 F1)
belongs to the full fine-tuning pipeline and is not claimed by the LoRA pipeline.

---

## Alternatives Considered

### Option A — Full Fine-Tuning Only

Designate the full fine-tuning pipeline as the sole canonical approach. The LoRA artifacts
in `models/` are retained as historical reference only. The LoRA training script is not
extracted from the notebook into `src/`.

Not chosen because: the LoRA artifacts in `models/` would become orphaned — a reviewer
examining the repository would find adapter configurations with no corresponding source code.
Choosing A would have left a visible inconsistency between `src/` and `models/`.

### Option B — LoRA Only

Designate the LoRA pipeline as the sole canonical approach. The full fine-tuning scripts
are deprecated immediately. The LoRA training code is extracted from the notebook into
`src/train_lora.py`.

Not chosen because: the confirmed cross-dataset evaluation result (81.16% Aegis accuracy)
belongs to the full fine-tuning pipeline. Making LoRA the sole canonical approach would
leave the portfolio with no confirmed cross-dataset metric unless stage 3 LoRA is re-run
(requiring D2 approval). This option also discards the strongest quantitative result in
the repository.

---

## Consequences

### Repository structure

| Concern | Outcome |
|---------|---------|
| `src/train.py` | Full fine-tuning — three-stage pipeline |
| `src/train_lora.py` | LoRA — extracts notebook code; blocked on D2, D3, D6 |
| `src/finetune.py` / `src/finetune_2.py` | Deprecated after Phase 3 modules are verified |
| `models/deberta-pi-lora-*` | Explained by `src/train_lora.py` — artifact mismatch resolved |
| Cross-dataset result in `results/` | 81.16% full FT result; LoRA Aegis result only if D2 approved |
| Notebook | Source for LoRA script extraction (D6); retained after extraction |

### Evaluation results

The confirmed 81.16% accuracy and 0.8255 F1-score (Unsafe) on the Aegis test set belong
to the full fine-tuning pipeline and are the primary cross-dataset results for this project.
The LoRA pipeline's within-distribution results (98.67% stage 1, 96.07% stage 2) are
secondary and explicitly scoped to held-out splits of their respective training datasets.

No cross-dataset evaluation result exists for the LoRA pipeline. This limitation is
documented honestly in both model card READMEs and is subject to change if D2 is
approved and stage 3 LoRA training is executed.

### CI impact

Two entry points require two named integration smoke tests. The shared module layer
(`config.py`, `data.py`, `utils.py`) has a single set of unit tests that covers both
pipelines' shared logic.

---

## Review Trigger

Revisit if:
- A third training approach (e.g., QLoRA, adapter fusion) is added to the project.
- The LoRA stage 2 run is found to be definitively incomplete and must be re-run,
  changing the recorded metrics.
- The full fine-tuning scripts are superseded by a materially different implementation.
- D2 is approved and stage 3 LoRA training produces an Aegis cross-dataset result,
  at which point the LoRA pipeline's evaluation standing should be updated.

---

## Evidence Basis

| Claim | Evidence | Ledger ref |
|-------|----------|-----------|
| Full fine-tuning scripts in `src/` | File listing; `src/finetune.py` lines 88–91, 120–135, 229–244; `src/finetune_2.py` lines 139–154 | §3, §7 |
| Full FT batch size=8 | `src/finetune.py` line 120; `src/finetune_2.py` lines 139–140 | §3 |
| Stage 3 FT Aegis accuracy 81.16% | `results/test_results_3datasets.txt` line 9 | §8 |
| Stage 2 FT Aegis accuracy 41.60% | `results/test_results_2dataset.txt` line 9 | §8 |
| LoRA training script absent from `src/` | File listing — only `finetune.py`, `finetune_2.py`, `test_model.py` present | §7, §9 |
| LoRA adapter config (r, alpha, dropout, targets) | All four `adapter_config.json` files | §7 |
| LoRA batch size=16 | `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `train_batch_size` | §3 |
| Stage 1 LoRA val accuracy 98.67% at epoch 2 | `trainer_state.json` fields `best_metric`, `epoch` | §6 |
| Stage 2 LoRA val accuracy 96.07% at epoch 8 | `trainer_state.json` fields `best_metric`, `epoch` | §6 |
| Stage 3 LoRA artifacts absent | File listing — no `deberta-pi-lora-stage3` directory | §7 |
| Stage 2 LoRA run may be incomplete | `trainer_state.json` fields `num_train_epochs=10`, `epoch=8.0`, `should_training_stop: false` | §10 |
| PEFT version 0.17.1 | `models/deberta-pi-lora-stage1-final/README.md` line 205 | §2 |
