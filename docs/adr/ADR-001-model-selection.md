# ADR-001: Use ProtectAI/deberta-v3-base-prompt-injection as the Base Model

**Date:** 2026-03-14
**Status:** Accepted
**Author:** BUDDY26

---

## Context

The project requires a pretrained transformer encoder capable of binary sequence
classification — given a text prompt, predict whether it contains a prompt injection
attack (label `1`) or is benign (label `0`).

Two categories of starting point were available:

1. **General-purpose pretrained encoders** — models trained on large general corpora
   (e.g., masked language modelling on web text) with no domain-specific pretraining
   on safety or adversarial prompt data.

2. **Domain-specific pretrained classifiers** — models that have already been trained
   or fine-tuned on data related to the target task before this project begins.

The key question: should the project start from a general-purpose base and train the full
classification objective from scratch, or start from a model that already carries
domain-relevant weights?

---

## Decision

Use `ProtectAI/deberta-v3-base-prompt-injection` as the base model for all three
training stages in both the full fine-tuning and LoRA pipelines.

---

## Rationale

### 1. Domain-specific initialisation

The model identifier `deberta-v3-base-prompt-injection` indicates that prior training on
prompt injection data has already occurred before this project's fine-tuning begins.
Starting from domain-specific weights rather than a general-purpose checkpoint reduces the
distance between the model's initialisation state and the target task. In practice, this
means fewer epochs are needed to reach a useful decision boundary, and the model is less
likely to waste capacity learning that prompt injection is a meaningful category at all —
it already knows this.

This property is evidenced in the LoRA stage 1 training log: the model achieved
**98.67% validation accuracy at epoch 2** out of a maximum of 10 configured epochs,
triggering early stopping at step 928. Starting from a cold general-purpose base would
be unlikely to converge this quickly on a domain-specific binary task.

### 2. Consistent use across both pipeline variants

Both the full fine-tuning pipeline (`src/finetune.py`, `src/finetune_2.py`) and all four
LoRA adapter configs confirm `ProtectAI/deberta-v3-base-prompt-injection` as the base.
This consistency across independently executed training runs confirms the choice was
deliberate and reproduced, not incidental.

### 3. Architectural compatibility with both fine-tuning approaches

The model's attention mechanism uses named projections (`query_proj`, `key_proj`,
`value_proj`, `o_proj`) that are directly addressable as LoRA target modules.
The `classifier` and `score` heads are preserved as fully fine-tuned modules
(`modules_to_save`). This means the same base model supports both full fine-tuning
and parameter-efficient LoRA adaptation without requiring architectural changes.

### 4. Native sequence classification support

All four adapter configs declare `task_type: SEQ_CLS`, confirming the model architecture
natively supports the sequence classification head required for binary output.
`AutoModelForSequenceClassification` loads it without modification.

---

## Alternatives Considered

No alternative base models are documented anywhere in this repository. Comparison to
other model families (general-purpose BERT, RoBERTa, DeBERTa-v3-base without prompt
injection pretraining, other ProtectAI models, generative models) is **out of scope
for this ADR** — the repository evidence does not include ablation studies or model
comparisons. Any such comparison would be speculative and is not recorded here.

| Alternative | Status |
|-------------|--------|
| Other base models | Not evaluated in this project — out of scope |

---

## Consequences

### Positive

- Domain-specific initialisation gives the model a head start on the classification task,
  as confirmed by rapid convergence in stage 1 LoRA training (98.67% at epoch 2).
- A single base model is shared across both pipeline variants, making cross-pipeline result
  comparisons meaningful.
- The attention projection naming (`query_proj` etc.) is compatible with PEFT LoRA without
  any custom configuration.
- `AutoModelForSequenceClassification` loads the model directly — no architectural
  customisation is required.

### Negative

- The model is published by a third-party organisation (ProtectAI) on HuggingFace Hub.
  Its training data, exact pretraining procedure, and architectural specifics beyond
  what is disclosed publicly are not documented in this repository.
- Full reproducibility requires the model to remain available at
  `ProtectAI/deberta-v3-base-prompt-injection` on HuggingFace Hub.

### Risks

- If ProtectAI removes, updates, or changes the model weights on HuggingFace Hub,
  restarting a training run will load different initialisation weights, producing
  results that cannot be directly compared to those recorded in `results/`.
- The model's existing prompt injection pretraining data is unknown. If it overlaps
  significantly with one of the three fine-tuning datasets, stage 1 validation accuracy
  (98.67%) may be inflated by data leakage from pretraining rather than representing
  genuine generalisation.

---

## Review Trigger

Revisit this decision if:

- `ProtectAI/deberta-v3-base-prompt-injection` becomes unavailable on HuggingFace Hub.
- A substantially different model family (encoder-decoder, decoder-only) is being
  evaluated for this task.
- Evidence of pretraining data overlap with the fine-tuning datasets is discovered,
  making the stage 1 results unreliable.

---

## Evidence Basis

All facts in this ADR are sourced from `docs/evidence-ledger.md`.

| Claim | Evidence | Ledger ref |
|-------|----------|-----------|
| Base model identifier | `src/finetune.py` lines 88–91 | §2 |
| Base model confirmed in LoRA pipeline | `models/deberta-pi-lora-stage1/checkpoint-928/adapter_config.json` field `base_model_name_or_path`; same in all four adapter configs | §2 |
| Task type SEQ_CLS | All four `adapter_config.json` files field `task_type` | §7 |
| LoRA target modules | All four `adapter_config.json` files field `target_modules` | §7 |
| Modules to save | All four `adapter_config.json` files field `modules_to_save` | §7 |
| Stage 1 LoRA best accuracy 98.67% at epoch 2 | `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` fields `best_metric`, `epoch` | §6 |
| Stage 1 LoRA early stopping at step 928 of max 4640 | `trainer_state.json` fields `best_global_step`, `max_steps` | §6 |
