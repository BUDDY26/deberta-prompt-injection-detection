# ADR-006: Notebook Disposition — Extract, Retain, or Convert

**Date:** 2026-03-14
**Status:** Accepted
**Author:** BUDDY26

---

## Context

The repository contains one Jupyter notebook:

```
notebooks/Module2Project (1).ipynb
```

This notebook was inspected in read-only mode. Its structure is as follows:

| Cell(s) | Content | Executed? |
|---------|---------|-----------|
| 0–7 | Preliminary exploration: distilbert pipeline, IMDB, SST2, pip installs | Yes |
| 8 | Empty markdown | — |
| 9 | Stage 1 LoRA training — Safe-Guard dataset, full training code | Yes, with output |
| 10 | Stage 2 LoRA training — SPML dataset + both model saves | Yes, with output |
| 11 | `!zip -r trained-models.zip trained-models` — Colab model export | Yes |
| 12–15 | Markdown note, duplicate comment cells | — |
| 16 | Planning comment for "Part #3 Nvidia" (Aegis) — no code written | Not executed |
| 17 | Commented-out Llama-3.1-NemoGuard code (different architecture) | Not executed |
| 18 | Empty | — |

### Key facts confirmed by inspection

- The notebook is the origin of the LoRA adapter artifacts in `models/`.
  Cell 10 explicitly saves to `"deberta-pi-lora-final-adapter"` (via `model.save_pretrained()`)
  and `"deberta-pi-lora-final-full"` (via `trainer2.save_model()`), matching the repository
  directory names exactly.

- The notebook contains LoRA training code for stages 1 and 2 only. Stage 3 (Aegis) LoRA
  was never coded. Cell 16 is a planning comment; cell 17 is a commented-out and unexecuted
  block using `nvidia/llama-3.1-nemoguard-8b-content-safety` — a generative Llama-family model
  with a different architecture and task formulation entirely unrelated to this project's
  DeBERTa LoRA pipeline.

- One confirmed discrepancy exists between the notebook source and the artifacts: the notebook
  currently shows `num_train_epochs=20` for both stages, but the `trainer_state.json` artifacts
  record `num_train_epochs: 10` and the notebook's own execution output shows `epoch: 10.0` at
  the final stage 2 evaluation. The notebook was edited after the training run; the artifacts
  reflect the original value of 10.

- The stage 2 run previously flagged as potentially incomplete (evidence-ledger §10) is now
  explained: training reached the `num_train_epochs=10` epoch limit at epoch 10. Early stopping
  patience=3 would have required epoch 11 to trigger (best at epoch 8 + 3 more), but the epoch
  limit was reached first. The run completed normally.

---

## Decision

**Extract** the LoRA training code for stages 1 and 2 from notebook cells 9–10 into
`src/train_lora.py`. **Retain** the notebook as a supplementary reference artifact — it
is not deleted, not converted to a script independently, and not treated as authoritative
after extraction.

`src/train_lora.py` becomes the canonical source for the LoRA pipeline after extraction.
The notebook is demoted to historical evidence of how the training was originally run.

### What is extracted

- Stage 1 LoRA training (notebook cell 9): dataset loading, tokenization, LoRA config,
  `TrainingArguments`, `Trainer`, early stopping, test evaluation, model save
- Stage 2 LoRA training (notebook cell 10): dataset loading, SPML preprocessing, continuing
  training from the stage 1 model, test evaluation, both final model saves
- Shared: `compute_metrics`, tokenizer and base model initialisation, `LoraConfig`

### What is NOT extracted

- Cells 0–7: preliminary exploration code (distilbert, IMDB, SST2) — unrelated to the
  DeBERTa LoRA pipeline
- Cell 11: Colab-specific zip command — not applicable outside Google Colab
- Cell 16: Planning comment — no code to extract
- Cell 17: Commented-out Llama-3.1-NemoGuard block — different model, different task
  formulation, never executed; not part of this project

### Required correction on extraction

`num_train_epochs` must be set to `10` in `src/train_lora.py`, not `20` as it currently
reads in the notebook source. The value of `10` is confirmed by:
- `trainer_state.json` for both stage 1 and stage 2 (`num_train_epochs: 10`)
- The notebook's own execution output (`epoch: 10.0` at final stage 2 evaluation)

The notebook's current value of `20` reflects a post-training edit that was never
re-executed. Extracting `20` would not reproduce the artifacts in `models/`.

### Notebook retention rationale

The notebook provides execution provenance — it shows that training actually ran and
completed, with visible cell outputs including loss curves (rendered as `<IPython.core.display.HTML object>`),
final metrics, and save confirmations. This provenance is meaningful for a portfolio
reviewer and should not be destroyed. The notebook is retained at its current path:

```
notebooks/Module2Project (1).ipynb
```

It is documented in the README and in `docs/architecture.md` as a supplementary reference.
It is not referenced as the source of truth for any parameter value — `src/train_lora.py`
is authoritative after extraction.

---

## Relationship to D2 (Stage 3 LoRA)

This ADR confirms that stage 3 LoRA was never written in the notebook. There is no code
to extract for stage 3. The decision recorded in ADR-002 (D2 resolution) is:

> The LoRA pipeline in `src/train_lora.py` covers stages 1 and 2 only, matching the
> original notebook. Stage 3 LoRA is absent from the original work and is out of scope
> for the reconstructed project unless explicitly added as a new, separate contribution.

This boundary is important for portfolio honesty: the Aegis cross-dataset evaluation result
(81.16% accuracy, F1 0.8255) belongs to the full fine-tuning pipeline only. `src/train_lora.py`
does not claim to reproduce or extend to that result.

---

## Consequences

### Positive

- `src/train_lora.py` gives the LoRA adapter artifacts in `models/` a reproducible source.
  A reviewer can read the script and understand exactly how `deberta-pi-lora-final-adapter`
  and `deberta-pi-lora-final-full` were produced.
- The notebook is preserved as execution evidence — its cell outputs confirm that training ran
  and completed.
- The parameter correction (`num_train_epochs: 20 → 10`) is made explicit and documented,
  demonstrating careful reconstruction rather than blind copy-paste.

### Negative

- The notebook and `src/train_lora.py` will diverge slightly (the `num_train_epochs`
  correction). This is unavoidable and must be documented in `src/train_lora.py` as a
  comment noting the discrepancy.
- The exploratory cells (0–7) in the notebook — distilbert, IMDB, SST2 — are visible to
  any reader and reflect a course-assignment origin. These are not removed (notebook is
  read-only after the decision); they provide honest context about how the project developed.

### Stage 3 LoRA — honest scope boundary

`src/train_lora.py` does not include a stage 3 block. The absence of stage 3 LoRA is
documented in the script itself (as a comment noting it was not part of the original
training run) and in model card READMEs for `deberta-pi-lora-final-adapter` and
`deberta-pi-lora-final-full`. Any future addition of stage 3 LoRA would constitute new
work and should be branched or flagged as an extension — not presented as part of the
original experiment.

---

## Review Trigger

Revisit if:
- Stage 3 LoRA is explicitly approved as new work (would require a separate decision and
  a new evidence record once training has actually run).
- The notebook is removed or relocated; `src/train_lora.py` remains authoritative regardless,
  but documentation references would need updating.
- The Colab-specific cell (cell 11, zip command) creates confusion for contributors; it can
  be noted in the README as a Colab artifact.

---

## Evidence Basis

| Claim | Evidence | Source |
|-------|----------|--------|
| Notebook is origin of LoRA adapter artifacts | Cell 10 saves to `deberta-pi-lora-final-adapter` and `deberta-pi-lora-final-full` — matching `models/` directory names | Notebook cell 10 execution output |
| Stage 3 LoRA code never written | Cell 16 is a planning comment; cell 17 is commented-out Llama code, not executed | Notebook cells 16–17 |
| `num_train_epochs` discrepancy: notebook shows 20, artifacts show 10 | `trainer_state.json` both stages field `num_train_epochs`; notebook execution output `epoch: 10.0` at final stage 2 evaluation | Notebook cell 10 output; `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` |
| Stage 2 run was not interrupted — stopped at epoch limit | Notebook execution output `epoch: 10.0`; early stopping patience=3 would require epoch 11, but epoch limit was 10 | Notebook cell 10 execution output |
| Final save methods | `model.save_pretrained("deberta-pi-lora-final-adapter")`; `trainer2.save_model("deberta-pi-lora-final-full")` | Notebook cell 10 |
| Cell 17 is Llama-3.1-NemoGuard, not DeBERTa LoRA | `AutoModelForCausalLM`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, `nvidia/llama-3.1-nemoguard-8b-content-safety` — different model family | Notebook cell 17 (commented) |
