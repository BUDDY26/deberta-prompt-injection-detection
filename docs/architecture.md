# Architecture Overview

**Project:** deberta-prompt-injection-detection
**Last updated:** 2026-03-15
**Status:** Complete

> **Legend used in this document**
>
> - `[CONFIRMED]` — fact sourced directly from `docs/evidence-ledger.md`
>
> All open design decisions referenced in earlier drafts of this document have been
> resolved and recorded as ADRs in `docs/adr/`. Sections 7 and 8 below are preserved
> as historical record of the pre-implementation gap analysis and decision process.

---

## 1. Project Purpose

`[CONFIRMED]` Binary sequence classification: given a text prompt, predict whether it contains
a prompt injection attack.

- Label `0` — safe (benign input)
- Label `1` — unsafe (prompt injection)

Task type: `SEQ_CLS` (sequence classification).
Source: `docs/evidence-ledger.md` §1

The project demonstrates a multi-stage fine-tuning strategy: instead of training once on a
single dataset, the model is trained sequentially across three increasingly diverse safety
datasets. The rationale is that each stage broadens the model's coverage of injection patterns
without discarding learning from prior stages.

---

## 2. Confirmed Current Repository Structure

`[CONFIRMED]` The following structure was observed during the entry protocol scan.

```
deberta-prompt-injection-detection/
│
├── src/
│   ├── finetune.py        # Stages 1+2: full fine-tuning pipeline + metric plotting
│   ├── finetune_2.py      # Stage 3:   full fine-tuning pipeline + metric plotting
│   ├── test_model.py      # Post-training evaluation script
│   └── .gitkeep
│
├── notebooks/
│   └── Module2Project (1).ipynb   # Contents not inspected — likely contains LoRA pipeline
│
├── models/
│   ├── 1️⃣ README.md                          # Explains model artifacts are local-only
│   ├── deberta-pi-lora-stage1/
│   │   └── checkpoint-928/                  # Best checkpoint; epoch 2; acc 98.67%
│   ├── deberta-pi-lora-stage1-final/        # Final adapter after stage 1
│   ├── deberta-pi-lora-stage2/
│   │   └── checkpoint-7208/                 # Best checkpoint; epoch 8; acc 96.07%
│   ├── deberta-pi-lora-final-adapter/       # Final adapter (adapter weights only)
│   └── deberta-pi-lora-final-full/          # Final adapter (includes training_args.bin)
│
├── results/
│   ├── test_results_2dataset.txt   # Stage 2 model on Aegis: 41.60% accuracy
│   └── test_results_3datasets.txt  # Stage 3 model on Aegis: 81.16% accuracy
│
├── docs/
│   ├── evidence-ledger.md          # Authoritative confirmed-facts reference (complete)
│   ├── architecture.md             # This document
│   ├── adr/
│   │   └── ADR-001-template.md     # Unfilled template — no real ADRs yet
│   ├── qa/
│   │   └── qa-plan.md              # Partially filled template
│   └── runbooks/
│       └── operations.md           # Partially filled template
│
├── tests/
│   ├── unit/                       # Empty — contains only .gitkeep
│   └── integration/                # Empty — contains only .gitkeep
│
├── scripts/
│   ├── bootstrap.sh
│   └── validate-structure.sh
│
├── .claude/
│   └── skills/                     # Entry protocol and other workflow skills
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI: lint → test → validate-structure → security
│
├── CLAUDE.md
├── README.md                       # Partially filled template
└── .env.example                    # Generic web-app vars — not ML-specific
```

Source: `docs/evidence-ledger.md` §7

---

## 3. Confirmed Training Pipeline Structure

`[CONFIRMED]` Two distinct training pipelines coexist in this repository and produced
separate model artifacts. They are not currently unified.

### Pipeline A — Full Fine-Tuning

Implemented in `src/finetune.py` (stages 1–2) and `src/finetune_2.py` (stage 3).
Outputs saved to paths named `deberta-pi-full-*` (not committed to the repository).

```
ProtectAI/deberta-v3-base-prompt-injection   ← base model (HuggingFace Hub)
          │
          ▼  Stage 1: xTRam1/safe-guard-prompt-injection
          │  LR=2e-5 | batch=8 | max_epochs=10 | early_stop_patience=3
          │  text col: "text" | label col: "label" (int)
          │  val: 10% of train (seed=42) | test: native split
          │
  deberta-pi-full-stage1-final               ← saved after stage 1
          │
          ▼  Stage 2: reshabhs/SPML_Chatbot_Prompt_Injection
          │  LR=2e-5 | batch=8 | max_epochs=15 | early_stop_patience=3
          │  text: "System Prompt" + " " + "User Prompt"
          │  label col: "Prompt injection" → renamed "label" (int)
          │  val: native if present, else 10% of train (seed=42)
          │  (in-memory continuation — no reload between stages)
          │
  deberta-pi-full-final                      ← saved after stage 2
          │
          ▼  Stage 3: nvidia/Aegis-AI-Content-Safety-Dataset-2.0
          │  LR=2e-5 | batch=8 | max_epochs=25 | early_stop_patience=3
          │  text col: "prompt" | label col: "prompt_label" (str: "unsafe"→1, else→0)
          │  val/test: native splits
          │
  deberta-pi-full-stage3-final               ← saved after stage 3
```

Source: `docs/evidence-ledger.md` §3

### Pipeline B — LoRA Fine-Tuning

Training script is **not present in `src/`**. Artifacts in `models/` confirm that this
pipeline was executed (likely from `notebooks/Module2Project (1).ipynb`).
All LoRA adapter configs share identical hyperparameters.

```
ProtectAI/deberta-v3-base-prompt-injection   ← same base model
          │
          ▼  Stage 1: (dataset presumed same — safe-guard-prompt-injection)
          │  LoRA: r=16 | alpha=32 | dropout=0.1
          │  target: query_proj, key_proj, value_proj, o_proj
          │  batch=16 | max_epochs=10 | early_stop_patience=3 | threshold=0.0
          │  Best: epoch 2 | step 928 | eval_accuracy=98.67%
          │
  models/deberta-pi-lora-stage1-final/       ← adapter weights saved
          │
          ▼  Stage 2: (dataset presumed same — SPML)
          │  LoRA: same adapter config as stage 1
          │  batch=16 | max_epochs=10 | early_stop_patience=3 | threshold=0.0
          │  Best: epoch 8 | step 7208 | eval_accuracy=96.07%
          │
  models/deberta-pi-lora-final-adapter/      ← adapter weights only
  models/deberta-pi-lora-final-full/         ← adapter + training_args.bin
```

**Note:** No LoRA checkpoint exists for stage 3 (Aegis dataset).
Source: `docs/evidence-ledger.md` §3, §6, §7

### Tokenization — confirmed for both pipelines

All stages use identical tokenization:

| Parameter | Value |
|-----------|-------|
| `truncation` | `True` |
| `padding` | `"max_length"` |
| `max_length` | `256` |

Source: `docs/evidence-ledger.md` §7

### Shared training settings — full fine-tuning (all three stages)

| Setting | Value |
|---------|-------|
| `eval_strategy` | `"epoch"` |
| `save_strategy` | `"epoch"` |
| `load_best_model_at_end` | `True` |
| `metric_for_best_model` | `"accuracy"` |
| `greater_is_better` | `True` |
| `save_total_limit` | `1` |
| `logging_steps` | `50` |
| `report_to` | `"none"` |
| `fp16` | `torch.cuda.is_available()` |
| `early_stopping_patience` | `3` |

Source: `docs/evidence-ledger.md` §3

---

## 4. Confirmed Artifact Flow

`[CONFIRMED]` The following input/output chain was derived from source code and model artifacts.

### Full fine-tuning — artifact chain

```
Stage 1 input:  ProtectAI/deberta-v3-base-prompt-injection (HuggingFace Hub)
Stage 1 output: deberta-pi-full-stage1-final/  (full model weights + tokenizer)

Stage 2 input:  in-memory continuation from stage 1 (no file reload)
Stage 2 output: deberta-pi-full-final/          (full model weights + tokenizer)

Stage 3 input:  deberta-pi-full-final/           (loaded from disk)
Stage 3 output: deberta-pi-full-stage3-final/    (full model weights + tokenizer)

Evaluation:     deberta-pi-full-stage3-final/ → src/test_model.py
                → results/test_results_3datasets.txt
```

Source: `docs/evidence-ledger.md` §3, §7

### LoRA — artifact chain

```
Stage 1 input:  ProtectAI/deberta-v3-base-prompt-injection
Stage 1 output: models/deberta-pi-lora-stage1-final/  (adapter weights only)
                models/deberta-pi-lora-stage1/checkpoint-928/  (full checkpoint)

Stage 2 input:  (presumed: stage 1 LoRA output — not confirmed from source code)
Stage 2 output: models/deberta-pi-lora-final-adapter/
                models/deberta-pi-lora-final-full/
                models/deberta-pi-lora-stage2/checkpoint-7208/

Stage 3:        No LoRA artifacts present for stage 3.
```

Source: `docs/evidence-ledger.md` §7

---

## 5. Repository Architecture

`[CONFIRMED]` The following structure reflects the repository as built across Phases 3–10.

### Guiding principles

1. **Single entry point.** `src/train.py` is the canonical full fine-tuning command; `src/train_lora.py` is the canonical LoRA command.
2. **Separation of concerns.** Configuration, data loading, training, evaluation, and inference are separate modules with clear interfaces.
3. **Reproducibility.** A global seed (`config.GLOBAL_SEED = 42`) is set once at startup; both training scripts accept `--seed` to override. Every training run co-locates a `run_config.json` alongside model artifacts.
4. **Traceability.** Every training run saves its hyperparameters and results in a consistent location.

### Source layout

```
src/
├── config.py       # All confirmed hyperparameters, dataset IDs, output paths
├── data.py         # Dataset loading and preprocessing (three stages)
├── utils.py        # compute_metrics, plot_training_metrics, set_global_seed, write_run_config
├── train.py        # Full fine-tuning entry point (three-stage sequential pipeline)
├── train_lora.py   # LoRA adapter training entry point (two-stage pipeline; ADR-002)
├── evaluate.py     # Parameterized post-training evaluation (Aegis test set; ADR-005)
├── inference.py    # Unified inference module and CLI (full FT + LoRA; auto-detects format)
├── finetune.py     # Original stages 1–2 script (retained per ADR-006)
├── finetune_2.py   # Original stage 3 script (retained per ADR-006)
└── test_model.py   # Original evaluation script (retained per ADR-006)
```

### Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/config.py` | All hyperparameters (LR, batch sizes, epoch counts, LoRA rank/alpha), dataset IDs, output directory roots, global seed. Single source of truth. |
| `src/data.py` | `load_stage1`, `load_stage2`, `load_stage3` — column normalization, tokenization, split logic. Returns tokenized train/val/test datasets. |
| `src/train.py` | Full fine-tuning entry point. Sets global seed, runs three sequential `Trainer` instances, saves per-stage models, writes `run_config.json`. |
| `src/train_lora.py` | LoRA adapter entry point. Wraps base model with PEFT, runs stages 1–2 sequentially in-memory, saves adapter weights and `run_config.json`. |
| `src/evaluate.py` | Standalone evaluation. Loads a full fine-tuned model from disk, runs batched inference on the Aegis test set, writes metrics to `results/evaluation/`. |
| `src/inference.py` | Unified inference. `load_model()` auto-detects full FT vs. LoRA adapter (via `adapter_config.json`). `predict()` / `predict_batch()` return label, `label_str`, probability, and raw softmax probabilities. CLI: `--model-path`, `--text`, `--output-format {text,json}`. |
| `src/utils.py` | `compute_metrics()`, `plot_training_metrics()`, `set_global_seed()`, `write_run_config()`. Shared by both training pipelines. |

### Test layout

```
tests/
├── conftest.py           # pytest path setup (flat src/ layout)
├── unit/
│   ├── test_config.py    # Hyperparameter constant assertions
│   ├── test_data.py      # Dataset loading and preprocessing (synthetic fixtures)
│   ├── test_inference.py # load_model, predict, predict_batch (fully mocked)
│   └── test_utils.py     # compute_metrics, set_global_seed
└── integration/          # Placeholder (see docs/qa/qa-plan.md)
```

### Results layout

```
results/
├── test_results_2dataset.txt    # Stage 2 model on Aegis: 41.60% accuracy (legacy)
├── test_results_3datasets.txt   # Stage 3 model on Aegis: 81.16% accuracy (legacy)
└── evaluation/                  # Canonical outputs from src/evaluate.py (created on first run)
```

---

## 6. Full Fine-Tuning vs. LoRA — Artifact Distinction

`[CONFIRMED]` Two sets of model artifacts exist with different characteristics.

| Property | Full Fine-Tuning | LoRA |
|----------|-----------------|------|
| Training script | `src/train.py` (canonical); `src/finetune.py`, `src/finetune_2.py` (original, retained per ADR-006) | `src/train_lora.py` (canonical); source notebook retained per ADR-006 |
| Adapter format | Full model weights | PEFT adapter (adapter_model.safetensors) |
| Base model frozen | No — all weights updated | Yes — only LoRA layers trained |
| LoRA rank | N/A | r=16 |
| LoRA alpha | N/A | 32 |
| LoRA dropout | N/A | 0.1 |
| Target modules | N/A | query_proj, key_proj, value_proj, o_proj |
| Modules saved fully | N/A | classifier, score |
| Train batch size | 8 | 16 |
| Stage 3 artifacts | `deberta-pi-full-stage3-final` (not in repo) | None |
| Stage 1 best eval accuracy | Not recorded | 98.67% |
| Stage 2 best eval accuracy | Not recorded | 96.07% |
| Stage 3 eval accuracy (Aegis) | 81.16% (confirmed result) | Not evaluated |
| PEFT version | N/A | 0.17.1 |

Source: `docs/evidence-ledger.md` §2, §3, §6, §8

Both pipelines are maintained side-by-side. See ADR-002 for the rationale.
`src/inference.py` supports both artifact formats — format is detected automatically via the
presence of `adapter_config.json` in the model directory.

---

## 7. Known Architecture Gaps

`[CONFIRMED]` The following gaps were identified during the entry protocol scan and
are recorded in `docs/evidence-ledger.md` §9 and §10.

| Gap | Severity | Source |
|-----|----------|--------|
| No `requirements.txt` — CI will fail on first run | Critical | evidence-ledger §9 |
| No `src/train.py` — the documented run command does not exist | Critical | evidence-ledger §9 |
| LoRA training script absent from `src/` — pipeline not reproducible from source | High | evidence-ledger §9 |
| Zero test files in `tests/unit/` and `tests/integration/` | High | evidence-ledger §9 |
| No global random seed set in any training script | High | evidence-ledger §10 |
| `src/test_model.py` saves to `test_results_2.txt` but file in repo is `test_results_2dataset.txt` | Medium | evidence-ledger §10 |
| Stage 3 not evaluated for LoRA pipeline | Medium | evidence-ledger §7 |
| Stage 2 LoRA run appears incomplete (8 of 10 epochs, `should_training_stop: false`) | Medium | evidence-ledger §10 |
| `docs/architecture.md` was empty — no design documentation | Medium | evidence-ledger §9 |
| All ADR slots are unfilled templates — no decisions recorded | Medium | evidence-ledger §9 |
| `.env.example` contains web-app variables unrelated to an ML project | Low | evidence-ledger §9 |
| No hardware specification recorded anywhere | Low | evidence-ledger §10 |
| Model card READMEs are auto-generated stubs — no fields filled in | Low | evidence-ledger §9 |

---

## 8. Open Design Decisions Requiring Approval

`[OPEN DECISION]` The following decisions must be made and recorded as ADRs before
implementation begins. An AI assistant must not resolve these unilaterally.

---

### Decision 1 — Canonical Training Approach

**Question:** Should the reconstructed project canonically implement full fine-tuning,
LoRA, or document both pipelines as separate, named options?

**Context:**
- Full fine-tuning is implemented in `src/` and produced the evaluated results (81.16% on Aegis).
- LoRA produced higher eval accuracy on stage 1 (98.67%) and stage 2 (96.07%), but its training
  script is absent from `src/` and no stage 3 LoRA results exist.
- Both pipelines share the same base model and dataset sequence.

**Options:**
- A) Full fine-tuning is canonical. LoRA pipeline is removed or moved to `notebooks/`.
- B) LoRA is canonical. Full fine-tuning scripts are retained as reference only.
- C) Both pipelines are maintained side-by-side, with explicit naming (`train_full.py`,
  `train_lora.py`).

**Impacts ADRs:** ADR-002 (training strategy)

---

### Decision 2 — Stage 3 LoRA

**Question:** If LoRA is chosen as canonical (Decision 1 option B or C), should stage 3
training be replicated with LoRA to produce a full three-stage LoRA pipeline?

**Context:** No stage 3 LoRA checkpoint exists in the repository. The stage 3 dataset
(Aegis) has native train/validation/test splits, and the full fine-tuning stage 3 result
shows a significant improvement (41.60% → 81.16% accuracy), which motivates running it.

---

### Decision 3 — Batch Size for LoRA Pipeline

**Question:** The full fine-tuning scripts use `per_device_train_batch_size=8` while
the LoRA `trainer_state.json` records `train_batch_size=16`. If LoRA is canonical,
which batch size should be documented and used going forward?

**Context:** The discrepancy is confirmed but the reason is not recorded. LoRA uses
fewer trainable parameters, which may justify a larger batch. Choosing 8 vs. 16 affects
training time and potentially results.

---

### Decision 4 — Module Structure for `src/`

**Question:** Should the reconstructed `src/` use a flat layout (one file per concern) or
a package layout with subdirectories (e.g., `src/pipeline/`, `src/models/`)?

**Context:** The existing code is three monolithic scripts. The flat layout proposed in
Section 5 is the minimal, lower-complexity option appropriate for a project of this size.
A package layout would be appropriate if the project grows substantially.

**Recommendation (not binding):** Flat layout. The project is a focused ML pipeline,
not a library. Premature packaging adds complexity without benefit.

---

### Decision 5 — Evaluation Scope

**Question:** Should the evaluation script test against all three datasets (stage-by-stage)
or only the final model against Aegis (current behavior)?

**Context:** The existing `results/` directory has only Aegis-test evaluations. Evaluating
each stage's model against a held-out split of its own dataset would provide a more complete
picture of per-stage learning, but requires more compute and storage.

---

### Decision 6 — Notebook Disposition

**Question:** What should happen to `notebooks/Module2Project (1).ipynb`?

**Options:**
- A) Inspect and extract the LoRA pipeline code into `src/` — then the notebook becomes
  secondary documentation.
- B) Retain the notebook as-is alongside new `src/` scripts.
- C) Convert the notebook to a clean tutorial notebook that demonstrates the final pipeline.

---

*Full system design lives in this document. Key technical decisions will be recorded in `docs/adr/`.*
*Last updated by Claude: 2026-03-15*
