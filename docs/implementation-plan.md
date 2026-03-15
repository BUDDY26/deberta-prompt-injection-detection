# Implementation Plan

**Project:** deberta-prompt-injection-detection
**Created:** 2026-03-14
**Status:** Pre-implementation — awaiting open decision approvals

> **Authority:** This document is the authoritative coding guide for this repository.
> It is read-only during coding passes. An AI assistant must not edit it while implementing —
> not to mark progress, not to add notes, not to correct phrasing.
>
> **Progress reporting:** Report progress in responses, not by editing this file.
> Example: "Completed: `src/config.py`, `src/data.py`. Next: `src/train.py`."
>
> **Conflict protocol:** If a true implementation conflict is discovered, stop, report
> the conflict, propose a minimal change, and wait for explicit approval before modifying
> this document or the code.

---

## 1. Reconstruction Goals

This plan reconstructs a portfolio-grade ML repository from existing experimental artifacts.
Training has already been completed. The goal is not to retrain — it is to make the existing
work reproducible, documented, testable, and readable by graduate admissions reviewers and
software engineering employers.

### Success criteria

| Criterion | Definition |
|-----------|-----------|
| Reproducible | A new engineer can clone the repo, install dependencies, and run the full training pipeline from a single command |
| Documented | Architecture, key decisions, and dataset handling are explained in prose, not only in code |
| Tested | Core data preprocessing and metric logic have unit test coverage |
| Portfolio-ready | Professional commit history, complete README, filled model cards, no unfilled placeholders |
| CI green | All four CI jobs (lint, test, validate-structure, security) pass on `main` |

### What this plan does NOT include

- Retraining the model (training is complete; results are in `results/`)
- Hyperparameter search or ablation studies
- Deployment infrastructure (API serving, containerization)
- Dataset curation or augmentation

---

## 2. Prerequisite: Open Decisions

The following decisions from `docs/architecture.md` Section 8 must be approved before
implementation phases begin. Each decision gates the phases listed.

| Decision | Question | Gates |
|----------|----------|-------|
| D1 — Canonical training approach | Full fine-tuning, LoRA, or both? | Phase 3, 4 |
| D2 — Stage 3 LoRA | Should stage 3 be replicated with LoRA? | Phase 4 |
| D3 — LoRA batch size | batch=8 or batch=16 for canonical LoRA config? | Phase 3, 4 |
| D4 — Module structure | Flat `src/` layout or package layout? | Phase 3 |
| D5 — Evaluation scope | Evaluate all stages, or only final model vs. Aegis? | Phase 5 |
| D6 — Notebook disposition | Extract, retain, or convert notebook? | Phase 6 |

**Do not begin Phase 3 or later until D1 and D4 are approved.**
Phases 1 and 2 are fully independent of these decisions and can begin immediately.

---

## 3. Phase Overview

```
Phase 0  Foundation            — requirements.txt, .env.example, placeholder fixes
Phase 1  ADR Layer             — record the six open decisions as binding ADRs
Phase 2  Documentation         — README, runbook, QA plan, model cards
Phase 3  Source Restructure    — modularize src/ per approved structure (needs D1, D4)
Phase 4  LoRA Script           — extract LoRA pipeline to src/ (needs D1, D2, D3)
Phase 5  Evaluation Module     — clean evaluate.py (needs D5)
Phase 6  Test Suite            — unit and integration tests
Phase 7  QA and Polish         — CI validation, final placeholder sweep, commit history
```

Phases 0–2 are documentation-only and can proceed without code changes.
Phases 3–5 require approval of the relevant open decisions.
Phase 6 requires Phase 3 and 5 to be complete.
Phase 7 is last and depends on all prior phases.

---

## 4. Phase 0 — Foundation

**Goal:** Make the repository minimally functional as a Python project. Fix the two critical
blockers that prevent CI from running at all.

**Depends on:** Nothing. Can begin immediately.

**Permission tier:** Allowed without asking (CLAUDE.md §3).

### Deliverables

#### 0.1 — `requirements.txt`

Create `requirements.txt` at the repository root with pinned versions for all libraries
confirmed in `docs/evidence-ledger.md` §2.

Libraries to include (versions to be determined by inspecting the installed environment
or pinning to known-compatible versions):

- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `peft` — confirmed version: `0.17.1`
- `scikit-learn`
- `matplotlib`
- `numpy`
- `tqdm`
- `accelerate` (required by HuggingFace Trainer for fp16)

Testing and dev dependencies (in a separate `requirements-dev.txt` or `[dev]` section):
- `pytest`
- `pytest-cov`
- `ruff`
- `black`
- `bandit`

**Acceptance criteria:** `pip install -r requirements.txt` completes without error on a
clean Python 3.11 environment. The CI `test` job no longer fails on the install step.

#### 0.2 — `.env.example` update

Replace the generic web-application variables with ML-relevant variables for this project:

- `HF_HOME` — HuggingFace cache directory
- `CUDA_VISIBLE_DEVICES` — GPU selection
- `MODEL_OUTPUT_DIR` — root directory for saved model checkpoints
- `RESULTS_DIR` — root directory for evaluation output files

Remove: `APP_ENV`, `APP_PORT`, `SECRET_KEY`, `DATABASE_URL`.

**Acceptance criteria:** Every variable in `.env.example` is relevant to the ML pipeline.
No web-application variables remain.

#### 0.3 — Fix unfilled placeholder tokens

Remove or replace the `{{LINT_COMMAND}}{{LINT_COMMAND}}` placeholder that appears in:
- `README.md` line 50
- `docs/runbooks/operations.md` line 58

Replace with the correct linting command: `ruff check src/ && black --check src/`

**Acceptance criteria:** `scripts/validate-structure.sh --strict` finds no remaining
`{{PLACEHOLDER}}` tokens in `README.md` or `docs/runbooks/operations.md`.

---

## 5. Phase 1 — ADR Layer

**Goal:** Record all six open design decisions as binding ADRs before any code changes.
This preserves the rationale behind every structural choice for the portfolio audience.

**Depends on:** Explicit approval of each decision (from owner, not AI assistant).

**Permission tier:** Allowed without asking for documentation improvements (CLAUDE.md §3).
However, the AI assistant must not write ADRs with invented rationale — decisions must
be approved by the user before the ADR is finalized.

**Workflow for each ADR:**
1. AI drafts ADR with options and consequences populated.
2. User selects option and approves.
3. AI writes the final ADR with "Status: Accepted".
4. This plan is updated only if the decision changes the phase structure.

### Deliverables

| File | Decision captured |
|------|------------------|
| `docs/adr/ADR-001-model-selection.md` | Why `ProtectAI/deberta-v3-base-prompt-injection` over other base models |
| `docs/adr/ADR-002-training-strategy.md` | D1 — Full fine-tuning vs. LoRA vs. both |
| `docs/adr/ADR-003-dataset-ordering.md` | Why stage 1→2→3 ordering; catastrophic forgetting tradeoffs |
| `docs/adr/ADR-004-module-structure.md` | D4 — Flat vs. package layout for `src/` |
| `docs/adr/ADR-005-evaluation-scope.md` | D5 — Which datasets to evaluate and at which stages |
| `docs/adr/ADR-006-notebook-disposition.md` | D6 — What to do with the notebook |

**Acceptance criteria:** Six ADR files exist, each with Status set to "Accepted" or
"Rejected" (not "Proposed"). ADR-001 and ADR-003 can be drafted and accepted
independently of the open decisions (their answers are already known from the evidence
ledger). ADR-002, 004, 005, 006 require user approval of the corresponding decision.

---

## 6. Phase 2 — Documentation

**Goal:** Complete all human-readable documentation files. The portfolio audience
(graduate admissions, employers) reads documentation before they read code.

**Depends on:** Phase 0 complete (placeholder fix), Phase 1 ADRs accepted (for accuracy).

**Permission tier:** Allowed without asking (CLAUDE.md §3).

### Deliverables

#### 2.1 — `README.md`

Replace the placeholder `README.md` with a complete project overview. Required sections:

- **Overview** (2–3 paragraphs): what the project does, why multi-stage fine-tuning, key result
- **Key Results table**: Stage 2 vs. Stage 3 accuracy on Aegis test set (confirmed numbers)
- **Repository structure**: current layout (tree)
- **Prerequisites**: Python 3.11, CUDA-capable GPU recommended
- **Installation**: `pip install -r requirements.txt`
- **Usage**: `python src/train.py` (after Phase 3 creates this file)
- **Evaluation**: `python src/evaluate.py --model <path> --dataset aegis`
- **Architecture**: 2-sentence summary, link to `docs/architecture.md`
- **Key decisions**: link to `docs/adr/`
- **License**: MIT

**Confirmed data to include:**
- Stage 2 → Stage 3 accuracy improvement: 41.60% → 81.16% on Aegis test set (N=1964)
- F1-score improvement: 0.3053 → 0.8255

Source: `docs/evidence-ledger.md` §8

#### 2.2 — `docs/runbooks/operations.md`

Complete the operations runbook with ML-specific content:

- **Prerequisites**: Python 3.11, pip, CUDA GPU (recommended), HuggingFace account for
  gated datasets (if applicable)
- **Environment setup**: clone → install → `.env` configuration
- **Training commands**: updated to reference `src/train.py` (after Phase 3)
- **Evaluation commands**: updated to reference `src/evaluate.py` (after Phase 5)
- **Troubleshooting**: GPU OOM (reduce batch size), dataset download failures, checkpoint
  recovery
- **Environment variables**: replace web-app vars with ML vars from Phase 0.2

#### 2.3 — `docs/qa/qa-plan.md`

Complete the QA plan with ML-specific test strategy:

- **Test strategy**: unit tests for data logic + integration smoke test for pipeline
- **Test file inventory**: populate after Phase 6 creates test files
- **Known gaps**: no end-to-end retraining test (too expensive for CI)
- **Coverage policy**: retain 80% threshold from template

#### 2.4 — Model card READMEs

Fill in the auto-generated HuggingFace model cards for the two final model directories:

- `models/deberta-pi-lora-final-adapter/README.md`
- `models/deberta-pi-lora-final-full/README.md`

Required fields:
- Model description: task, base model, fine-tuning approach
- Training data: three dataset names with HuggingFace IDs
- Training procedure: LoRA config (r=16, alpha=32, dropout=0.1, target modules)
- Evaluation results: confirmed metrics from `docs/evidence-ledger.md` §8
- PEFT version: 0.17.1

**Acceptance criteria:** No "More Information Needed" fields remain in the model cards
for the two final models.

---

## 7. Phase 3 — Source Code Restructure

**Goal:** Reorganize `src/` from three monolithic scripts into the modular layout
approved in Decision D4, implementing the approach approved in Decision D1.

**Depends on:** Phase 0 complete; Decisions D1 (canonical approach) and D4 (module
structure) approved.

**Permission tier:** Requires explicit approval (changing function signatures and creating
new files — CLAUDE.md §3).

> **Governance note:** Before any file is renamed, moved, or deleted under this phase,
> the AI assistant must confirm that approval covers the full scope of changes in this phase,
> not just one file at a time.

### Deliverables

#### 3.1 — `src/config.py`

Centralized configuration for all hyperparameters and paths. No magic numbers in other
modules. Contains:

- Training hyperparameters per stage (LR, batch sizes, epoch limits, patience) — values
  sourced from `docs/evidence-ledger.md` §3
- Dataset identifiers (HuggingFace IDs) — sourced from evidence ledger §4
- Tokenization parameters (max_length=256, padding, truncation) — sourced from evidence ledger §7
- Output directory roots (configurable via environment variables from Phase 0.2)
- Global random seed (to be set at training startup)

#### 3.2 — `src/data.py`

Dataset loading and preprocessing, one function per stage. Encapsulates all column
normalization and split logic currently spread across `finetune.py` and `finetune_2.py`.

Functions:
- `load_stage1_dataset(tokenizer, config)` — safe-guard dataset
- `load_stage2_dataset(tokenizer, config)` — SPML dataset
- `load_stage3_dataset(tokenizer, config)` — Aegis dataset
- `tokenize_batch(batch, tokenizer, text_field, max_length)` — shared tokenization

Column handling is sourced directly from `docs/evidence-ledger.md` §3 and §4.

#### 3.3 — `src/utils.py`

Shared utilities extracted from the existing scripts:

- `plot_training_metrics(trainer, dataset_name, stage_num, output_dir)` — already in
  both scripts; consolidate into one place
- `set_global_seed(seed)` — sets Python, NumPy, and PyTorch seeds
- `save_text_results(content, output_path)` — consistent results file writer
- `compute_metrics(eval_pred)` — the HuggingFace accuracy metric function

#### 3.4 — `src/train.py`

Main entry point. Orchestrates the full three-stage pipeline sequentially.

Structure:
1. Parse arguments / load config
2. Set global seed
3. Load base model and tokenizer
4. Stage 1: load data → build Trainer → train → save → evaluate
5. Stage 2: reload or continue in-memory → load data → build Trainer → train → save → evaluate
6. Stage 3: load stage 2 output → load data → build Trainer → train → save → evaluate
7. Print summary of all stage results

**The in-memory continuation behavior between stages 1 and 2** — confirmed in
evidence ledger §3 — must be preserved exactly as-is unless changed by an approved decision.

#### 3.5 — Deprecate `src/finetune.py` and `src/finetune_2.py`

Once `src/train.py` is complete and verified, the original scripts are candidates for
removal. **Do not delete them until `src/train.py` produces equivalent results and the
user explicitly approves deletion.** They should not be deleted in the same commit that
creates `src/train.py`.

**Acceptance criteria for Phase 3:**
- `python src/train.py --help` runs without error
- All hyperparameters from evidence ledger §3 are present in `src/config.py`
- All dataset column mappings from evidence ledger §4 are in `src/data.py`
- `ruff check src/` passes with no errors
- `black --check src/` passes with no errors

---

## 8. Phase 4 — LoRA Script (Conditional)

**Goal:** Extract the LoRA training pipeline from the notebook into `src/` so that
the LoRA approach is reproducible from source code.

**Depends on:** Phase 3 complete; Decisions D1 (LoRA chosen as canonical or both),
D2 (stage 3 LoRA decision), D3 (batch size).

**Condition:** This phase is **only executed if Decision D1 selects LoRA or both pipelines.**
If Decision D1 selects full fine-tuning only, skip this phase.

**Permission tier:** Requires explicit approval (new file, changing entry points — CLAUDE.md §3).

### Deliverables

#### 4.1 — Inspect `notebooks/Module2Project (1).ipynb`

Before writing any code, read the notebook to confirm:
- Dataset sequence used in the LoRA run
- Exact hyperparameters used (particularly batch size — D3)
- Whether stage 3 was attempted in the notebook

This inspection is read-only and can be done without approval.

#### 4.2 — `src/train_lora.py`

LoRA training entry point. Structure mirrors `src/train.py` but uses PEFT:
- Initializes base model, applies `get_peft_model()` with LoRA config from evidence ledger §7
- LoRA config: r=16, alpha=32, dropout=0.1, target_modules per evidence ledger
- Runs stages per D2 decision (2 or 3 stages)
- Uses batch size per D3 decision

Shares `src/config.py`, `src/data.py`, `src/utils.py` with the full fine-tuning pipeline.

**Acceptance criteria:** `python src/train_lora.py --help` runs without error. Config
matches all LoRA adapter config values confirmed in evidence ledger §7.

---

## 9. Phase 5 — Evaluation Module

**Goal:** Replace `src/test_model.py` with a clean `src/evaluate.py` that fixes the
confirmed file path bug and implements the evaluation scope approved in Decision D5.

**Depends on:** Phase 3 complete; Decision D5 approved.

**Permission tier:** Requires approval if D5 expands scope beyond the current single-dataset
evaluation (new behavior). If D5 keeps current scope, this is allowed without asking.

### Deliverables

#### 5.1 — `src/evaluate.py`

Standalone evaluation script. Improvements over `src/test_model.py`:

- Accepts `--model-path` and `--dataset` as command-line arguments (no hardcoded paths)
- Saves results to `results/evaluation/<model_name>_<dataset>_results.txt` using a
  consistent, parameterized path — fixing the confirmed path mismatch in evidence ledger §10
- Reports identical metrics to existing script: accuracy, binary precision/recall/F1,
  per-class metrics, confusion matrix, classification report
- Shared `compute_metrics` and `save_text_results` from `src/utils.py`

If D5 approves multi-dataset evaluation:
- Loop over specified datasets, report per-dataset results
- Produce a summary table across all evaluated datasets

**Acceptance criteria:**
- `python src/evaluate.py --model-path models/deberta-pi-lora-final-full --dataset aegis`
  produces a result file matching the numbers in `results/test_results_3datasets.txt`
- Output file path is deterministic and under `results/evaluation/`
- No hardcoded model paths remain

---

## 10. Phase 6 — Test Suite

**Goal:** Bring `tests/unit/` from empty to substantive coverage of the core data and
metric logic. Add one integration smoke test.

**Depends on:** Phase 3 (`src/data.py`, `src/utils.py`, `src/config.py`) and
Phase 5 (`src/evaluate.py`) complete.

**Permission tier:** Allowed without asking (adding test files — CLAUDE.md §3).

### Deliverables

#### 6.1 — `tests/unit/test_data.py`

Tests for `src/data.py`. No HuggingFace network calls — use synthetic fixtures.

Required test cases:
- Stage 1: `"text"` column passed through tokenizer correctly; `"label"` column preserved
- Stage 2: `"System Prompt"` + `" "` + `"User Prompt"` concatenation produces expected string
- Stage 2: fallback column selection logic when only one prompt field is present
- Stage 2: `"Prompt injection"` column renamed to `"label"` before tokenization
- Stage 3: `"prompt_label"` string `"unsafe"` maps to `1`; all other values map to `0`
- Tokenization: output always has `input_ids`, `attention_mask`, `labels` keys
- Tokenization: `max_length=256` is respected

#### 6.2 — `tests/unit/test_utils.py`

Tests for `src/utils.py`:
- `compute_metrics`: all-correct predictions → accuracy 1.0
- `compute_metrics`: all-wrong predictions → accuracy 0.0
- `compute_metrics`: mixed predictions → expected fractional accuracy
- `set_global_seed`: calling with same seed twice produces deterministic results

#### 6.3 — `tests/unit/test_config.py`

Tests for `src/config.py`:
- All required hyperparameter keys exist
- Learning rate is `2e-5` for all stages (confirmed value)
- `max_length` is `256` (confirmed value)
- `early_stopping_patience` is `3` for all stages (confirmed value)

#### 6.4 — `tests/integration/test_pipeline.py`

One end-to-end smoke test. Uses a tiny synthetic dataset (10 examples per split).
Runs one epoch of stage 1 training and confirms:
- No exceptions raised
- A checkpoint directory is created
- A results file is written to the expected path

This test is marked with `@pytest.mark.slow` and excluded from the default CI run
(too compute-intensive). It is documented in the QA plan as a local-only test.

**Acceptance criteria:**
- `pytest tests/unit/ -v` passes with no failures
- Coverage on `src/data.py`: ≥ 80%
- Coverage on `src/utils.py`: ≥ 80%

---

## 11. Phase 7 — QA and Polish

**Goal:** Verify the full repository is in a clean, portfolio-ready state. Confirm CI passes.
Sweep for remaining placeholders. Review commit history.

**Depends on:** All prior phases complete.

**Permission tier:** Allowed without asking (documentation and formatting — CLAUDE.md §3).

### Deliverables

#### 7.1 — CI validation

Run `scripts/validate-structure.sh --strict` locally and resolve any reported failures.
Confirm all four CI jobs pass: lint, test, validate-structure, security.

#### 7.2 — Final placeholder sweep

Search all `.md`, `.yml`, `.sh` files for any remaining `{{PLACEHOLDER}}` or
"fill in" / "More Information Needed" tokens. Resolve or document as intentionally deferred.

#### 7.3 — `docs/qa/qa-plan.md` — test file inventory

Populate the test file inventory table with the actual test files created in Phase 6
and their coverage percentages.

#### 7.4 — CLAUDE.md update

Update the CLAUDE.md entry protocol status line and architecture summary:
- `*Entry protocol completed: yes — 2026-03-14*`
- Section 6 (Architecture Summary): fill the 3–5 sentence description

#### 7.5 — Final documentation review

Read README.md, docs/architecture.md, and docs/implementation-plan.md as a first-time
reader would. Identify and fix any internal inconsistencies or broken references.

---

## 12. Phase Dependencies Summary

```
Phase 0  ──────────────────────────────────────────────► Phase 1
(Foundation)                                             (ADRs)
   │                                                        │
   │                                                        ▼
   └──────────────────────────────────────────────────► Phase 2
                                                         (Docs)
                                                            │
                    D1 + D4 approved ◄──────────────────────┤
                          │                                  │
                          ▼                                  │
                      Phase 3  ──────────────────────────────┤
                   (Src Restructure)                         │
                          │                                  │
               D1=LoRA? ──┤         D5 approved ◄────────────┤
                          │              │                    │
                          ▼              ▼                    │
                      Phase 4       Phase 5                   │
                    (LoRA Script) (Evaluate)                  │
                          │              │                    │
                          └──────┬───────┘                    │
                                 ▼                            │
                             Phase 6  ──────────────────────► Phase 7
                           (Test Suite)                    (QA + Polish)
```

---

## 13. File Creation Summary

The following files will be created or substantially modified. Files requiring approval
are marked with `[A]`.

| File | Phase | Action | Approval Required |
|------|-------|--------|-------------------|
| `requirements.txt` | 0 | Create | No |
| `.env.example` | 0 | Modify | No |
| `README.md` | 0, 2 | Modify | No |
| `docs/runbooks/operations.md` | 0, 2 | Modify | No |
| `docs/adr/ADR-001-model-selection.md` | 1 | Create | Yes (decision approval) |
| `docs/adr/ADR-002-training-strategy.md` | 1 | Create | Yes (D1) |
| `docs/adr/ADR-003-dataset-ordering.md` | 1 | Create | Yes (decision approval) |
| `docs/adr/ADR-004-module-structure.md` | 1 | Create | Yes (D4) |
| `docs/adr/ADR-005-evaluation-scope.md` | 1 | Create | Yes (D5) |
| `docs/adr/ADR-006-notebook-disposition.md` | 1 | Create | Yes (D6) |
| `docs/qa/qa-plan.md` | 2, 7 | Modify | No |
| `models/deberta-pi-lora-final-adapter/README.md` | 2 | Modify | No |
| `models/deberta-pi-lora-final-full/README.md` | 2 | Modify | No |
| `src/config.py` | 3 | Create | Yes [A] |
| `src/data.py` | 3 | Create | Yes [A] |
| `src/utils.py` | 3 | Create | Yes [A] |
| `src/train.py` | 3 | Create | Yes [A] |
| `src/finetune.py` | 3 | Deprecate/delete | Yes [A] — separate approval for deletion |
| `src/finetune_2.py` | 3 | Deprecate/delete | Yes [A] — separate approval for deletion |
| `src/train_lora.py` | 4 | Create (if D1 selects LoRA) | Yes [A] |
| `src/evaluate.py` | 5 | Create | Yes [A] |
| `src/test_model.py` | 5 | Deprecate/delete | Yes [A] — separate approval for deletion |
| `tests/unit/test_data.py` | 6 | Create | No |
| `tests/unit/test_utils.py` | 6 | Create | No |
| `tests/unit/test_config.py` | 6 | Create | No |
| `tests/integration/test_pipeline.py` | 6 | Create | No |
| `CLAUDE.md` | 7 | Modify (status + arch summary) | No |

---

## 14. Phase 8 — Experiment Reproducibility

**Goal:** Make training and evaluation runs reproducible and auditable. Every model
directory must contain a complete, self-contained record of the parameters that produced it.

**Depends on:** Phase 7 (CI-clean baseline).

**Permission tier:** Allowed without asking (additive, no signature changes — CLAUDE.md §3).

### Deliverables

#### 8.1 — `GLOBAL_SEED` constant in `src/config.py`

Add `GLOBAL_SEED = 42` under a new `Reproducibility` section. This centralizes the
seed value so both training pipelines read from a single source of truth.

#### 8.2 — `write_run_config()` helper in `src/utils.py`

Add a function that writes `run_config.json` to a caller-specified output directory.
The file captures:
- ISO 8601 UTC timestamp
- Short git commit hash (empty string if unavailable)
- Python version
- PyTorch version
- All caller-supplied fields (seed, hyperparameters, pipeline name)

#### 8.3 — CLI `--seed` argument and `write_run_config()` call in `src/train.py`

- Import `write_run_config` from `utils`
- Replace hardcoded `set_global_seed(42)` with `set_global_seed(args.seed)`
- Add `argparse` with `--seed INT` (default: `config.GLOBAL_SEED`)
- Call `write_run_config(config.PLOTS_DIR, {...})` immediately after seed is set

#### 8.4 — CLI `--seed` argument and `write_run_config()` call in `src/train_lora.py`

Same pattern as 8.3. The config snapshot includes LoRA-specific hyperparameters
(`r`, `alpha`, `dropout`, `target_modules`, `bias`) in addition to training constants.

### Files modified

| File | Change |
|------|--------|
| `src/config.py` | Add `GLOBAL_SEED = 42` |
| `src/utils.py` | Add `write_run_config()` function; add `json`, `subprocess`, `sys`, `datetime` imports |
| `src/train.py` | Add `--seed` CLI arg; replace hardcoded seed; call `write_run_config()` |
| `src/train_lora.py` | Add `--seed` CLI arg; replace hardcoded seed; call `write_run_config()` |
| `docs/implementation-plan.md` | Add this Phase 8 section |

---

## 15. Phase 10 — Inference Utilities

**Goal:** Make the repository usable for direct prompt-injection inference without
requiring a dataset download or evaluation pipeline. Any saved model artifact — full
fine-tuned or LoRA adapter — should be loadable and queryable from a single entry point.

**Depends on:** Phase 7 (CI-clean baseline); Phase 5 (`src/evaluate.py` established
the tokenization and inference patterns this phase reuses).

**Permission tier:** Allowed without asking (new files, no signature changes — CLAUDE.md §3).

### Deliverables

#### 10.1 — `src/inference.py`

Unified inference module. Public API:

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `load_model` | `(model_path: str) → (model, tokenizer, device)` | Auto-detects artifact format; puts model in eval mode |
| `predict` | `(text, model, tokenizer, device) → dict` | Single-text inference with label and probability output |
| `predict_batch` | `(texts, model, tokenizer, device) → list[dict]` | Batch inference; same dict structure per item |

**Format detection:** `load_model` checks for `adapter_config.json` in `model_path`.
If present, the directory is a LoRA adapter and `PeftModel.from_pretrained` is used;
the base model ID is read from `adapter_config.json` (`base_model_name_or_path`).
If absent, `AutoModelForSequenceClassification.from_pretrained` is used directly.
A clear `ImportError` is raised if peft is unavailable and a LoRA path is requested.

**Return dict structure** (from `predict` and each element of `predict_batch`):

```python
{
    "label":         int,   # 0 = safe, 1 = injection
    "label_str":     str,   # "safe" or "injection"
    "probability":   float, # softmax confidence in the predicted label
    "probabilities": {"safe": float, "injection": float},
}
```

Probabilities are derived from `torch.nn.functional.softmax` applied to raw logits —
matching the decision rule in `src/evaluate.py` (`torch.argmax`) while adding
calibrated confidence output.

**CLI** (via `argparse`, `if __name__ == "__main__"`):

```
python src/inference.py --model-path <path> --text "..."
python src/inference.py --model-path <path> --text "..." --output-format json
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-path` | Yes | — | Path to model directory |
| `--text` | Yes | — | Input text to classify |
| `--output-format` | No | `text` | `text` (human-readable) or `json` (machine-readable) |

Loading progress is written to `stderr`; the prediction result is written to `stdout`,
keeping the JSON output format composable with downstream tooling.

#### 10.2 — `tests/unit/test_inference.py`

30 unit tests. All mocked — no network access, no disk model loading. Coverage:

- `_is_lora_adapter`: detects `adapter_config.json` presence and absence
- `_read_base_model_id`: correct read, missing field, empty string
- `load_model`: full FT path (mocked AutoModel), LoRA path (mocked PeftModel),
  peft import guard (monkeypatched `sys.modules`)
- `predict`: correct label, correct `label_str`, probability matches predicted label,
  probabilities sum to 1.0, return key set
- `predict_batch`: empty list, single item, multiple items, structure matches `predict`
- `_format_text`: label string present, confidence formatted correctly

### Files added

| File | Action |
|------|--------|
| `src/inference.py` | Create |
| `tests/unit/test_inference.py` | Create |

### Acceptance criteria

- `python src/inference.py --help` exits 0 and prints usage
- `python -c "import sys; sys.path.insert(0, 'src'); import inference"` exits 0
- `pytest tests/ -v` passes with 0 failures (new tests increase pass count)
- `ruff check src/inference.py tests/unit/test_inference.py` — 0 errors
- `black --check src/inference.py tests/unit/test_inference.py` — 0 reformats

### Verification status

**Complete.** Verified 2026-03-15:
- ruff: PASS
- black: PASS
- pytest: 79 passed, 17 skipped, 0 failed (+23 new inference tests)
- `--help`: PASS
- import isolation: PASS

---

*This implementation plan is read-only during coding passes.*
*Last updated: 2026-03-15*
