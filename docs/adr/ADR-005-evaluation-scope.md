# ADR-005: Evaluation Scope — Which Models and Datasets to Evaluate

**Date:** 2026-03-14
**Status:** Accepted
**Author:** BUDDY26

---

## Context

The repository contains two confirmed post-training result files, both produced by
`src/test_model.py`. Both evaluate a **full fine-tuning model against the Aegis dataset only**:

| File | Model evaluated | Dataset | Accuracy | F1 (Unsafe) |
|------|----------------|---------|----------|-------------|
| `results/test_results_2dataset.txt` | `deberta-pi-full-stage2-final` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | 41.60% | 0.3053 |
| `results/test_results_3datasets.txt` | `deberta-pi-full-stage3-final` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | 81.16% | 0.8255 |

Source: `results/test_results_2dataset.txt` lines 5–6, 9–13;
`results/test_results_3datasets.txt` lines 5–6, 9–13; evidence-ledger §8.

### What `src/test_model.py` currently does

- Loads a hardcoded model path (`MODEL_PATH = "deberta-pi-full-stage3-final"`)
- Loads the Aegis test split (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`)
- Reports: overall accuracy, binary precision/recall/F1 (Unsafe class), confusion matrix,
  full classification report
- Writes results to the console only — no parameterized output file path

Source: `src/test_model.py` lines 13–16, 44–57; evidence-ledger §5, §9 (confirmed path bug).

### What is NOT currently evaluated

- No evaluation of any model against the Safe-Guard or SPML test splits
- No evaluation of any LoRA model against any dataset — no LoRA result files exist anywhere
  in the repository
- No evaluation of stage 1 or stage 2 full fine-tuning models against their own held-out
  test splits

Source: repository file listing (`results/` contains exactly two files, both Aegis-only);
evidence-ledger §7 (result files table).

### Decision D5 — Evaluation Scope

This ADR governs **Decision D5**: what `src/evaluate.py` (Phase 5) must evaluate.
The scope choice determines:
- Which result files are produced
- How much new evaluation code is required
- What claims the portfolio can make

---

## Decision

**Option A — Final model vs. Aegis only** is accepted as the canonical evaluation scope.

`src/evaluate.py` will:
- Accept `--model-path` and `--dataset` as command-line arguments
- Evaluate one model against one dataset per invocation
- Write results to a deterministic path under `results/evaluation/`
- Report the same metrics as `src/test_model.py`: overall accuracy, binary
  precision/recall/F1 (Unsafe class), confusion matrix, full classification report

The two confirmed result files in `results/` both evaluate a full fine-tuning model against
the Aegis test set (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`). Option A reproduces this
behavior exactly, replacing hardcoded paths with parameterized arguments and a fixed output
path. No new evaluation logic is introduced; no training re-runs are required.

This choice is the most faithful to the confirmed repository state. Both result files in
`results/` are Aegis-only. The evaluation scope defined here matches what was actually
recorded during the original experiment. Evidence-ledger §7 confirms no other result files
exist; §8 records the two confirmed values (41.60% and 81.16% on the Aegis test set, N=1964).

**The LoRA pipeline has no cross-dataset evaluation result.** This limitation is documented
honestly in the model card READMEs for `deberta-pi-lora-final-adapter` and
`deberta-pi-lora-final-full`. It is not a defect of this decision — it reflects the scope
of the original training experiment.

---

## Alternatives Considered

### Option A — Final model vs. Aegis only (selected)

**Scope:** Reproduce the two confirmed result files. Evaluate the stage 2 and stage 3 full
fine-tuning models against the Aegis test set, with parameterized paths replacing the
current hardcoded values in `test_model.py`.

**What `src/evaluate.py` would do:**
- Accept `--model-path` and `--dataset` as arguments
- Evaluate one model against one dataset per invocation
- Write results to a deterministic path under `results/evaluation/`
- Produce identical metrics to the confirmed result files (accuracy, binary F1, confusion
  matrix, classification report)

**Result files produced:**
```
results/evaluation/deberta-pi-full-stage2-final_aegis_results.txt
results/evaluation/deberta-pi-full-stage3-final_aegis_results.txt
```

| Dimension | Assessment |
|-----------|-----------|
| Repository complexity | Lowest — directly matches confirmed behavior |
| Result reproducibility | Exact — reproduces the two confirmed result files |
| Portfolio clarity | Strong — confirms the 41.60% → 81.16% improvement story clearly |
| New evaluation logic | None — same metrics, same dataset, parameterized only |

**Drawback:** Does not evaluate the LoRA pipeline. A reviewer who reads the model cards
for `deberta-pi-lora-final-adapter` and `deberta-pi-lora-final-full` will find
within-distribution training metrics (val accuracy 98.67%, 96.07%) but no cross-dataset
evaluation result for LoRA. This limitation is documented honestly in the model card READMEs.

---

### Option B — Per-stage evaluation against each stage's held-out test split

**Scope:** In addition to the Aegis cross-dataset evaluations, evaluate each full
fine-tuning stage model against the held-out test split of its own training dataset.

**What `src/evaluate.py` would do:**
- Everything in Option A, plus:
  - Evaluate `deberta-pi-full-stage1-final` against the Safe-Guard test split
  - Evaluate `deberta-pi-full-stage2-final` against the SPML test split (or val if no test)
  - Evaluate `deberta-pi-full-stage3-final` against the Aegis test split (already in Option A)

**Result files produced (additions over Option A):**
```
results/evaluation/deberta-pi-full-stage1-final_safeguard_results.txt
results/evaluation/deberta-pi-full-stage2-final_spml_results.txt
```

| Dimension | Assessment |
|-----------|-----------|
| Repository complexity | Moderate — three datasets, three models, parameterized evaluation |
| Result reproducibility | New — stage 1 and stage 2 within-distribution results are not currently confirmed; they would be newly generated |
| Portfolio clarity | Higher — shows per-stage learning progression; gives the LoRA pipeline a comparison baseline |
| New evaluation logic | Modest — additional label mapping required for SPML column names |

**Important:** The stage 1 and stage 2 within-distribution result values do not currently
exist in the repository. These would be **new results** generated by running the evaluate
script for the first time, not reproductions of existing confirmed values. This is honest
new work, not reconstruction.

**Drawback:** The saved stage 1 and stage 2 full fine-tuning model directories
(`deberta-pi-full-stage1-final`, `deberta-pi-full-final`) are not committed to the
repository — they are outputs of the training pipeline and must be re-generated by running
`src/train.py`. Option B therefore cannot be run without re-training or the models being
present locally. This is a prerequisite dependency that should be documented.

---

### Option C — Cross-dataset matrix evaluation

**Scope:** Evaluate multiple models against multiple datasets in a single matrix run,
producing a table of cross-dataset transfer results.

**Example matrix:**

| Model | Safe-Guard test | SPML test | Aegis test |
|-------|----------------|-----------|-----------|
| `deberta-pi-full-stage1-final` | ✓ | ✓ | ✓ |
| `deberta-pi-full-stage2-final` | ✓ | ✓ | ✓ (confirmed) |
| `deberta-pi-full-stage3-final` | ✓ | ✓ | ✓ (confirmed) |
| `deberta-pi-lora-final-full` | ✓ | ✓ | ✓ |

**What `src/evaluate.py` would do:**
- Accept a list of model paths and a list of dataset names
- Run all combinations, handle each dataset's column schema and label mapping
- Produce a summary table across all combinations

| Dimension | Assessment |
|-----------|-----------|
| Repository complexity | Highest — multi-model, multi-dataset loop; separate preprocessing per dataset |
| Result reproducibility | Mixed — two results confirmed; the rest are new and require training artifacts not in the repository |
| Portfolio clarity | Potentially compelling cross-transfer results, but claims require the training runs to be re-executed |
| New evaluation logic | Substantial — each dataset requires its own column handling and label mapping |

**Drawback:** Only two of the many possible result cells are confirmed by existing repository
artifacts. All other cells would be newly generated values. Running the full matrix requires
all model checkpoints to be present, which in turn requires re-running both the full
fine-tuning and LoRA training pipelines. This is disproportionate complexity for Phase 5 of
a portfolio reconstruction project where training has already been completed.

---

## Comparison Summary

| | Option A | Option B | Option C |
|--|----------|----------|----------|
| Confirmed results reproduced | 2 | 2 | 2 |
| New results generated | 0 | 2 | 6+ |
| Models required to be present | 1 (stage3-final) | 3 (all FT stages) | All FT + LoRA |
| Evaluate.py implementation complexity | Low | Moderate | High |
| All results verifiable from existing artifacts | Yes | No | No |

---

## Consequences of Each Option

### If Option A is chosen

- `src/evaluate.py` is a clean, minimal replacement for `src/test_model.py`.
- The two confirmed result files are verifiable and reproducible from the committed code alone.
- The LoRA pipeline has no cross-dataset evaluation result. This is the honest state of the
  original experiment and is documented as such.
- Phase 5 implementation is low-risk and straightforward.

### If Option B is chosen

- Phase 5 implementation requires handling Safe-Guard and SPML column schemas in the
  evaluate script, in addition to Aegis.
- Two new within-distribution result files are generated — they are new work, not
  reconstruction. This should be called out clearly in the commit message and README.
- Option B is appropriate if the portfolio goal includes showing per-stage learning progression.

### If Option C is chosen

- Phase 5 implementation is substantially larger than the other options and introduces
  significant new behavior beyond reconstructing the original experiment.
- Cross-dataset transfer claims require re-running training, which is outside the stated
  goal of this repository reconstruction project (evidence-ledger §1; implementation-plan §1).
- Option C is not recommended under the current project scope without a separate approval
  to re-run training experiments.

---

## Evidence Basis

| Claim | Evidence | Ledger ref |
|-------|----------|-----------|
| Only two result files exist, both Aegis-only | `results/` directory listing — two files only | §7 |
| Stage 2 model Aegis accuracy: 41.60% | `results/test_results_2dataset.txt` line 9 | §8 |
| Stage 3 model Aegis accuracy: 81.16% | `results/test_results_3datasets.txt` line 9 | §8 |
| `test_model.py` hardcodes model path and dataset | `src/test_model.py` lines 13–14, 44–45 | §9 |
| No LoRA evaluation result files exist | `results/` directory listing — no LoRA-named result file | §7 |
| LoRA stage 1 best val accuracy: 98.67% | `trainer_state.json` field `best_metric` | §6 |
| LoRA stage 2 best val accuracy: 96.07% | `trainer_state.json` field `best_metric` | §6 |
| Training re-runs are out of scope | Implementation plan §1 "What this plan does NOT include" | implementation-plan.md |
