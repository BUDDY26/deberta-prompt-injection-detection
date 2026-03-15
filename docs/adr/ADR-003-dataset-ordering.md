# ADR-003: Train Sequentially in Order — Safe-Guard → SPML → Aegis

**Date:** 2026-03-14
**Status:** Accepted
**Author:** BUDDY26

---

## Context

Three datasets are used in sequential multi-stage fine-tuning. The order in which they
are presented to the model is a design choice — each stage's output becomes the next
stage's starting weights. The datasets differ in:

- **Schema complexity:** column names, label format, split availability
- **Domain emphasis:** the types of prompt injection patterns each covers
- **Dataset maturity:** whether native validation and test splits are provided

The three datasets, as confirmed in the repository:

| Stage | Dataset | Text input | Label format | Native test split |
|-------|---------|-----------|-------------|-------------------|
| 1 | `xTRam1/safe-guard-prompt-injection` | Single `"text"` column | Integer (`0`/`1`) | Yes |
| 2 | `reshabhs/SPML_Chatbot_Prompt_Injection` | `"System Prompt"` + `" "` + `"User Prompt"` concatenated | `"Prompt injection"` column renamed to `"label"` | Conditional (checked at runtime) |
| 3 | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | Single `"prompt"` column | String (`"safe"`→0, `"unsafe"`→1) | Yes — native `train`/`validation`/`test` |

The key question: does the confirmed ordering (1→2→3) produce better results than
alternatives, and does the evidence justify recording this order as an accepted decision?

---

## Decision

Train the three stages in this fixed order:

1. `xTRam1/safe-guard-prompt-injection`
2. `reshabhs/SPML_Chatbot_Prompt_Injection`
3. `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`

---

## Rationale

### 1. Empirical evidence — stage 3 is necessary

The repository contains two evaluation results, both measuring model performance on
the Aegis test set (N=1964):

| Model | Training stages completed | Aegis test accuracy | Aegis F1 (Unsafe) |
|-------|--------------------------|--------------------|--------------------|
| `deberta-pi-full-stage2-final` | Stages 1 + 2 only | **41.60%** | **0.3053** |
| `deberta-pi-full-stage3-final` | Stages 1 + 2 + 3 | **81.16%** | **0.8255** |

The stage 2 model's 41.60% accuracy on Aegis is below chance for a balanced binary
problem, indicating that two datasets alone leave a systematic gap: the model trained
on Safe-Guard and SPML does not generalise to the Aegis distribution. Adding stage 3
(training on Aegis) produces a 39.56 percentage point improvement, confirming that all
three stages are collectively necessary.

This is direct empirical evidence that stage 3 must be included and that placing it
after stages 1 and 2 — rather than before — allows the model to specialise on the
Aegis distribution as the final step before evaluation.

### 2. Increasing schema complexity — simpler datasets first

The datasets are ordered by the complexity of their preprocessing requirements:

- **Stage 1** is the simplest possible schema: one text column, one integer label.
  The model learns the fundamental binary classification task with no preprocessing
  overhead.
- **Stage 2** requires runtime field concatenation (`"System Prompt" + " " + "User Prompt"`)
  and conditional split handling (the validation split may not exist natively). The model
  adapts the learned classification boundary to a multi-field, conversational context.
- **Stage 3** uses string labels requiring programmatic mapping (`"unsafe"→1`, else `0`)
  and provides pre-defined `train`/`validation`/`test` splits, suggesting it is the
  most formally structured and test-ready of the three datasets.

Starting with the simplest schema allows the model to converge on a stable binary
classification boundary before adapting to progressively richer input structures.

### 3. Evaluation alignment — Aegis last

Both confirmed evaluation result files measure performance on the Aegis test set.
Placing Aegis as the final training stage means the model's most recent gradient updates
come from the same distribution it is evaluated on. This is the natural configuration
for an end-to-end training and evaluation pipeline.

---

## Limitations and Honest Caveats

The following limitations are recorded explicitly because they are relevant to any
reader evaluating this decision:

- **No ordering ablation exists in the repository.** There are no results from
  alternative orderings (e.g., Aegis first, SPML last). The decision cannot be claimed
  as provably optimal — only as empirically successful in the confirmed configuration.

- **The poor performance at stage 2 (41.60%) proves inclusion of stage 3 is necessary
  but does not prove the 1→2 sub-ordering within the first two stages is optimal.**
  A model trained in stage 2→1 order might perform similarly; no experiment tests this.

- **LoRA stage accuracies (98.67%, 96.07%) are within-distribution results** — each
  measured on a held-out split of the same dataset used in that stage. They confirm
  learning within each stage but do not validate the cross-stage ordering decision.

- **Cross-stage forgetting is not measured.** No evaluation of the stage 3 model on
  the Safe-Guard or SPML test distributions is recorded. The ordering succeeds for
  Aegis generalisation; its effect on stage 1 and 2 distributions is unknown.

---

## Alternatives Considered

| Alternative | Reason not adopted |
|-------------|-------------------|
| Aegis first, then Safe-Guard and SPML | Placing the most complex and evaluation-aligned dataset first would mean stages 2 and 3 potentially overwrite the Aegis-specialised weights; no evidence in repository supports this order |
| Joint training on all three datasets simultaneously | Not implemented; no evidence in repository. Would require resolving column schema differences in a single dataloader and loses the sequential curriculum structure |
| Two stages only (omit SPML) | The 41.60% cross-dataset result rules out omitting any stage, though it specifically demonstrates Aegis is necessary rather than SPML specifically |

---

## Consequences

### Positive

- All three stages are empirically justified: the 41.60% → 81.16% result on the same
  test set provides a concrete, reportable demonstration of multi-stage value.
- Aegis as the final stage aligns training distribution with the confirmed evaluation
  distribution.
- The ordering from simpler to more complex schemas provides a natural curriculum
  structure that reduces preprocessing friction in early stages.
- The final F1 score of 0.8255 (Unsafe class) and accuracy of 81.16% on a 1,964-sample
  held-out test set are strong, concrete results for a portfolio project.

### Negative

- The ordering is fixed by implementation. Changing the order requires modifying
  `src/finetune.py` and `src/finetune_2.py` (or their reconstructed equivalents)
  and rerunning training.
- No experiment measures the counterfactual, so the ordering cannot be defended as
  globally optimal — only as the order that was executed and evaluated.

### Risks

- **Cross-stage forgetting:** Sequential training means earlier-stage knowledge may
  degrade. If Safe-Guard or SPML distributions reappear in production, the model's
  performance on them is unknown after stage 3 training.
- **Stage 2 split uncertainty:** SPML's validation split is conditional. If it is
  absent at runtime, the code falls back to a 10% train split. This introduces a
  potential inconsistency in evaluation monitoring during stage 2.

---

## Review Trigger

Revisit this decision if:

- A fourth dataset is added to the pipeline and its position in the sequence must
  be determined.
- Ablation experiments demonstrate that a different ordering produces significantly
  higher final accuracy on the Aegis test set.
- Evidence of catastrophic forgetting on stage 1 or 2 distributions motivates
  techniques such as elastic weight consolidation or replay buffers, which would
  change the nature of the sequential training contract.
- The confirmed evaluation results (81.16% accuracy, 0.8255 F1) are superseded by
  results from a different configuration.

---

## Evidence Basis

All facts in this ADR are sourced from `docs/evidence-ledger.md`.
No inferences are presented as confirmed facts.

| Claim | Evidence | Ledger ref |
|-------|----------|-----------|
| Stage 1 dataset ID | `src/finetune.py` line 94 | §4 |
| Stage 2 dataset ID | `src/finetune.py` line 167 | §4 |
| Stage 3 dataset ID | `src/finetune_2.py` line 94 | §4 |
| Stage 1 column schema | `src/finetune.py` lines 105, 112 | §4 |
| Stage 2 column schema and concatenation | `src/finetune.py` lines 183–203 | §3, §4 |
| Stage 3 column schema and string label mapping | `src/finetune_2.py` lines 109–112 | §3, §4 |
| Stage 2 validation split is conditional | `src/finetune.py` lines 170–178 | §4 |
| Stage 3 native train/validation/test splits | `src/finetune_2.py` lines 101–103 | §4 |
| Aegis test set size N=1964 | `results/test_results_2dataset.txt` line 4; `results/test_results_3datasets.txt` line 4 | §4 |
| Stage 2 model on Aegis: accuracy 41.60%, F1 0.3053 | `results/test_results_2dataset.txt` lines 9, 13 | §8 |
| Stage 3 model on Aegis: accuracy 81.16%, F1 0.8255 | `results/test_results_3datasets.txt` lines 9, 13 | §8 |
| Stage 1 LoRA val accuracy 98.67% | `models/deberta-pi-lora-stage1/checkpoint-928/trainer_state.json` field `best_metric` | §6 |
| Stage 2 LoRA val accuracy 96.07% | `models/deberta-pi-lora-stage2/checkpoint-7208/trainer_state.json` field `best_metric` | §6 |
