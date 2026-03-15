# ADR-004: Source Module Structure — Flat Layout vs. Package Layout

**Date:** 2026-03-14
**Status:** Accepted
**Author:** BUDDY26

---

## Context

The current `src/` directory contains three monolithic scripts with no package structure:

```
src/
├── finetune.py       # Stages 1+2 — full fine-tuning, 284 lines
├── finetune_2.py     # Stage 3  — full fine-tuning, 188 lines
├── test_model.py     # Evaluation script, 317 lines
└── .gitkeep
```

These scripts are not importable as modules — they execute sequentially from top to bottom
when run. No `__init__.py` exists. No shared utilities are extracted; the same functions
(e.g., `plot_training_metrics`, `compute_metrics`) are duplicated across `finetune.py`
and `finetune_2.py`.

Phase 3 of the implementation plan restructures `src/` into a modular layout. Two
structural approaches were available. The choice determines how modules are organised,
how they import from each other, and whether the project is pip-installable.

---

## Confirmed current state

| File | Lines | Duplicated content |
|------|-------|--------------------|
| `src/finetune.py` | 284 | `compute_metrics`, `plot_training_metrics` |
| `src/finetune_2.py` | 188 | `compute_metrics`, `plot_training_metrics` (identical copies) |
| `src/test_model.py` | 317 | Independent — evaluation only |

Source: `docs/evidence-ledger.md` §7

The implementation plan (Phase 3) identifies five to six target modules regardless of layout:
`config`, `data`, `utils`, `train`, `evaluate`, and `train_lora` (per ADR-002 Decision).

---

## Decision

Use a flat layout (Option A). All modules live directly under `src/` as top-level Python
files. No subdirectories. No `__init__.py`.

### Target layout

```
src/
├── config.py       # Hyperparameters, dataset IDs, path roots
├── data.py         # Dataset loading and preprocessing for all three stages
├── utils.py        # Shared utilities: metric computation, plotting, seed setting
├── train.py        # Full fine-tuning entry point — orchestrates three-stage pipeline
├── train_lora.py   # LoRA entry point (per ADR-002; blocked on D2, D3, D6)
└── evaluate.py     # Post-training evaluation and results output
```

### Running

```bash
python src/train.py
python src/evaluate.py --model-path <path> --dataset aegis
```

### Module imports

Modules import each other using `sys.path` insertion or by running from the repository
root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python src/train.py
```

Tests use a `conftest.py` at the repository root that adds `src/` to `sys.path` before
any test module is collected.

### Rationale

- Matches the implementation plan's Phase 3 layout exactly. No deviation from the approved
  plan is required.
- The project is a training pipeline script, not a reusable library. Flat layout is idiomatic
  for this use case in the HuggingFace/PyTorch ecosystem.
- With six modules (including `train_lora.py`), a flat layout remains fully navigable.
- No `__init__.py`, `pyproject.toml`, or `setup.py` is needed — no maintenance overhead
  beyond the source files themselves.
- The run command (`python src/train.py`) matches what is already documented in
  `README.md` and `docs/runbooks/operations.md`, requiring no updates to those files.

---

## Alternative Considered

### Option B — Package Layout

`src/` structured as one or more proper Python packages with `__init__.py` files.

Two sub-variants were considered:

**B1 — Single package (`src/deberta_pid/`):**
```
src/
└── deberta_pid/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── utils.py
    ├── train.py
    ├── evaluate.py
    └── train_lora.py
```

**B2 — Subdirectory grouping by concern:**
```
src/
├── __init__.py
├── config.py
├── pipeline/
│   ├── __init__.py
│   ├── train.py
│   └── train_lora.py
├── data/
│   ├── __init__.py
│   └── loader.py
└── evaluation/
    ├── __init__.py
    └── evaluate.py
```

Not chosen because: the project is not intended to be pip-installable or used as an importable
library; the `__init__.py` and `pyproject.toml` overhead is not justified for six files; the
`python -m deberta_pid.train` invocation pattern is less familiar to ML practitioners than
`python src/train.py`; and the implementation plan's Phase 3 describes flat files, so choosing
B would have required updating the plan before writing any code.

---

## Consequences

### Development experience

| Concern | Outcome |
|---------|---------|
| Run command | `python src/train.py`, `python src/train_lora.py` |
| Import style between modules | `sys.path` insertion or `PYTHONPATH=src` |
| Test import style | `sys.path` added in `conftest.py` at repo root |
| `__init__.py` required | No |
| `pyproject.toml` / `setup.py` required | No |
| README and runbook run commands | No changes needed — already document `python src/train.py` |

### CI impact

| Concern | Outcome |
|---------|---------|
| `ruff check src/` | Works with no configuration changes |
| `pytest tests/` | Requires `conftest.py` at repo root to add `src/` to `sys.path` |
| Structure validation script | No changes needed |

### Phase 3 implementation scope

The flat layout is the layout described in the implementation plan. Phase 3 proceeds exactly
as planned:

| Deliverable | File path |
|-------------|-----------|
| Configuration module | `src/config.py` |
| Data module | `src/data.py` |
| Utilities module | `src/utils.py` |
| Full fine-tuning entry point | `src/train.py` |
| Evaluation entry point | `src/evaluate.py` |
| LoRA entry point (Phase 4) | `src/train_lora.py` |

### Known limitation

`sys.path` manipulation is less elegant than clean package imports. If the project grows
beyond 8–10 source files, or if it is packaged for distribution, Option B should be revisited.
The Review Trigger below records the conditions under which this decision should be re-examined.

---

## Review Trigger

Revisit if:
- The project is packaged for distribution (PyPI, internal package index), at which
  point Option B becomes clearly appropriate.
- The number of source modules grows beyond 10, at which point flat layout becomes
  unwieldy.
- The CI toolchain changes in a way that makes `sys.path` manipulation in
  `conftest.py` unreliable.

---

## Evidence Basis

| Claim | Evidence | Ledger ref |
|-------|----------|-----------|
| Current `src/` contains three monolithic scripts | File listing | §7 |
| `finetune.py` and `finetune_2.py` both define `compute_metrics` and `plot_training_metrics` | `src/finetune.py` lines 19–85; `src/finetune_2.py` lines 19–85 (identical functions) | §7 |
| No `__init__.py` exists in `src/` | File listing — only `.gitkeep`, `finetune.py`, `finetune_2.py`, `test_model.py` | §7 |
| Implementation plan Phase 3 proposes flat layout | `docs/implementation-plan.md` §7 deliverables 3.1–3.4 | N/A (plan doc) |
| CI runs `ruff check src/ tests/` and `black --check src/ tests/` | `.github/workflows/ci.yml` lines 44, 47 | §7 |
| CI runs `pytest tests/ -v --cov=src` | `.github/workflows/ci.yml` lines 88–89 | §7 |
| `docs/runbooks/operations.md` run command: `python src/train.py` | `docs/runbooks/operations.md` line 42 | §9 |
