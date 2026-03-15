# QA Plan

**Project:** deberta-prompt-injection-detection
**Test framework:** pytest
**Last updated:** 2026-03-14

---

## Test Strategy

This project is a data-processing and model-training pipeline, not a web application. The testing
strategy reflects that distinction: the highest-value tests are unit tests that verify data
preprocessing logic (tokenization, label mapping, dataset loading) and metric computation, since
these are the areas most likely to produce silent errors that corrupt training without raising
exceptions.

End-to-end retraining tests are explicitly out of scope for CI — a full three-stage training run
requires a GPU and several hours; automating it in CI would be impractical and wasteful. The CI
pipeline instead verifies preprocessing and utility logic at the unit level, plus a lightweight
smoke test that confirms the pipeline can be imported and instantiated without errors.

**Confidence target:** High confidence in data correctness (preprocessing, label mapping, split
handling); moderate confidence in pipeline wiring (smoke test); zero automated confidence in final
model accuracy (that is verified manually against `results/`).

---

## Test Layers

### Unit Tests — `tests/unit/`

- **Scope:** Individual functions in isolation — tokenization, label encoding, metric computation,
  dataset schema handling, configuration values
- **Mocking policy:** Mock HuggingFace dataset downloads (use small synthetic fixtures or cached
  samples); do not mock internal preprocessing logic
- **Coverage target:** 80% minimum per source file in `src/`
- **Framework:** pytest
- **Key targets (once Phase 3 modularizes `src/`):**
  - `src/data.py` — dataset loading, column renaming, label mapping, split fallback logic
  - `src/utils.py` — `compute_metrics`, `plot_training_metrics`, seed setting
  - `src/config.py` — configuration values match evidence ledger (hyperparameters, paths)

### Integration Tests — `tests/integration/`

- **Scope:** Pipeline smoke test — confirm that modules can be imported and a forward pass can
  be instantiated with a small model stub or the real base model (if available)
- **Environment:** Requires installed dependencies; GPU not required (CPU smoke test)
- **When to run:** Before every merge; always in CI

---

## Test Command

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
```

CI runs the full coverage command. Coverage XML is uploaded as a CI artifact (retained 7 days,
Python 3.11 run only).

---

## Coverage Policy

| Result | Threshold |
|--------|-----------|
| Pass | ≥ 80% overall |
| Warning | 60–79% |
| Fail | < 60% |

---

## Test File Inventory

*Phase 6 complete as of 2026-03-14. Coverage percentages are not yet measured; CI will report actuals on first run.*

| Source File | Unit Test File | Integration Test | Notes |
|-------------|----------------|-----------------|-------|
| `src/config.py` | `tests/unit/test_config.py` | No | 37 assertions covering all confirmed hyperparameters, dataset IDs, output dirs, and LoRA constants; LoRA tests skipped when peft absent |
| `src/data.py` | `tests/unit/test_data.py` | No | Covers all four text-column branches in `_preprocess_spml_batch`, label rename for all three stages, max_length=256 propagation; no network calls (load_dataset monkeypatched) |
| `src/utils.py` | `tests/unit/test_utils.py` | No | Covers `compute_metrics` (all-correct, all-wrong, mixed) and `set_global_seed` reproducibility across Python / NumPy / PyTorch |
| `src/train.py` | No (training logic) | Not yet | Full training run requires GPU + hours; intentionally excluded from CI |
| `src/train_lora.py` | No (training logic) | Not yet | Constants covered by `test_config.py::TestLoraConstants` |
| `src/evaluate.py` | No | Not yet | Parameterized replacement for `src/test_model.py`; tested manually against `results/` |

**Pytest entry point:** `tests/conftest.py` — adds `src/` to `sys.path` for flat-layout import compatibility (ADR-004).

---

## CI Integration

Tests run automatically on every push and pull request via `.github/workflows/ci.yml`.
CI test matrix: Python 3.11 and 3.12, runs on `ubuntu-latest`.
The `test` job depends on the `lint` job — lint must pass before tests run.

---

## Known Gaps

- **No end-to-end retraining test.** Full three-stage training requires a GPU and multiple hours.
  This is intentionally excluded from CI. Model quality is verified manually against `results/`.

- **No cross-dataset generalisation test.** The confirmed 81.16% Aegis result is from a manually
  run evaluation, not an automated test. Automating evaluation against a held-out set would
  require downloading the full Aegis dataset in CI, which is impractical.

- **Stage 2 SPML split uncertainty** — resolved in Phase 6. Both the native-validation and
  10%-fallback paths of `load_stage2` are covered by `tests/unit/test_data.py`.

- **LoRA training logic not tested.** `src/train_lora.py` adapter loading and PEFT integration
  have no smoke test. LoRA constants are verified in `test_config.py::TestLoraConstants` (skipped
  if peft absent). A smoke test requiring a real PEFT model is a future integration-test target.
