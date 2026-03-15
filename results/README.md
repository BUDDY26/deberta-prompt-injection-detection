# Results

This directory contains evaluation output from the DeBERTa prompt injection detection pipeline.

---

## File Inventory

| File | Source | Description |
|------|--------|-------------|
| `test_results_2dataset.txt` | `src/test_model.py` (legacy) | Evaluation on Safe-Guard + SPML datasets; model trained on two stages |
| `test_results_3datasets.txt` | `src/test_model.py` (legacy) | Evaluation on Safe-Guard + SPML + Aegis; confirms 81.16% Aegis accuracy |
| `evaluation/` | `src/evaluate.py` (canonical) | Deterministic evaluation outputs — created on first `evaluate.py` run |

---

## Legacy vs. Canonical Result Files

### Legacy (`test_results_*.txt`)

These files were produced by `src/test_model.py`, the original monolithic evaluation script
retained from the development notebook. They record the confirmed cross-dataset result:

- **Stage 2 model on Aegis test set:** 41.60% accuracy, F1 0.3053
- **Stage 3 model on Aegis test set:** 81.16% accuracy, F1 0.8255

These files are authoritative evidence — do not delete or modify them.
See `docs/evidence-ledger.md` §8 for full details.

### Canonical (`evaluation/`)

New evaluation runs via `src/evaluate.py` write structured JSON and text output to
`results/evaluation/`. This directory is created on the first `evaluate.py` run.

```bash
python src/evaluate.py --model-path models/deberta-pi-lora-final-adapter --dataset aegis
```

---

## Key Confirmed Result

The stage 3 full fine-tuning model achieves **81.16% accuracy / F1 0.8255** on the
`nvidia/Aegis-AI-Content-Safety-Dataset-2.0` test set (cross-dataset evaluation).

Source: `test_results_3datasets.txt` and `docs/evidence-ledger.md` §8.
