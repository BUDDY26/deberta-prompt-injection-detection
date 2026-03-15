"""
Unit tests for src/utils.py.

Covered:
  compute_metrics  -- all-correct, all-wrong, and mixed prediction arrays
  set_global_seed  -- reproducibility of random draws across Python / NumPy / PyTorch

plot_training_metrics is not tested here: it requires a Trainer instance and
produces a matplotlib figure; visual output is verified manually during QA.
"""

import random

import numpy as np
import pytest
import torch

import utils


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class _EvalPred:
    """Minimal stand-in for transformers EvalPrediction."""

    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels

    def __iter__(self):
        # compute_metrics destructures via: logits, labels = eval_pred
        return iter((self.logits, self.labels))


def _make_logits(predictions):
    """
    Convert a list of 0/1 predictions into (N, 2) logit arrays.

    For prediction=1: logits = [0.0, 1.0]  → argmax gives 1
    For prediction=0: logits = [1.0, 0.0]  → argmax gives 0
    """
    rows = []
    for p in predictions:
        if p == 1:
            rows.append([0.0, 1.0])
        else:
            rows.append([1.0, 0.0])
    return np.array(rows, dtype=np.float32)


def test_compute_metrics_all_correct():
    labels = np.array([0, 1, 0, 1])
    logits = _make_logits(labels)
    result = utils.compute_metrics(_EvalPred(logits, labels))
    assert result["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_all_wrong():
    labels = np.array([0, 1, 0, 1])
    preds = np.array([1, 0, 1, 0])
    logits = _make_logits(preds)
    result = utils.compute_metrics(_EvalPred(logits, labels))
    assert result["accuracy"] == pytest.approx(0.0)


def test_compute_metrics_mixed():
    # 3 correct out of 4 → 0.75
    labels = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 0])  # last one wrong
    logits = _make_logits(preds)
    result = utils.compute_metrics(_EvalPred(logits, labels))
    assert result["accuracy"] == pytest.approx(0.75)


def test_compute_metrics_returns_dict():
    labels = np.array([0, 1])
    logits = _make_logits([0, 1])
    result = utils.compute_metrics(_EvalPred(logits, labels))
    assert isinstance(result, dict)
    assert "accuracy" in result


# ---------------------------------------------------------------------------
# set_global_seed — reproducibility
# ---------------------------------------------------------------------------


def test_set_global_seed_python_random():
    utils.set_global_seed(42)
    first = [random.random() for _ in range(5)]
    utils.set_global_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_global_seed_numpy():
    utils.set_global_seed(42)
    first = np.random.rand(5).tolist()
    utils.set_global_seed(42)
    second = np.random.rand(5).tolist()
    assert first == pytest.approx(second)


def test_set_global_seed_torch():
    utils.set_global_seed(42)
    first = torch.rand(5).tolist()
    utils.set_global_seed(42)
    second = torch.rand(5).tolist()
    assert first == pytest.approx(second)


def test_set_global_seed_different_seeds_differ():
    utils.set_global_seed(1)
    draw_1 = np.random.rand(5).tolist()
    utils.set_global_seed(2)
    draw_2 = np.random.rand(5).tolist()
    assert draw_1 != draw_2
