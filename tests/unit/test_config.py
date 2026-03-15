"""
Unit tests for src/config.py and LoRA constants in src/train_lora.py.

All expected values are sourced from docs/evidence-ledger.md.
Tests assert the confirmed constants rather than importing magic numbers,
so a silent edit to a hyperparameter will cause a test failure.

LoRA constant tests are grouped in TestLoraConstants and skipped when peft
is not installed. Non-LoRA tests always run regardless of peft availability.
"""

import importlib

import pytest

import config

_peft_available = importlib.util.find_spec("peft") is not None


# ---------------------------------------------------------------------------
# Base model and dataset identifiers  (evidence-ledger §2, §4)
# ---------------------------------------------------------------------------


def test_base_model():
    assert config.BASE_MODEL == "ProtectAI/deberta-v3-base-prompt-injection"


def test_stage1_dataset():
    assert config.STAGE1_DATASET == "xTRam1/safe-guard-prompt-injection"


def test_stage2_dataset():
    assert config.STAGE2_DATASET == "reshabhs/SPML_Chatbot_Prompt_Injection"


def test_stage3_dataset():
    assert config.STAGE3_DATASET == "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"


# ---------------------------------------------------------------------------
# Tokenization  (evidence-ledger §7)
# ---------------------------------------------------------------------------


def test_max_length():
    assert config.MAX_LENGTH == 256


def test_tokenizer_padding():
    assert config.TOKENIZER_PADDING == "max_length"


def test_tokenizer_truncation():
    assert config.TOKENIZER_TRUNCATION is True


# ---------------------------------------------------------------------------
# Full FT learning rates — all three stages use 2e-5  (evidence-ledger §3)
# ---------------------------------------------------------------------------


def test_stage1_lr():
    assert config.STAGE1_LR == 2e-5


def test_stage2_lr():
    assert config.STAGE2_LR == 2e-5


def test_stage3_lr():
    assert config.STAGE3_LR == 2e-5


# ---------------------------------------------------------------------------
# Full FT epoch counts  (evidence-ledger §3)
# ---------------------------------------------------------------------------


def test_stage1_epochs():
    assert config.STAGE1_EPOCHS == 10


def test_stage2_epochs():
    assert config.STAGE2_EPOCHS == 15


def test_stage3_epochs():
    assert config.STAGE3_EPOCHS == 25


# ---------------------------------------------------------------------------
# Full FT early stopping patience — all stages use 3  (evidence-ledger §3)
# ---------------------------------------------------------------------------


def test_stage1_patience():
    assert config.STAGE1_PATIENCE == 3


def test_stage2_patience():
    assert config.STAGE2_PATIENCE == 3


def test_stage3_patience():
    assert config.STAGE3_PATIENCE == 3


# ---------------------------------------------------------------------------
# Full FT output directory names  (evidence-ledger §3)
# ---------------------------------------------------------------------------


def test_stage1_output_dir():
    assert config.STAGE1_OUTPUT_DIR == "deberta-pi-full-stage1"


def test_stage1_final_dir():
    assert config.STAGE1_FINAL_DIR == "deberta-pi-full-stage1-final"


def test_stage2_output_dir():
    assert config.STAGE2_OUTPUT_DIR == "deberta-pi-full-stage2"


def test_stage2_final_dir():
    # Stage 3 reloads from this path (comment in config.py)
    assert config.STAGE2_FINAL_DIR == "deberta-pi-full-final"


def test_stage3_output_dir():
    assert config.STAGE3_OUTPUT_DIR == "deberta-pi-full-stage3"


def test_stage3_final_dir():
    assert config.STAGE3_FINAL_DIR == "deberta-pi-full-stage3-final"


# ---------------------------------------------------------------------------
# Common TrainingArguments defaults  (evidence-ledger §3)
# ---------------------------------------------------------------------------


def test_eval_strategy():
    assert config.EVAL_STRATEGY == "epoch"


def test_save_strategy():
    assert config.SAVE_STRATEGY == "epoch"


def test_load_best_model_at_end():
    assert config.LOAD_BEST_MODEL_AT_END is True


def test_metric_for_best_model():
    assert config.METRIC_FOR_BEST_MODEL == "accuracy"


def test_greater_is_better():
    assert config.GREATER_IS_BETTER is True


def test_save_total_limit():
    assert config.SAVE_TOTAL_LIMIT == 1


def test_report_to():
    assert config.REPORT_TO == "none"


# ---------------------------------------------------------------------------
# LoRA constants from src/train_lora.py  (evidence-ledger §3, §7; ADR-006)
#
# peft is required at import time by train_lora. _peft_available is checked
# at collection time; the entire class is skipped when peft is absent so the
# non-LoRA tests above always run regardless of the environment.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _peft_available, reason="peft not installed; skipping LoRA constant tests"
)
class TestLoraConstants:
    """LoRA constant assertions sourced from evidence-ledger §3, §7, and ADR-006."""

    def setup_method(self):
        import train_lora  # safe: only reached when _peft_available is True

        self.tl = train_lora

    def test_lora_r(self):
        assert self.tl.LORA_R == 16

    def test_lora_alpha(self):
        assert self.tl.LORA_ALPHA == 32

    def test_lora_dropout(self):
        assert self.tl.LORA_DROPOUT == 0.1

    def test_lora_target_modules(self):
        assert self.tl.LORA_TARGET_MODULES == [
            "query_proj",
            "key_proj",
            "value_proj",
            "o_proj",
        ]

    def test_lora_modules_to_save(self):
        assert self.tl.LORA_MODULES_TO_SAVE == ["classifier", "score"]

    def test_lora_bias(self):
        assert self.tl.LORA_BIAS == "none"

    def test_lora_lr(self):
        # LoRA uses 2e-4, higher than full FT (2e-5) — evidence-ledger §3
        assert self.tl.LORA_LR == 2e-4

    def test_lora_train_batch(self):
        assert self.tl.LORA_TRAIN_BATCH == 16

    def test_lora_eval_batch(self):
        assert self.tl.LORA_EVAL_BATCH == 32

    def test_lora_stage1_epochs(self):
        # Confirmed 10 from trainer_state.json; notebook was edited to 20 post-training (ADR-006)
        assert self.tl.LORA_STAGE1_EPOCHS == 10

    def test_lora_stage2_epochs(self):
        assert self.tl.LORA_STAGE2_EPOCHS == 10

    def test_lora_patience(self):
        assert self.tl.LORA_PATIENCE == 3

    def test_lora_stage1_output_dir(self):
        assert self.tl.LORA_STAGE1_OUTPUT_DIR == "deberta-pi-lora-stage1"

    def test_lora_stage1_final_dir(self):
        assert self.tl.LORA_STAGE1_FINAL_DIR == "deberta-pi-lora-stage1-final"

    def test_lora_stage2_output_dir(self):
        assert self.tl.LORA_STAGE2_OUTPUT_DIR == "deberta-pi-lora-stage2"

    def test_lora_final_adapter_dir(self):
        assert self.tl.LORA_FINAL_ADAPTER_DIR == "deberta-pi-lora-final-adapter"

    def test_lora_final_full_dir(self):
        assert self.tl.LORA_FINAL_FULL_DIR == "deberta-pi-lora-final-full"
