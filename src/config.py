"""
Centralized configuration for the full fine-tuning pipeline.

All values are sourced directly from docs/evidence-ledger.md.
No value in this file is invented or assumed — every constant has a confirmed source.

LoRA-specific constants (hyperparameters, adapter settings, output paths) are defined
locally in src/train_lora.py. The shared constants below (base model, tokenizer,
TrainingArguments defaults, plots directory) are reused by both pipelines.
"""

# ---------------------------------------------------------------------------
# Base model  (evidence-ledger §2)
# ---------------------------------------------------------------------------
BASE_MODEL = "ProtectAI/deberta-v3-base-prompt-injection"

# ---------------------------------------------------------------------------
# Dataset identifiers  (evidence-ledger §4; ordering fixed by ADR-003)
# ---------------------------------------------------------------------------
STAGE1_DATASET = "xTRam1/safe-guard-prompt-injection"
STAGE2_DATASET = "reshabhs/SPML_Chatbot_Prompt_Injection"
STAGE3_DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"

# ---------------------------------------------------------------------------
# Tokenization  (evidence-ledger §7, confirmed identical across all three stages)
# ---------------------------------------------------------------------------
MAX_LENGTH = 256
TOKENIZER_PADDING = "max_length"
TOKENIZER_TRUNCATION = True

# ---------------------------------------------------------------------------
# Full fine-tuning — output directory names  (evidence-ledger §3)
# ---------------------------------------------------------------------------
STAGE1_OUTPUT_DIR = "deberta-pi-full-stage1"
STAGE1_FINAL_DIR = "deberta-pi-full-stage1-final"

STAGE2_OUTPUT_DIR = "deberta-pi-full-stage2"
STAGE2_FINAL_DIR = "deberta-pi-full-final"  # stage 3 reloads from this path

STAGE3_OUTPUT_DIR = "deberta-pi-full-stage3"
STAGE3_FINAL_DIR = "deberta-pi-full-stage3-final"

PLOTS_DIR = "training_plots"

# ---------------------------------------------------------------------------
# Full fine-tuning — per-stage hyperparameters  (evidence-ledger §3)
# ---------------------------------------------------------------------------

# Stage 1: xTRam1/safe-guard-prompt-injection
STAGE1_LR = 2e-5
STAGE1_TRAIN_BATCH = 8
STAGE1_EVAL_BATCH = 16
STAGE1_EPOCHS = 10
STAGE1_PATIENCE = 3

# Stage 2: reshabhs/SPML_Chatbot_Prompt_Injection
STAGE2_LR = 2e-5
STAGE2_TRAIN_BATCH = 8
STAGE2_EVAL_BATCH = 16
STAGE2_EPOCHS = 15
STAGE2_PATIENCE = 3

# Stage 3: nvidia/Aegis-AI-Content-Safety-Dataset-2.0
STAGE3_LR = 2e-5
STAGE3_TRAIN_BATCH = 8
STAGE3_EVAL_BATCH = 16
STAGE3_EPOCHS = 25
STAGE3_PATIENCE = 3

# Common TrainingArguments settings — identical across all three stages
# (evidence-ledger §3)
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "accuracy"
GREATER_IS_BETTER = True
SAVE_TOTAL_LIMIT = 1
LOGGING_STEPS = 50
REPORT_TO = "none"
