"""
pytest configuration for the deberta-prompt-injection-detection test suite.

Adds src/ to sys.path so that all test files can import config, data, utils,
and evaluate without requiring PYTHONPATH=src to be set in the environment.
Sourced from ADR-004 (flat src/ layout) and implementation-plan §6.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
