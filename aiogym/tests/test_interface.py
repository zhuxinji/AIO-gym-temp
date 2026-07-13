#!/usr/bin/env python3
"""Unified direct runner for the split AIO-Gym interface test modules."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_contract import run_all as run_env_tests
from test_evaluation_contract import run_all as run_evaluation_tests
from test_models_api import run_all as run_model_tests


def main():
    run_model_tests()
    run_env_tests()
    run_evaluation_tests()
    print("\nALL INTERFACE TESTS PASS OK")


if __name__ == "__main__":
    main()
