"""
Aggregate runner for LUTorch tests.

Delegates to pytest to run all test_*.py files in this directory.
Usage:
    python run_tests_with_different_seeds.py
or directly via pytest:
    pytest src/spiky/lutorch/tests/ -v
"""
import os
import sys


def main() -> int:
    import pytest
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    return pytest.main([tests_dir, "-v", "-s"])


if __name__ == "__main__":
    raise SystemExit(main())
