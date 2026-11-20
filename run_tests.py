"""Run all unit/integration tests in one command: python run_tests.py"""
import sys
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "tests", *sys.argv[1:]]))
