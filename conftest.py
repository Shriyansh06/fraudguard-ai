"""
conftest.py ? pytest configuration for FraudGuard AI.

Ensures the project root is on sys.path so all tests can import from src/.
"""

import sys
from pathlib import Path

# Add the project root (directory containing this conftest.py) to sys.path
# This allows `from src.config import ...` to work in all test files.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
