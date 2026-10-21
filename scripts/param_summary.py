"""Print parameter counts for T, D and P under the default config:  python scripts/param_summary.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import parameter_summary  # noqa: E402

if __name__ == "__main__":
    parameter_summary()
