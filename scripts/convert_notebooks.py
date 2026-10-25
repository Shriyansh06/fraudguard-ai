"""
FraudGuard AI ? Convert notebook .py files to .ipynb format.

Usage:
    python scripts/convert_notebooks.py

Requires: jupytext
    pip install jupytext

Converts all notebooks/**.py (percent format) to .ipynb files
so they open natively in Jupyter Notebook / JupyterLab.
"""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def main():
    py_files = sorted(NOTEBOOKS_DIR.glob("*.py"))
    if not py_files:
        print("No .py notebook files found in notebooks/")
        return

    print(f"Found {len(py_files)} notebook files to convert.\n")

    try:
        subprocess.run(["jupytext", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "jupytext is not installed. Install it with:\n"
            "    pip install jupytext\n"
            "Then re-run this script."
        )
        sys.exit(1)

    for py_file in py_files:
        ipynb_file = py_file.with_suffix(".ipynb")
        print(f"Converting: {py_file.name} -> {ipynb_file.name}")
        result = subprocess.run(
            ["jupytext", "--to", "notebook", str(py_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  [OK] {ipynb_file.name}")
        else:
            print(f"  [FAIL] {result.stderr.strip()}")

    print(f"\nConversion complete. .ipynb files are in: {NOTEBOOKS_DIR}")
    print("Open with: jupyter notebook notebooks/")


if __name__ == "__main__":
    main()
