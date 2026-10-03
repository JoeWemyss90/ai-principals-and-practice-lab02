#!/usr/bin/env python3
"""
Check that this machine can run the Week 2 lab.

    python setup_check.py

Prints a report and exits 0 if everything needed is present. make_submission.py
runs it for you and puts the output in your zip, so a marker can see what your
environment looked like. Standard library only for the checks themselves.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "UCI_Credit_Card.csv"

REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
]
OPTIONAL = [
    ("scipy", "scipy"),
    ("seaborn", "seaborn"),
    ("jupyterlab", "jupyterlab"),
]

problems: list[str] = []


def line(label: str, status: str, detail: str = "") -> None:
    print(f"  {label:<22} {status:<8} {detail}".rstrip())


def check_python() -> None:
    print("\nPython")
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 9)
    line("version", "OK" if ok else "TOO OLD", sys.version.split()[0])
    if not ok:
        problems.append("Python 3.9 or newer is needed; 3.11+ is what the module assumes.")
    line("executable", "", sys.executable)
    line("platform", "", platform.platform())

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    line("virtual env", "YES" if in_venv else "NO",
         sys.prefix if in_venv else "using the system Python")
    if not in_venv:
        print("      Not fatal, but a per-module venv is what the setup note asks for.")


def check_packages() -> None:
    print("\nPackages")
    for module, pip_name in REQUIRED:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "installed")
            line(pip_name, "OK", version)
        except Exception:
            line(pip_name, "MISSING", f"pip install {pip_name}")
            problems.append(f"{pip_name} is missing — pip install {pip_name}")

    for module, pip_name in OPTIONAL:
        try:
            mod = importlib.import_module(module)
            line(pip_name, "OK", getattr(mod, "__version__", "installed") + "  (optional)")
        except Exception:
            line(pip_name, "-", "optional, not installed")


def check_dataset() -> None:
    print("\nDataset")
    if not CSV.exists():
        line("UCI_Credit_Card.csv", "MISSING", f"expected in {HERE}")
        problems.append(
            "UCI_Credit_Card.csv is not in this folder. It is in the Week 02 Lab "
            "Files archive on the course site."
        )
        return

    line("UCI_Credit_Card.csv", "FOUND", f"{CSV.stat().st_size / 1e6:.1f} MB")
    try:
        import pandas as pd

        df = pd.read_csv(CSV)
        shape_ok = df.shape == (30000, 25)
        line("shape", "OK" if shape_ok else "ODD", str(df.shape) + "  (expected (30000, 25))")
        if not shape_ok:
            problems.append(f"The CSV loaded as {df.shape}, not (30000, 25). Wrong file?")

        target = "default.payment.next.month"
        if target in df.columns:
            rate = round(float(df[target].mean()), 3)
            rate_ok = 0.21 <= rate <= 0.23
            line("default rate", "OK" if rate_ok else "ODD", f"{rate}  (expected ~0.221)")
            if not rate_ok:
                problems.append(f"Default rate came out at {rate}, expected about 0.221.")
        else:
            line("target column", "MISSING", target)
            problems.append(f"The column '{target}' is not in the file.")
    except Exception as exc:  # noqa: BLE001
        detail = repr(exc)
        if len(detail) > 300:                # a parser error can quote the whole file
            detail = detail[:300] + " ..."
        line("load", "FAILED", detail)
        problems.append(f"The CSV would not load: {detail}")


def main() -> int:
    print("=" * 68)
    print("  Environment check — AI Principles and Practice, Week 2")
    print("=" * 68)

    check_python()
    check_packages()
    check_dataset()

    print("\n" + "=" * 68)
    if problems:
        print(f"  {len(problems)} thing(s) to fix:\n")
        for p in problems:
            print(f"    - {p}")
        print("\n  Fix these before the lab. If you are stuck, post this whole")
        print("  output in the module channel rather than describing it.")
    else:
        print("  Everything needed is present. You are ready for the lab.")
    print("=" * 68)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
