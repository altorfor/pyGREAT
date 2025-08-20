#!/usr/bin/env python
# scripts/run_simple.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make 'pygreat' importable when running directly from repo
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pygreat.pygreat import run_from_parfile


def main():
    ap = argparse.ArgumentParser(description="Run pyGREAT pipeline from a GREAT-style parameter file.")
    ap.add_argument("--parfile", required=True, help="Path to GREAT parameter file")
    ap.add_argument("--inspect-tree", action="store_true", help="Print a compact tree of the HDF5 outputs at the end")
    args = ap.parse_args()

    run_from_parfile(args.parfile, inspect_tree=args.inspect_tree)


if __name__ == "__main__":
    main()
