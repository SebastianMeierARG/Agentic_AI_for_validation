#!/usr/bin/env python
"""
json_to_xlsx.py
---------------
Converts one or more audit_results.json files to .xlsx, keeping the same name.

Edit the FILES list below to point to the JSON files you want to convert.

Usage:
  python json_to_xlsx.py
"""

import json
import pandas as pd
from pathlib import Path

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
FILES = [
    # Add as many paths as needed:
    "documents/BPN/outputs/20260326T134537Z/audit_results.json",
    "documents/UCI/outputs/20260326T141129Z/audit_results.json"
    "documents/CapitalFlow/outputs/20260326T140118Z/audit_results.json",
]
# ─────────────────────────────────────────────────────────────────────────────


def convert(json_path: str):
    src = Path(json_path)
    if not src.exists():
        print(f"  SKIP  {src} — file not found.")
        return

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        print(f"  SKIP  {src} — unexpected JSON structure.")
        return

    dest = src.with_suffix(".xlsx")
    df.to_excel(dest, index=False, engine="openpyxl")
    print(f"  OK    {dest}  ({len(df)} rows, {len(df.columns)} columns)")


def main():
    if not FILES:
        print("No files listed. Edit the FILES list in json_to_xlsx.py and re-run.")
        return

    print(f"Converting {len(FILES)} file(s)...\n")
    for path in FILES:
        convert(path)
    print("\nDone.")


if __name__ == "__main__":
    main()
