#!/usr/bin/env python3
"""
prepare.py - robust CSV copy + header normalization
Usage:
  python src/prepare.py <raw_csv> <out_csv>
This script:
 - auto-detects delimiter (',' or ';')
 - normalizes column names (strip quotes, replace spaces with '_')
 - writes a cleaned CSV to out_csv
"""
import sys
from pathlib import Path
import pandas as pd
import csv

def detect_delim(path, nlines=2):
    # simple heuristic: sample first line(s) and check for ';' vs ','
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = ''.join([next(f) for _ in range(nlines)])
    # count semicolons / commas on first line
    first_line = sample.splitlines()[0] if sample else ''
    if first_line.count(';') > first_line.count(','):
        return ';'
    return ','

def clean_columns(df):
    new_cols = []
    for c in df.columns:
        c2 = str(c).strip()
        # remove surrounding quotes
        if (c2.startswith('"') and c2.endswith('"')) or (c2.startswith("'") and c2.endswith("'")):
            c2 = c2[1:-1]
        # replace spaces and semicolons with underscore, remove extra quotes
        c2 = c2.replace(';', '_').replace(' ', '_').replace('-', '_')
        # collapse multiple underscores
        while '__' in c2:
            c2 = c2.replace('__','_')
        c2 = c2.strip().strip('_')
        new_cols.append(c2)
    df.columns = new_cols
    return df

def main(src, out):
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"{src} not found")
    delim = detect_delim(src)
    # use engine python to be tolerant with quoting
    df = pd.read_csv(src, sep=delim, engine='python')
    df = clean_columns(df)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] Wrote cleaned CSV to {out} (detected sep='{delim}')")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/prepare.py raw.csv out.csv")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
