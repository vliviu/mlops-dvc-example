#!/usr/bin/env python3
"""
featurize.py - split dataset and ensure headers normalized
Usage:
  python src/featurize.py <in_csv> <out_train_csv> <out_test_csv> <test_size> <random_state>
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main(in_csv, out_train, out_test, test_size, random_state):
    df = pd.read_csv(in_csv, engine='python')
    # If header looks like single giant string, try to re-run prepare to normalize
    if len(df.columns) == 1 and ';' in df.columns[0]:
        # fallback: try to split header manually
        raw_header = df.columns[0]
        # attempt to parse with semicolon delimiter
        df = pd.read_csv(in_csv, sep=';', engine='python')
    # normalize columns: strip quotes and replace spaces with underscores
    df.columns = [str(c).strip().strip('"').strip("'").replace(' ', '_').replace(';','_') for c in df.columns]
    test_size = float(test_size)
    random_state = int(random_state)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(out_test).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_train, index=False)
    test.to_csv(out_test, index=False)
    print(f"[OK] Wrote train ({len(train)}) -> {out_train} and test ({len(test)}) -> {out_test}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python src/featurize.py in.csv out_train.csv out_test.csv test_size random_state")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
