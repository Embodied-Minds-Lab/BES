#!/usr/bin/env python3
"""
Build the hard-split MuSiQue parquets used by all 6 release training runs:

  TRAIN  hard_train.parquet    musique rows with hop in {3,4}  (~5562 rows)
  TEST   hard_test.parquet     full official musique validation set
                                (bamboogle rows dropped)

Both files set data_source="musique"; we do NOT split by hop or keep
bamboogle, so verl reports a single val/test_score/musique panel.

Hop is read from metadata['question_decomposition'] length.
"""
import argparse
import os
from collections import Counter

import pandas as pd


def hop_of(row) -> int | None:
    md = row.get("metadata")
    if not isinstance(md, dict):
        return None
    qd = md.get("question_decomposition")
    if qd is None:
        return None
    try:
        return len(qd)
    except TypeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir",
        required=True,
        help="dir containing train.parquet + test.parquet",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="output dir (default: same as src_dir)",
    )
    ap.add_argument(
        "--train_hops",
        type=int,
        nargs="+",
        default=[3, 4],
        help="hop counts to keep in train (default: 3 4)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or args.src_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- TRAIN ----------------
    src_train = os.path.join(args.src_dir, "train.parquet")
    print(f"=== TRAIN  read {src_train}")
    df = pd.read_parquet(src_train)
    print(f"  rows: {len(df)}")
    df["_hop"] = df.apply(hop_of, axis=1)
    print(f"  hop dist: {Counter(df['_hop'])}")

    keep = df["_hop"].isin(args.train_hops)
    hard = df[keep].copy()
    print(f"  keep hop in {args.train_hops}: {len(hard)} rows")

    hard["data_source"] = "musique"
    hard = hard.drop(columns=["_hop"]).reset_index(drop=True)
    out_train = os.path.join(out_dir, "hard_train.parquet")
    hard.to_parquet(out_train, index=False)
    print(f"  wrote {out_train}  ({os.path.getsize(out_train)//1024} KB)")

    # ---------------- TEST ----------------
    src_test = os.path.join(args.src_dir, "test.parquet")
    print(f"\n=== TEST   read {src_test}")
    tdf = pd.read_parquet(src_test)
    print(f"  rows: {len(tdf)}")
    print(f"  data_source dist (orig): {Counter(tdf['data_source'])}")

    tdf = tdf[tdf["data_source"] == "musique"].copy()
    tdf["data_source"] = "musique"
    tdf = tdf.reset_index(drop=True)
    print(f"  rows kept (musique only): {len(tdf)}")

    out_test = os.path.join(out_dir, "hard_test.parquet")
    tdf.to_parquet(out_test, index=False)
    print(f"  wrote {out_test}  ({os.path.getsize(out_test)//1024} KB)")


if __name__ == "__main__":
    main()
