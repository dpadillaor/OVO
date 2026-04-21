"""
Analyze fusion decisions from a fusion_decisions.csv produced by OVO.

Usage:
    python scripts/analyze_fusion_decisions.py <path/to/fusion_decisions.csv>

    # Filter to a specific pair
    python scripts/analyze_fusion_decisions.py <csv> --i1 84 --i2 130

    # Show only accepted fusions
    python scripts/analyze_fusion_decisions.py <csv> --result ACCEPTED
"""

import argparse
import pandas as pd


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["centroid_dist"] = pd.to_numeric(df["centroid_dist"], errors="coerce")
    df["cos_sim"] = pd.to_numeric(df["cos_sim"], errors="coerce")
    df["p_dist"] = pd.to_numeric(df["p_dist"], errors="coerce")
    return df


def summary(df: pd.DataFrame):
    total = len(df)
    accepted = (df["result"] == "ACCEPTED").sum()
    print(f"\n=== Summary ===")
    print(f"Total pairs evaluated : {total}")
    print(f"Accepted (fused)      : {accepted}")
    print(f"Rejected              : {total - accepted}")
    print(f"\nRejection breakdown:")
    print(df[df["result"] == "REJECTED"]["reason"].value_counts().to_string())
    print(f"\nFrames with update_map: {df['frame_id'].nunique()}")


def show(df: pd.DataFrame, args):
    mask = pd.Series([True] * len(df))

    if args.i1 is not None:
        mask &= (df["i1"] == args.i1) | (df["i2"] == args.i1)
    if args.i2 is not None:
        mask &= (df["i1"] == args.i2) | (df["i2"] == args.i2)
    if args.result:
        mask &= df["result"] == args.result.upper()
    if args.frame is not None:
        mask &= df["frame_id"] == args.frame
    if args.reason:
        mask &= df["reason"] == args.reason

    filtered = df[mask]
    if filtered.empty:
        print("No rows match the filter.")
        return

    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"\n{len(filtered)} rows:\n")
    print(filtered.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to fusion_decisions.csv")
    parser.add_argument("--i1", type=int, default=None, help="Filter by instance ID (either side)")
    parser.add_argument("--i2", type=int, default=None, help="Filter by second instance ID")
    parser.add_argument("--result", type=str, default=None, help="ACCEPTED or REJECTED")
    parser.add_argument("--reason", type=str, default=None, help="centroid | cos_sim | overlap")
    parser.add_argument("--frame", type=int, default=None, help="Filter by frame_id")
    args = parser.parse_args()

    df = load(args.csv)
    summary(df)

    has_filter = any([args.i1, args.i2, args.result, args.reason, args.frame])
    if has_filter:
        print(f"\n=== Filtered rows ===")
        show(df, args)


if __name__ == "__main__":
    main()
