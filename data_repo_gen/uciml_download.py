#!/usr/bin/env python3
"""
uciml_download.py — Fetch UCI ML datasets by ID and save to CSV.

Usage:
  data-repo-gen-fetch 109 159
  data-repo-gen-fetch 109 --outdir datasets
  data-repo-gen-fetch 109 159 --separate
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return re.sub(r"-{2,}", "-", s)


def _dedupe_columns(existing, new_cols):
    existing_set = set(existing)
    out, counts = [], {}
    for c in new_cols:
        if c not in existing_set and c not in out:
            out.append(c)
        else:
            counts[c] = counts.get(c, 0) + 1
            out.append(f"{c}_target{counts[c]}")
    return out


def fetch_and_write(dataset_id: int, outdir: Path, separate: bool) -> None:
    ds = fetch_ucirepo(id=int(dataset_id))
    name = ds.metadata.get("name", f"uci-{dataset_id}") or f"uci-{dataset_id}"
    slug = slugify(name)
    ds_dir = outdir / f"{dataset_id}_{slug}"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Data (Pandas DataFrames; may be None)
    X = pd.DataFrame(ds.data.features) if ds.data.features is not None else pd.DataFrame()
    y = pd.DataFrame(ds.data.targets) if ds.data.targets is not None else pd.DataFrame()

    # Save metadata + variables for transparency/FAIRness
    (ds_dir / "metadata.json").write_text(json.dumps(ds.metadata, indent=2), encoding="utf-8")
    pd.DataFrame(ds.variables).to_csv(ds_dir / "variables.csv", index=False)

    if separate:
        if not X.empty:
            X.to_csv(ds_dir / f"{dataset_id}_{slug}_features.csv", index=False)
        if not y.empty:
            y.to_csv(ds_dir / f"{dataset_id}_{slug}_targets.csv", index=False)
        print(f"[ok] {dataset_id}: wrote features/targets CSV + metadata to {ds_dir}")
    else:
        combined = X.copy()
        if not y.empty:
            y2 = y.copy()
            y2.columns = _dedupe_columns(combined.columns, y2.columns)
            combined = pd.concat([combined, y2], axis=1)
        combined.to_csv(ds_dir / f"{dataset_id}_{slug}.csv", index=False)
        print(f"[ok] {dataset_id}: wrote combined CSV + metadata to {ds_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Download UCI ML datasets by ID (via ucimlrepo) and save to CSV."
    )
    p.add_argument("ids", nargs="+", help="Dataset IDs (space-separated), e.g. 109 159")
    p.add_argument("--outdir", default="datasets", help='Output directory (default: "datasets")')
    p.add_argument(
        "--separate",
        action="store_true",
        help="Write features and targets to separate CSV files.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ds_id in args.ids:
        try:
            fetch_and_write(int(ds_id), outdir, args.separate)
        except Exception as e:
            print(f"[error] {ds_id}: {e}")
            print("Try disabling SSL authentication in the HTTPS request.")


if __name__ == "__main__":
    main()
