#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml


def describe_split(df: pd.DataFrame, name: str, class_col: str | None, top_k: int):
    print(f"=== {name} Split ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    if "original_index" in df.columns:
        print(
            f"original_index coverage: {df['original_index'].notna().mean() * 100:.1f}% "
            f"({df['original_index'].nunique():,} unique)"
        )
    else:
        print("original_index coverage: column missing")

    if class_col:
        if class_col in df.columns:
            counts = df[class_col].value_counts(dropna=False)
            pct = df[class_col].value_counts(normalize=True, dropna=False)
            print(f"Top {top_k} '{class_col}' classes:")
            for cls in counts.head(top_k).index:
                print(f"  {cls}: {counts[cls]:,} ({pct[cls] * 100:.2f}%)")
        else:
            print(f"Column '{class_col}' not found for class statistics")

    print()


def describe_maps(maps_path: Path):
    print("=== Maps ===")
    maps = joblib.load(maps_path)
    keys_of_interest = [
        "family_id2idx",
        "major_id2idx",
        "leaf_id2idx",
        "config",
    ]
    for key in keys_of_interest:
        if key in maps:
            value = maps[key]
            if isinstance(value, dict) and key != "config":
                print(f"{key}: {len(value):,} entries")
            else:
                print(f"{key}: {json.dumps(value)[:200]}{'...' if len(json.dumps(value)) > 200 else ''}")
        else:
            print(f"{key}: missing")
    print()


def load_indices_mapping(indices_path: Path):
    try:
        return joblib.load(indices_path)
    except Exception:
        with open(indices_path, "r") as fh:
            return json.load(fh)


def describe_indices(indices_path: Path):
    print("=== Indices Mapping ===")
    mapping = load_indices_mapping(indices_path)
    for split_name, indices in mapping.items():
        print(f"{split_name}: {len(indices):,} entries (unique: {len(set(indices)):,})")
    print()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # Let pandas infer the delimiter (handles both comma and semicolon cases)
    df = pd.read_csv(path, sep=None, engine="python")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect dataset split artifacts.")
    parser.add_argument(
        "--input-params",
        required=True,
        type=Path,
        help="Path to the input_params.yaml file used by the split job",
    )
    parser.add_argument(
        "--class-col",
        default="leaf_id",
        help="Column to use for class distribution statistics (default: leaf_id)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top classes to display (default: 10)")
    return parser.parse_args()


def load_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"input_params file not found: {path}")
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def resolve_indices_path(params: dict) -> Path | None:
    candidates = [
        params.get("dataset_indices_map_path"),
        params.get("paths", {}).get("indices_map") if isinstance(params.get("paths"), dict) else None,
    ]
    for candidate in candidates:
        if candidate:
            return Path(candidate)

    train_path = params.get("dataset_train_path")
    if train_path:
        base = Path(train_path)
        return base.with_name(f"{base.stem}_indices.joblib")
    return None


def main():
    args = parse_args()
    params = load_params(args.input_params)

    try:
        dataset_path = Path(params["dataset_path"])
        train_path = Path(params["dataset_train_path"])
        caltest_path = Path(params["dataset_caltest_path"])
        maps_path = Path(params["maps_path"])
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Required parameter '{missing}' missing from {args.input_params}") from exc

    df_raw = load_dataframe(dataset_path)
    df_train = load_dataframe(train_path)
    df_caltest = load_dataframe(caltest_path)
    df_pre_split = pd.concat([df_train, df_caltest], ignore_index=True)

    describe_split(df_raw, "Raw Dataset (input)", None, args.top_k)
    describe_split(df_pre_split, "Preprocessed Dataset (before split)", args.class_col, args.top_k)
    describe_split(df_train, "Train", args.class_col, args.top_k)
    describe_split(df_caltest, "Cal/Test", args.class_col, args.top_k)

    describe_maps(maps_path)

    indices_path = resolve_indices_path(params)
    if indices_path:
        if indices_path.exists():
            try:
                describe_indices(indices_path)
            except Exception as exc:
                print(f"Failed to load index mapping from {indices_path}: {exc}")
        else:
            print(f"Indices mapping file not found: {indices_path}")


if __name__ == "__main__":
    main()

