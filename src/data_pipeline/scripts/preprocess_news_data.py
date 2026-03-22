from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Normalize text by lowercasing and collapsing extra whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_dataset(input_path: Path) -> pd.DataFrame:
    """Load raw JSONL news data and build a cleaned text classification frame."""
    df = pd.read_json(input_path, lines=True)

    required_cols = ["headline", "short_description", "category"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.dropna(subset=required_cols)

    df["headline"] = df["headline"].astype(str)
    df["short_description"] = df["short_description"].astype(str)
    df["category"] = df["category"].astype(str)

    df["text"] = (df["headline"] + " " + df["short_description"]).map(clean_text)

    # Keep only non-empty text/category rows.
    df = df[(df["text"].str.len() > 0) & (df["category"].str.len() > 0)]

    return df[["text", "category"]].reset_index(drop=True)


def make_splits(df: pd.DataFrame, test_size: float, val_size: float, random_state: int):
    """Create stratified train/val/test splits from the processed dataframe."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["category"],
    )

    val_ratio_from_train = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio_from_train,
        random_state=random_state,
        stratify=train_df["category"],
    )

    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess news dataset and create splits.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/News_Category_Dataset_v3.json"),
        help="Path to raw JSONL dataset",
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=Path("data/processed/news_processed.csv"),
        help="Path to save cleaned full dataset CSV",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory to save train/val/test CSV files",
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    if args.test_size <= 0 or args.val_size <= 0 or (args.test_size + args.val_size) >= 1:
        raise ValueError("test-size and val-size must be > 0 and sum to < 1")

    args.processed_output.parent.mkdir(parents=True, exist_ok=True)
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    df = preprocess_dataset(args.input)
    train_df, val_df, test_df = make_splits(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    df.to_csv(args.processed_output, index=False)
    train_df.to_csv(args.splits_dir / "train.csv", index=False)
    val_df.to_csv(args.splits_dir / "val.csv", index=False)
    test_df.to_csv(args.splits_dir / "test.csv", index=False)

    print(f"Processed rows: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Val rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Saved full processed data to: {args.processed_output}")
    print(f"Saved split files to: {args.splits_dir}")


if __name__ == "__main__":
    main()
