from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def load_split(path: Path) -> tuple[pd.Series, pd.Series]:
    """Load a split CSV and return text/features and labels."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    required_cols = ["text", "category"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Split file {path} is missing columns: {missing}")

    return df["text"].astype(str), df["category"].astype(str)


def build_pipeline(max_features: int, min_df: int, max_df: float) -> Pipeline:
    """Build baseline TF-IDF + Logistic Regression text classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="auto",
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, x: pd.Series, y: pd.Series) -> dict:
    """Compute common multiclass classification metrics."""
    preds = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "report": classification_report(y, preds, output_dict=True, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF + Logistic Regression model.")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/models/baseline_tfidf_logreg.joblib"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/models/baseline_metrics.json"),
    )
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.95)

    args = parser.parse_args()

    x_train, y_train = load_split(args.splits_dir / "train.csv")
    x_val, y_val = load_split(args.splits_dir / "val.csv")
    x_test, y_test = load_split(args.splits_dir / "test.csv")

    model = build_pipeline(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    model.fit(x_train, y_train)

    val_metrics = evaluate(model, x_val, y_val)
    test_metrics = evaluate(model, x_test, y_test)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.model_output)

    metrics = {
        "model": "tfidf_logistic_regression_baseline",
        "train_size": int(len(x_train)),
        "val_size": int(len(x_val)),
        "test_size": int(len(x_test)),
        "params": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
        },
        "validation": val_metrics,
        "test": test_metrics,
    }

    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Model saved to: {args.model_output}")
    print(f"Metrics saved to: {args.metrics_output}")
    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation f1_macro: {val_metrics['f1_macro']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test f1_macro: {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
