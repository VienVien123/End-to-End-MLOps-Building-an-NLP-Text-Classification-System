from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "train_baseline_tfidf_logreg")


def log(message: str) -> None:
    """Simple console logger with timestamp."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def configure_mlflow(tracking_uri: str, experiment_name: str):
    """Configure MLflow tracking server and enable sklearn autologging."""
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed. Install with `pip install mlflow` or use --disable-mlflow."
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)

    # Fail fast for remote tracking servers if they are unreachable.
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        try:
            MlflowClient().search_experiments(max_results=1)
        except Exception as exc:
            raise RuntimeError(
                "Cannot connect to MLflow Tracking Server. "
                f"Please check server status and MLFLOW_TRACKING_URI: {tracking_uri}"
            ) from exc

    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_models=True, silent=True)
    return mlflow


def load_split(path: Path) -> tuple[pd.Series, pd.Series]:
    """Load a split CSV and return text and labels."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)

    required_cols = ["text", "category"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Split file {path} is missing columns: {missing}")

    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].astype(str)
    df["category"] = df["category"].astype(str)

    return df["text"], df["category"]


def apply_sample(
    x: pd.Series,
    y: pd.Series,
    sample_size: int | None,
    random_state: int,
) -> tuple[pd.Series, pd.Series]:
    """Optionally sample a subset for faster debugging."""
    if sample_size is None or sample_size <= 0 or sample_size >= len(x):
        return x, y

    sampled_idx = x.sample(n=sample_size, random_state=random_state).index
    return x.loc[sampled_idx], y.loc[sampled_idx]


def build_pipeline(
    max_features: int,
    min_df: int,
    max_df: float,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Build TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
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
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=random_state,
                    n_jobs=None,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Compute multiclass classification metrics."""
    preds = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "report": classification_report(
            y,
            preds,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline TF-IDF + Logistic Regression model."
    )

    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "baseline_tfidf_logreg.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "baseline_metrics.json",
    )

    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for train split to debug faster.",
    )

    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="MLflow tracking URI. Prefer tracking server URI, e.g. http://127.0.0.1:5000.",
    )
    parser.add_argument("--mlflow-experiment", type=str, default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-run-name", type=str, default="tfidf-logreg-baseline")
    parser.add_argument("--disable-mlflow", action="store_true")

    args = parser.parse_args()

    log("Loading train/val/test splits...")
    x_train, y_train = load_split(args.splits_dir / "train.csv")
    x_val, y_val = load_split(args.splits_dir / "val.csv")
    x_test, y_test = load_split(args.splits_dir / "test.csv")

    log(f"Original sizes -> train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")

    x_train, y_train = apply_sample(
        x_train,
        y_train,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    if args.sample_size is not None:
        log(f"Sampled train size -> train: {len(x_train)}")

    log("Building TF-IDF + Logistic Regression pipeline...")
    model = build_pipeline(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    mlflow = None
    active_run = None

    if not args.disable_mlflow:
        log(f"Configuring MLflow with URI: {args.mlflow_uri}")
        mlflow = configure_mlflow(args.mlflow_uri, args.mlflow_experiment)

    training_start = time.time()

    if mlflow is not None:
        log("Starting MLflow run...")
        active_run = mlflow.start_run(run_name=args.mlflow_run_name)

    try:
        log("Training model...")
        model.fit(x_train, y_train)

        log("Evaluating on validation set...")
        val_metrics = evaluate(model, x_val, y_val)

        log("Evaluating on test set...")
        test_metrics = evaluate(model, x_test, y_test)

        training_duration_sec = time.time() - training_start

        ensure_parent_dir(args.model_output)
        ensure_parent_dir(args.metrics_output)

        log(f"Saving model to: {args.model_output}")
        joblib.dump(model, args.model_output)

        metrics = {
            "model": "tfidf_logistic_regression_baseline",
            "train_size": int(len(x_train)),
            "val_size": int(len(x_val)),
            "test_size": int(len(x_test)),
            "training_duration_sec": round(training_duration_sec, 4),
            "params": {
                "max_features": args.max_features,
                "min_df": args.min_df,
                "max_df": args.max_df,
                "max_iter": args.max_iter,
                "random_state": args.random_state,
                "sample_size": args.sample_size,
            },
            "validation": val_metrics,
            "test": test_metrics,
        }

        log(f"Saving metrics to: {args.metrics_output}")
        with args.metrics_output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        if mlflow is not None:
            log("Logging extra metrics/params/artifacts to MLflow...")
            mlflow.log_metrics(
                {
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1_macro": test_metrics["f1_macro"],
                    "training_duration_sec": training_duration_sec,
                }
            )

            mlflow.log_params(
                {
                    "split_train_size": int(len(x_train)),
                    "split_val_size": int(len(x_val)),
                    "split_test_size": int(len(x_test)),
                    "max_features": args.max_features,
                    "min_df": args.min_df,
                    "max_df": args.max_df,
                    "max_iter": args.max_iter,
                    "random_state": args.random_state,
                    "sample_size": args.sample_size if args.sample_size is not None else "full",
                }
            )

            mlflow.log_artifact(str(args.metrics_output))
            mlflow.log_artifact(str(args.model_output))

        log("Training completed successfully.")
        print()
        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model saved to:       {args.model_output}")
        print(f"Metrics saved to:     {args.metrics_output}")
        print(f"Validation accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"Validation f1_macro:  {val_metrics['f1_macro']:.4f}")
        print(f"Test accuracy:        {test_metrics['accuracy']:.4f}")
        print(f"Test f1_macro:        {test_metrics['f1_macro']:.4f}")
        print(f"Training time (sec):  {training_duration_sec:.2f}")

        if mlflow is not None and active_run is not None:
            print(f"MLflow run_id:        {active_run.info.run_id}")
            print(f"MLflow tracking URI:  {args.mlflow_uri}")

        print("=" * 60)

    finally:
        if mlflow is not None and active_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()