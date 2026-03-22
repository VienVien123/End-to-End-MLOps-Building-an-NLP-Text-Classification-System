from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "train_baseline_tfidf_logreg")
DEFAULT_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "news_text_classifier")


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def configure_mlflow(tracking_uri: str):
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Install with `pip install mlflow`.") from exc

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow, MlflowClient()


def find_best_run(client, experiment_name: str, metric_name: str, max_results: int, run_filter: str):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. Train first or check --experiment-name."
        )

    filter_parts = ["attributes.status = 'FINISHED'"]
    if run_filter.strip():
        filter_parts.append(f"({run_filter})")
    final_filter = " and ".join(filter_parts)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=final_filter,
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=max_results,
    )

    best_run = None
    best_metric = None
    for run in runs:
        metric_value = run.data.metrics.get(metric_name)
        if metric_value is None:
            continue
        if isinstance(metric_value, float) and math.isnan(metric_value):
            continue
        best_run = run
        best_metric = float(metric_value)
        break

    if best_run is None:
        raise ValueError(
            f"No finished run with metric '{metric_name}' found in experiment '{experiment_name}'."
        )

    return best_run, best_metric


def register_and_promote_model(client, model_name: str, source_model_uri: str, run_id: str, metric_name: str, metric_value: float):
    # Ensure registered model exists.
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    model_version = client.create_model_version(
        name=model_name,
        source=source_model_uri,
        run_id=run_id,
    )

    version = model_version.version
    client.set_registered_model_alias(model_name, "Production", version)
    client.set_registered_model_alias(model_name, "Champion", version)
    client.set_model_version_tag(model_name, version, "promotion_metric", metric_name)
    client.set_model_version_tag(model_name, version, "promotion_metric_value", str(metric_value))

    return version


def walk_artifacts_recursive(client, run_id: str, path: str = ""):
    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            yield from walk_artifacts_recursive(client, run_id, artifact.path)
        else:
            yield artifact.path


def export_production_joblib(client, mlflow, run_id: str, model_name: str, output_path: Path) -> Path:
    ensure_parent_dir(output_path)

    # Prefer explicit joblib artifact if present in run.
    joblib_artifact_path = None
    for artifact_path in walk_artifacts_recursive(client, run_id):
        if artifact_path.endswith(".joblib"):
            joblib_artifact_path = artifact_path
            break

    if joblib_artifact_path:
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{joblib_artifact_path}"
        )
        shutil.copy2(downloaded, output_path)
        return output_path

    # Fallback: export from registry Production alias.
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}@Production")
    joblib.dump(model, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find best MLflow run by metric, register model, and promote alias to Production."
    )
    parser.add_argument("--tracking-uri", type=str, default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--metric-name", type=str, default="val_f1_macro")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--source-artifact-path", type=str, default="model")
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument(
        "--run-filter",
        type=str,
        default="",
        help="Additional MLflow search filter, e.g. tags.pipeline = 'baseline'.",
    )
    parser.add_argument("--min-metric", type=float, default=None)
    parser.add_argument(
        "--production-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "production" / "production_model.joblib",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    mlflow, client = configure_mlflow(args.tracking_uri)
    log(f"Using MLflow tracking URI: {args.tracking_uri}")
    best_run, best_metric = find_best_run(
        client=client,
        experiment_name=args.experiment_name,
        metric_name=args.metric_name,
        max_results=args.max_results,
        run_filter=args.run_filter,
    )

    run_id = best_run.info.run_id
    source_model_uri = f"runs:/{run_id}/{args.source_artifact_path}"

    log(f"Best run_id: {run_id}")
    log(f"Best {args.metric_name}: {best_metric:.6f}")
    log(f"Source model URI: {source_model_uri}")

    if args.min_metric is not None and best_metric < args.min_metric:
        raise ValueError(
            f"Best {args.metric_name}={best_metric:.6f} is below min threshold {args.min_metric:.6f}."
        )

    if args.dry_run:
        log("Dry-run enabled. No registry promotion or file export performed.")
        return

    version = register_and_promote_model(
        client=client,
        model_name=args.model_name,
        source_model_uri=source_model_uri,
        run_id=run_id,
        metric_name=args.metric_name,
        metric_value=best_metric,
    )
    log(f"Promoted model '{args.model_name}' version {version} to alias Production.")

    output_file = export_production_joblib(
        client=client,
        mlflow=mlflow,
        run_id=run_id,
        model_name=args.model_name,
        output_path=args.production_output,
    )
    log(f"Exported production model to: {output_file}")


if __name__ == "__main__":
    main()
