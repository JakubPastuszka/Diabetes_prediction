"""Node functions for the data_processing pipeline (diabetes classification)."""
# pylint: disable=invalid-name
from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import wandb

load_dotenv()

logger = logging.getLogger(__name__)


def preprocess(data: pd.DataFrame, parameters: dict[str, Any]) -> pd.DataFrame:
    """Clean raw rows: optional duplicate removal, zero-as-missing imputation.

    Zero-as-missing handling matches the Sprint 1 baseline: selected columns
    treat ``0`` as invalid; those values are imputed with the column median.

    Args:
        data: Raw dataframe loaded by the Data Catalog from SQLite.
        parameters: Full ``parameters.yml`` dict; uses ``preprocess`` section.

    Returns:
        Cleaned dataframe ready for splitting.
    """
    cfg = parameters.get("preprocess", {})
    drop_dupes = bool(cfg.get("drop_duplicates", True))
    drop_na_rows = bool(cfg.get("drop_na_rows", False))

    if drop_dupes:
        n_before = len(data)
        data = data.drop_duplicates()
        logger.info("Removed %d duplicate rows", n_before - len(data))

    zero_cols = list(cfg.get("zero_as_missing_columns", []))
    if zero_cols:
        subset = [c for c in zero_cols if c in data.columns]
        for col in subset:
            data[col] = data[col].replace(0, pd.NA)
            data[col] = pd.to_numeric(data[col], errors="coerce")
            med = data[col].median()
            data[col] = data[col].fillna(med)
        logger.info("Imputed zeros as missing for columns: %s", subset)

    if drop_na_rows:
        n_before = len(data)
        data = data.dropna()
        logger.info("Dropped %d rows with NA after cleaning", n_before - len(data))

    logger.info("After preprocess: %d rows", len(data))
    return data


def split_data(
    data: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into train / validation / test using ratios from parameters.

    Args:
        data: Preprocessed dataframe.
        parameters: Must define ``target_column`` and nested ``split`` keys.

    Returns:
        Tuple ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    data.columns = [f"{column}" for column in data.columns]

    target = parameters["target_column"]
    split_params = parameters["split"]

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=split_params["val_ratio"],
        random_state=split_params["random_state"],
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict[str, Any],
) -> RandomForestClassifier:
    """Train ``RandomForestClassifier``; all hyperparameters come from YAML.

    Args:
        X_train: Training features.
        y_train: Training target.
        parameters: Uses the ``model`` sub-dict as keyword arguments for sklearn.

    Returns:
        Fitted classifier.
    """
    model_params = dict(parameters["model"])
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    logger.info(
        "Trained RandomForestClassifier with params keys: %s",
        sorted(model_params.keys()),
    )
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Compute validation metrics without external logging side effects."""
    y_pred = model.predict(X_val)
    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }


def evaluate_and_log(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: dict[str, Any],
) -> dict[str, float]:
    """Compute validation metrics and log run data to Weights & Biases.

    Args:
        model: Fitted classifier.
        X_val: Validation features.
        y_val: Validation target.
        parameters: Full ``parameters.yml`` dictionary for W&B config.

    Returns:
        Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    model_params = parameters["model"]
    split_params = parameters["split"]
    run_name = f"rf-n{model_params['n_estimators']}-d{model_params['max_depth']}"
    api_key = os.getenv("WANDB_API_KEY", "")
    mode = os.getenv("WANDB_MODE", "online")
    if not api_key and mode != "offline":
        logger.warning("WANDB_API_KEY not found; switching W&B run to offline mode.")
        mode = "offline"

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "suml-projekt"),
        entity=os.getenv("WANDB_ENTITY"),
        name=run_name,
        config={
            "model_type": "RandomForestClassifier",
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "random_state": model_params["random_state"],
            "test_size": split_params["test_size"],
            "val_ratio": split_params["val_ratio"],
        },
        tags=["sprint3", "baseline", "classification"],
        mode=mode,
    )

    metrics = evaluate_model(model, X_val, y_val)
    wandb.log(metrics)

    if hasattr(model, "feature_importances_"):
        table = wandb.Table(
            data=list(zip(X_val.columns, model.feature_importances_, strict=False)),
            columns=["feature", "importance"],
        )
        wandb.log({"feature_importance": table})

    artifact = wandb.Artifact(
        name="baseline-model",
        type="model",
        description=(
            "RandomForestClassifier for diabetes prediction "
            f"(n_estimators={model_params['n_estimators']}, max_depth={model_params['max_depth']})"
        ),
    )
    artifact.add_file("data/06_models/baseline_model.pkl")
    run.log_artifact(artifact)
    wandb.finish()

    logger.info(
        "W&B logged run %s. Validation metrics — accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        run_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics
