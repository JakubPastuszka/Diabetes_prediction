"""Node functions for the AutoGluon AutoML pipeline."""
# pylint: disable=invalid-name
from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv
import wandb

load_dotenv()

logger = logging.getLogger(__name__)

LOWER_IS_BETTER_METRICS = {
    "root_mean_squared_error",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
}


def _data_with_target(
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
) -> pd.DataFrame:
    return pd.concat([features, target.rename(target_column)], axis=1)


def _wandb_mode() -> str:
    mode = os.getenv("WANDB_MODE", "online")
    if not os.getenv("WANDB_API_KEY") and mode != "offline":
        logger.warning("WANDB_API_KEY not found; switching W&B run to offline mode.")
        return "offline"
    return mode


def _actual_metric_value(metric_name: str, score: float) -> float:
    if metric_name in LOWER_IS_BETTER_METRICS:
        return -score
    return score


def train_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict[str, Any],
) -> Any:
    """Train AutoGluon TabularPredictor using parameters from YAML.

    Args:
        X_train: Training features.
        y_train: Training target.
        parameters: Full ``parameters.yml`` dictionary.

    Returns:
        Fitted AutoGluon predictor.
    """
    target_column = parameters["target_column"]
    automl_params = parameters["automl"]
    train_data = _data_with_target(X_train, y_train, target_column)

    logger.info(
        "AutoGluon training started: presets=%s, time_limit=%s",
        automl_params["presets"],
        automl_params["time_limit"],
    )

    predictor = TabularPredictor(
        label=target_column,
        eval_metric=automl_params["eval_metric"],
        path=automl_params["model_path"],
    ).fit(
        train_data,
        presets=automl_params["presets"],
        time_limit=automl_params["time_limit"],
        verbosity=1,
    )

    logger.info("AutoGluon training finished.")
    return predictor


def evaluate_automl(
    predictor: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate AutoGluon predictor and log metrics plus leaderboard to W&B.

    Args:
        predictor: Fitted AutoGluon predictor.
        X_val: Validation features.
        y_val: Validation target.
        parameters: Full ``parameters.yml`` dictionary.

    Returns:
        Best model metadata and validation metric.
    """
    target_column = parameters["target_column"]
    automl_params = parameters["automl"]
    eval_metric = automl_params["eval_metric"]
    val_data = _data_with_target(X_val, y_val, target_column)
    leaderboard = predictor.leaderboard(data=val_data, silent=True)

    best_row = leaderboard.iloc[0]
    best_model = str(best_row["model"])
    best_score = float(best_row["score_val"])
    best_metric_value = _actual_metric_value(eval_metric, best_score)
    run_name = f"automl-{automl_params['presets']}-{automl_params['time_limit']}s"

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "suml-projekt"),
        entity=os.getenv("WANDB_ENTITY"),
        name=run_name,
        config={
            "model_type": "AutoGluon",
            "presets": automl_params["presets"],
            "time_limit": automl_params["time_limit"],
            "eval_metric": eval_metric,
            "best_model": best_model,
        },
        tags=["sprint4", "automl", "autogluon", "classification"],
        mode=_wandb_mode(),
    )

    wandb.log({eval_metric: best_metric_value})
    wandb.log(
        {"leaderboard": wandb.Table(dataframe=leaderboard.reset_index(drop=True))}
    )
    wandb.finish()

    metrics = {
        "best_model": best_model,
        eval_metric: float(best_metric_value),
        "n_models_trained": int(len(leaderboard)),
        "presets": automl_params["presets"],
        "time_limit": automl_params["time_limit"],
    }
    logger.info(
        "AutoGluon best model: %s, %s=%.4f",
        best_model,
        eval_metric,
        best_metric_value,
    )
    return metrics
