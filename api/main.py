"""FastAPI application for diabetes prediction.

Serves predictions from a trained RandomForestClassifier (baseline) or an
AutoGluon TabularPredictor, depending on availability.

Usage (from the suml-projekt/ directory):
    uvicorn api.main:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.model_loader import BASE_DIR, load_automl_model, load_baseline_model
from api.schemas import DiabetesFeatures, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared model registry — populated once at startup, reused for every request.
_models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup; release resources on shutdown."""
    logger.info("Loading models…")
    _models["baseline"] = load_baseline_model()
    _models["automl"] = load_automl_model()

    if _models["baseline"] is None and _models["automl"] is None:
        logger.warning(
            "Neither baseline nor AutoGluon model could be loaded. "
            "Run `kedro run` to train them first. "
            "POST /predict will return 503 until a model is available."
        )
    else:
        loaded = [k for k, v in _models.items() if v is not None]
        logger.info("Models loaded: %s", loaded)

    yield

    _models.clear()
    logger.info("Models released.")


app = FastAPI(
    title="Diabetes Prediction API",
    description=(
        "REST API serwujące predykcje cukrzycy na podstawie danych Pima Indians. "
        "Obsługuje model baseline (RandomForest) oraz AutoGluon (jeśli wytrenowany)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["diagnostics"],
)
def health() -> HealthResponse:
    """Check whether the API is running and which models are loaded.

    Returns:
        JSON with ``status`` and model availability flags.
    """
    return HealthResponse(
        status="ok",
        baseline_model_loaded=_models.get("baseline") is not None,
        automl_model_loaded=_models.get("automl") is not None,
    )


@app.get(
    "/model-info",
    summary="Model metadata",
    tags=["diagnostics"],
)
def model_info() -> dict:
    """Return metadata about the loaded model(s): type and validation metrics.

    Returns:
        Dictionary with info for each loaded model (baseline and/or automl).

    Raises:
        HTTPException 503: No models are loaded.
    """
    if _models.get("baseline") is None and _models.get("automl") is None:
        raise HTTPException(status_code=503, detail="No models loaded.")

    info: dict = {}

    if _models.get("baseline") is not None:
        metrics: dict = {}
        metrics_path = BASE_DIR / "data" / "08_reporting" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        info["baseline"] = {
            "type": "RandomForestClassifier",
            "metrics": metrics,
        }

    if _models.get("automl") is not None:
        metrics = {}
        metrics_path = BASE_DIR / "data" / "08_reporting" / "automl_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        info["automl"] = {
            "type": "AutoGluon TabularPredictor",
            "metrics": metrics,
        }

    return info


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict diabetes outcome",
    tags=["prediction"],
)
def predict(features: DiabetesFeatures) -> PredictionResponse:
    """Predict the diabetes outcome (0 = healthy, 1 = diabetic) for given features.

    Feature values are validated by Pydantic before they reach this function.
    Returns ``422`` for invalid inputs, ``503`` when no model is loaded,
    and ``500`` if the prediction itself fails unexpectedly.

    Args:
        features: Patient features matching the Pima Indians dataset schema.

    Returns:
        Prediction (0 or 1), optional probability, and the model name used.

    Raises:
        HTTPException 503: No model has been loaded.
        HTTPException 500: An error occurred during prediction.
    """
    # Prefer AutoGluon if available; fall back to baseline.
    model = _models.get("automl") or _models.get("baseline")
    model_type = (
        "AutoGluon"
        if _models.get("automl") is not None
        else "RandomForestClassifier (baseline)"
    )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model is loaded. "
                "Run `kedro run` to train the models, then restart the API."
            ),
        )

    input_df = pd.DataFrame([features.model_dump()])

    try:
        if _models.get("automl") is not None:
            prediction_raw = model.predict(input_df)
            proba_raw = model.predict_proba(input_df)
            prediction_value = int(
                prediction_raw.iloc[0]
                if hasattr(prediction_raw, "iloc")
                else prediction_raw[0]
            )
            # AutoGluon returns a DataFrame; grab the column for class 1.
            if hasattr(proba_raw, "iloc"):
                prob_cols = list(proba_raw.columns)
                positive_col = 1 if 1 in prob_cols else prob_cols[-1]
                probability = float(proba_raw[positive_col].iloc[0])
            else:
                probability = float(proba_raw[0][1])

            best_model_name = (
                model.get_model_names(stack_name="core")[0]
                if hasattr(model, "get_model_names")
                else "AutoGluon"
            )

        else:
            prediction_raw = model.predict(input_df)
            prediction_value = int(
                prediction_raw.iloc[0]
                if hasattr(prediction_raw, "iloc")
                else prediction_raw[0]
            )
            proba_raw = (
                model.predict_proba(input_df)
                if hasattr(model, "predict_proba")
                else None
            )
            probability = (
                float(proba_raw[0][1]) if proba_raw is not None else None
            )
            best_model_name = model_type

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {exc}",
        ) from exc

    logger.info(
        "Prediction: %d (prob=%.3f) via %s",
        prediction_value,
        probability if probability is not None else -1.0,
        best_model_name,
    )
    return PredictionResponse(
        prediction=prediction_value,
        probability=probability,
        model=best_model_name,
    )