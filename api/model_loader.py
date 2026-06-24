"""Model loading utilities for the diabetes prediction API."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base directory: api/model_loader.py → suml-projekt/
BASE_DIR = Path(__file__).parent.parent

BASELINE_MODEL_PATH = BASE_DIR / "data" / "06_models" / "baseline_model.pkl"
AUTOGLUON_MODEL_PATH = BASE_DIR / "data" / "06_models" / "autogluon"


def load_baseline_model() -> Any | None:
    """Load the sklearn RandomForestClassifier from disk.

    Returns:
        Fitted sklearn model, or ``None`` if the file does not exist.
    """
    if not BASELINE_MODEL_PATH.exists():
        logger.warning(
            "Baseline model not found at %s. Run `kedro run` first.",
            BASELINE_MODEL_PATH,
        )
        return None
    try:
        with BASELINE_MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
        logger.info("Baseline model loaded from %s", BASELINE_MODEL_PATH)
        return model
    except Exception:
        logger.exception("Failed to load baseline model")
        return None


def load_automl_model() -> Any | None:
    """Load the AutoGluon TabularPredictor from disk.

    Returns:
        Fitted AutoGluon predictor, or ``None`` if the directory does not exist.
    """
    if not AUTOGLUON_MODEL_PATH.exists():
        logger.warning(
            "AutoGluon model not found at %s. Run `kedro run` first.",
            AUTOGLUON_MODEL_PATH,
        )
        return None
    try:
        from autogluon.tabular import TabularPredictor  # noqa: PLC0415

        predictor = TabularPredictor.load(str(AUTOGLUON_MODEL_PATH))
        logger.info("AutoGluon model loaded from %s", AUTOGLUON_MODEL_PATH)
        return predictor
    except ImportError:
        logger.warning("autogluon not installed; AutoGluon model will not be loaded.")
        return None
    except Exception:
        logger.exception("Failed to load AutoGluon model")
        return None