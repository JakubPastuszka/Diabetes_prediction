"""Node'y pipeline'u SDV: generowanie i ewaluacja danych syntetycznych."""
from __future__ import annotations

import logging

import pandas as pd
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
import wandb

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    real_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Generuje dane syntetyczne synthesizerem GaussianCopula.

    Args:
        real_data: Dane rzeczywiste (Pima Indians) wczytane z katalogu Kedro.
        parameters: Sekcja ``params:synthetic`` z ``parameters.yml``.

    Returns:
        DataFrame z wygenerowanymi danymi syntetycznymi.
    """
    metadata = Metadata.detect_from_dataframe(real_data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)

    n_samples = parameters["n_samples"]
    synthetic = synthesizer.sample(num_rows=n_samples)
    logger.info("Wygenerowano %d rekordów syntetycznych", len(synthetic))
    return synthetic


def evaluate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Ewaluuje jakość danych syntetycznych i loguje wyniki do W&B.

    Liczy dwa raporty SDV: diagnostic (poprawność struktury, ~1.0) oraz
    quality (podobieństwo statystyczne, <1.0). Wyniki loguje do W&B.

    Args:
        real_data: Dane rzeczywiste (Pima Indians).
        synthetic_data: Dane wygenerowane przez SDV.
        parameters: Sekcja ``params:synthetic`` z ``parameters.yml``.

    Returns:
        Słownik z kluczami ``diagnostic_score`` i ``quality_score``.
    """
    metadata = Metadata.detect_from_dataframe(real_data)

    diagnostic = run_diagnostic(real_data, synthetic_data, metadata)
    quality = evaluate_quality(real_data, synthetic_data, metadata)

    scores = {
        "diagnostic_score": diagnostic.get_score(),
        "quality_score": quality.get_score(),
    }
    logger.info(
        "SDV scores — diagnostic=%.4f quality=%.4f",
        scores["diagnostic_score"],
        scores["quality_score"],
    )

    with wandb.init(
        project=parameters["wandb_project"],
        entity=parameters["wandb_entity"],
        job_type="sdv_evaluation",
        config={"n_samples": len(synthetic_data)},
    ):
        wandb.log(
            {
                "sdv/diagnostic_score": scores["diagnostic_score"],
                "sdv/quality_score": scores["quality_score"],
            }
        )

    return scores
