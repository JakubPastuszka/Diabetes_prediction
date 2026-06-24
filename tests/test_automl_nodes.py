"""Unit tests for AutoGluon pipeline nodes."""

from __future__ import annotations

import pandas as pd
import pytest

from suml_projekt.pipelines.automl import nodes


@pytest.fixture
def parameters() -> dict:
    return {
        "target_column": "Outcome",
        "automl": {
            "presets": "medium_quality",
            "time_limit": 120,
            "eval_metric": "f1",
            "model_path": "data/06_models/autogluon",
        },
    }


def test_train_automl_uses_parameters(monkeypatch, parameters):
    class FakePredictor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_kwargs = {}
            self.train_data = None

        def fit(self, train_data, **kwargs):
            self.train_data = train_data
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(nodes, "TabularPredictor", FakePredictor)

    X_train = pd.DataFrame({"Glucose": [90, 120], "BMI": [25.0, 30.0]})
    y_train = pd.Series([0, 1], name="Outcome")
    predictor = nodes.train_automl(X_train, y_train, parameters)

    assert predictor.kwargs == {
        "label": "Outcome",
        "eval_metric": "f1",
        "path": "data/06_models/autogluon",
    }
    assert predictor.fit_kwargs == {
        "presets": "medium_quality",
        "time_limit": 120,
        "verbosity": 1,
    }
    assert list(predictor.train_data.columns) == ["Glucose", "BMI", "Outcome"]


def test_evaluate_automl_logs_metrics_and_leaderboard(monkeypatch, parameters):
    leaderboard = pd.DataFrame(
        {
            "model": ["WeightedEnsemble_L2", "RandomForest"],
            "score_val": [0.8, 0.7],
            "pred_time_val": [0.01, 0.02],
            "fit_time": [1.0, 2.0],
        }
    )
    logged = []
    init_calls = []
    finish_calls = []

    class FakePredictor:
        def leaderboard(self, data, silent):
            assert list(data.columns) == ["Glucose", "Outcome"]
            assert silent is True
            return leaderboard

    class FakeTable:
        def __init__(self, dataframe):
            self.dataframe = dataframe

    def fake_init(**kwargs):
        init_calls.append(kwargs)

    def fake_log(payload):
        logged.append(payload)

    def fake_finish():
        finish_calls.append(True)

    monkeypatch.setattr(nodes.wandb, "init", fake_init)
    monkeypatch.setattr(nodes.wandb, "log", fake_log)
    monkeypatch.setattr(nodes.wandb, "finish", fake_finish)
    monkeypatch.setattr(nodes.wandb, "Table", FakeTable)
    monkeypatch.setenv("WANDB_MODE", "offline")

    metrics = nodes.evaluate_automl(
        FakePredictor(),
        pd.DataFrame({"Glucose": [90, 120]}),
        pd.Series([0, 1], name="Outcome"),
        parameters,
    )

    assert metrics == {
        "best_model": "WeightedEnsemble_L2",
        "f1": 0.8,
        "n_models_trained": 2,
        "presets": "medium_quality",
        "time_limit": 120,
    }
    assert init_calls[0]["mode"] == "offline"
    assert {"f1": 0.8} in logged
    assert "leaderboard" in logged[-1]
    assert finish_calls == [True]
