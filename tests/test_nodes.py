"""Unit tests for pipeline nodes (no Kedro session required)."""

from __future__ import annotations

import pandas as pd
import pytest

from suml_projekt.pipelines.data_processing.nodes import (
    evaluate_model,
    preprocess,
    split_data,
    train_model,
)


@pytest.fixture
def parameters() -> dict:
    return {
        "target_column": "Outcome",
        "split": {"test_size": 0.3, "val_ratio": 0.5, "random_state": 42},
        "model": {"n_estimators": 5, "max_depth": 3, "random_state": 42},
        "preprocess": {
            "drop_duplicates": False,
            "drop_na_rows": False,
            "zero_as_missing_columns": ["Glucose"],
        },
    }


def test_preprocess_imputes_zero_glucose(parameters):
    expected_median_glucose = 100.0
    df = pd.DataFrame(
        {
            "Glucose": [0, 100],
            "Other": [1, 2],
            "Outcome": [0, 1],
        }
    )
    out = preprocess(df.copy(), parameters)
    assert out["Glucose"].iloc[0] == expected_median_glucose


def test_split_train_val_test_sizes(parameters):
    n_rows = 100
    exp_train, exp_val, exp_test = 70, 15, 15
    df = pd.DataFrame(
        {
            "f1": range(n_rows),
            "f2": range(n_rows, n_rows * 2),
            "Outcome": [0, 1] * 50,
        }
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, parameters)
    assert len(X_train) == exp_train
    assert len(X_val) == exp_val
    assert len(X_test) == exp_test
    assert list(y_train.index) == list(X_train.index)


def test_split_data_normalizes_column_names(parameters):
    class SqlColumnName(str):
        pass

    df = pd.DataFrame(
        {
            SqlColumnName("f1"): range(10),
            SqlColumnName("Outcome"): [0, 1] * 5,
        }
    )
    X_train, *_ = split_data(df, parameters)
    assert {type(column) for column in X_train.columns} == {str}


def test_train_and_evaluate_roundtrip(parameters):
    df = pd.DataFrame(
        {
            "a": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "Outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, parameters)
    model = train_model(X_train, y_train, parameters)
    metrics = evaluate_model(model, X_val, y_val)
    lo, hi = 0.0, 1.0
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    for v in metrics.values():
        assert lo <= v <= hi
