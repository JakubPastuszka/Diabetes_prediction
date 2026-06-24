import json
import logging
import os
import sqlite3

import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_data():
    load_dotenv()
    db_path = os.getenv("DB_PATH")

    if not db_path:
        raise ValueError("Brak DB_PATH w .env")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM diabetes", conn)

    return df


def clean_data(df):
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in zero_cols:
        df[col] = df[col].replace(0, pd.NA)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df


def split_data(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
    }

    return metrics


def save_metrics(metrics):
    path = os.getenv("METRICS_PATH")

    if not path:
        raise ValueError("Brak METRICS_PATH w .env")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def main():
    df = load_data()
    df = clean_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    model = train_model(X_train, y_train)

    metrics = evaluate(model, X_val, y_val)

    save_metrics(metrics)
    cm = confusion_matrix(y_val, model.predict(X_val))
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
