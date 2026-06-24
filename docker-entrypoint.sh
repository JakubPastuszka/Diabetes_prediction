#!/bin/bash
set -e

echo "=== SUML-PRO startup ==="

# 1. Bootstrap database if not present
if [ ! -f "data/01_raw/dataset.db" ]; then
    echo "[1/3] Creating SQLite database..."
    python scripts/bootstrap_dataset_db.py
else
    echo "[1/3] Database already exists, skipping."
fi

# 2. Run Kedro baseline pipeline only if model not present
if [ ! -f "data/06_models/baseline_model.pkl" ]; then
    echo "[2/3] Training baseline model (kedro run)..."
    python -m kedro run
else
    echo "[2/3] Baseline model already exists, skipping."
fi

# 3. AutoGluon — only if explicitly requested via RUN_AUTOML=true
if [ "${RUN_AUTOML:-false}" = "true" ]; then
    if [ ! -d "data/06_models/autogluon" ]; then
        echo "[2/3] Training AutoGluon model (kedro run --pipeline automl)..."
        python -m kedro run --pipeline automl
    else
        echo "[2/3] AutoGluon model already exists, skipping."
    fi
else
    echo "[2/3] AutoGluon skipped (set RUN_AUTOML=true to enable)."
fi

# 4. Start FastAPI
echo "[3/3] Starting FastAPI..."
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
EOF
