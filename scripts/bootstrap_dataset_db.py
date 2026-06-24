#!/usr/bin/env python3
"""Create ``data/01_raw/dataset.db`` with the ``diabetes`` table (Pima Indians).

Run from the Kedro project root (``suml-projekt/``):

    python scripts/bootstrap_dataset_db.py

Uses the same public CSV as in Sprint 1 EDA (Plotly mirror). Path matches
``conf/local/credentials.yml`` → ``sqlite:///data/01_raw/dataset.db``.
"""

from __future__ import annotations

import logging
import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DB_REL = Path("data/01_raw/dataset.db")
TABLE_NAME = "diabetes"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db_path = PROJECT_ROOT / DB_REL
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:  # noqa: S310
        df = pd.read_csv(resp)

    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        n = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

    logger.info("SQLite ready: %s rows=%s table=%s", db_path, n, TABLE_NAME)


if __name__ == "__main__":
    main()
