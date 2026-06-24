import logging
import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Brak wymaganej zmiennej srodowiskowej: {name}")
    return value


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    db_path_raw = _get_required_env("DB_PATH")
    source_csv_raw = _get_required_env("RAW_DATA_PATH")
    table_name = os.getenv("DB_TABLE_NAME", "diabetes")

    db_path = _resolve_path(project_root, db_path_raw)
    source_csv = _resolve_path(project_root, source_csv_raw)

    if not source_csv.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z danymi: {source_csv}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)

    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    logger.info("Baza utworzona: %s", db_path)
    logger.info("Tabela: %s", table_name)
    logger.info("Liczba rekordow: %s", row_count)


if __name__ == "__main__":
    main()
