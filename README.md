# ASI_PRO — MLOps Project (Sprint 1-6)

Projekt zespołowy realizowany w ramach przedmiotu ASI (MLOps).  
Aktualny zakres: pipeline Kedro + śledzenie eksperymentów w Weights & Biases,
REST API (FastAPI), dashboard Streamlit oraz pipeline generowania danych
syntetycznych (SDV).

## Tech Stack

- `Python`, `scikit-learn`, `pandas`
- `SQLite` jako źródło danych
- `Kedro 0.19.9` do orkiestracji pipeline
- `Weights & Biases (wandb)` do eksperymentów
- `FastAPI` + `uvicorn` — REST API serwujące predykcje (Sprint 5)
- `Streamlit` — dashboard użytkownika (Sprint 6)
- `SDV` — generowanie i ewaluacja danych syntetycznych (Sprint 6)

## Struktura projektu

- `asi-projekt/conf/base/catalog.yml` — definicje datasetów
- `asi-projekt/conf/base/parameters.yml` — parametry modelu i preprocessingu
- `asi-projekt/conf/base/parameters_synthetic.yml` — parametry pipeline'u SDV
- `asi-projekt/conf/local/credentials.yml` — lokalne credentials (gitignored)
- `asi-projekt/src/asi_projekt/pipelines/data_processing/` — preprocessing, split, trening, ewaluacja
- `asi-projekt/src/asi_projekt/pipelines/automl/` — AutoGluon (Sprint 4)
- `asi-projekt/src/asi_projekt/pipelines/synthetic/` — generowanie + ewaluacja danych syntetycznych (Sprint 6)
- `asi-projekt/api/` — aplikacja FastAPI (Sprint 5)
- `asi-projekt/app/streamlit_app.py` — dashboard Streamlit (Sprint 6)
- `asi-projekt/notebooks/01_eda.ipynb` — EDA i baseline z etapu notebookowego

## Co działa
WSZYSTKO B)
- ładowanie danych z `SQLite` przez `Data Catalog` (`pandas.SQLTableDataset`)
- preprocessing -> split train/val/test -> trening modelu -> ewaluacja
- zapis artefaktów pipeline:
  - `data/05_model_input/*.pkl`
  - `data/06_models/baseline_model.pkl`
  - `data/08_reporting/metrics.json`
- integracja W&B:
  - nowy run przy każdym `kedro run`
  - logowanie `config`, metryk (`accuracy`, `precision`, `recall`, `f1`) i artefaktu modelu
- REST API (FastAPI): `/health`, `/model-info`, `/predict` (model ładowany raz przy starcie)
- dashboard Streamlit z 3 zakładkami: Predykcja (request do API), Dane (podgląd z bazy),
  Dane syntetyczne (generowanie SDV + porównanie z oryginałem)
- pipeline SDV: generowanie danych + ewaluacja (`run_diagnostic`, `evaluate_quality`),
  oba score'y logowane do W&B
  - `data/03_primary/synthetic_data.csv`
  - `data/08_reporting/synthetic_scores.json`

## Szybki start

1. Przejdź do katalogu projektu:

   ```bash
   cd asi-projekt
   ```

2. Zainstaluj zależności:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Utwórz lokalne credentials bazy:

   ```yaml
   # conf/local/credentials.yml
   db_credentials:
     con: "sqlite:///data/01_raw/dataset.db"
   ```

4. Uzupełnij `.env` (lokalnie, nie commitować):

   ```env
   WANDB_API_KEY=your_key_here
   WANDB_ENTITY=your_team_name
   WANDB_PROJECT=ASI-Project
   DATABASE_PATH=data/01_raw/dataset.db
   DB_TABLE_NAME=diabetes
   ```

5. Uruchom pipeline:

   ```bash
   python -W "default:Kedro is not yet fully compatible" -m kedro run
   ```

6. Uruchom pipeline treningowy:

```bash
   python -m kedro run
```

7. (Opcjonalnie) uruchom wizualizację DAG:

```bash
   python -m kedro viz run

## Dashboard Streamlit (Sprint 6)

W osobnym terminalu, przy działającym API:

```bash
python -m streamlit run app/streamlit_app.py
```

Dashboard: `http://localhost:8501`. Adres API konfigurowalny przez zmienną
środowiskową `API_URL` (domyślnie `http://127.0.0.1:8000`).

## Pipeline danych syntetycznych — SDV (Sprint 6)

Generuje dane syntetyczne i ewaluuje ich jakość, logując wyniki do W&B:

```bash
python -m kedro run --pipeline=synthetic
```
Wynik: `diagnostic_score` (~1.0, poprawność struktury) i `quality_score`
(podobieństwo statystyczne, celowo < 1.0) — zapisane do
`data/08_reporting/synthetic_scores.json` i zalogowane do W&B.


## Weights and Biases (Sprint 3)

- Logowanie do W&B:

  ```bash
  wandb login
  ```

- Przykładowe eksperymenty:

  ```bash
  python -W "default:Kedro is not yet fully compatible" -m kedro run --params="model.n_estimators=100,model.max_depth=10"
  python -W "default:Kedro is not yet fully compatible" -m kedro run --params="model.n_estimators=300,model.max_depth=10"
  python -W "default:Kedro is not yet fully compatible" -m kedro run --params="model.n_estimators=100,model.max_depth=20"
  python -W "default:Kedro is not yet fully compatible" -m kedro run --params="model.n_estimators=500,model.max_depth=20"
  python -W "default:Kedro is not yet fully compatible" -m kedro run --params="model.n_estimators=200,model.max_depth=null"
  ```

- Dashboard projektu:
  - `https://wandb.ai/13c_2/ASI-Project`

## Bezpieczeństwo

- Nigdy nie commituj `.env` ani `conf/local/credentials.yml`.
- Klucze API i hasła trzymamy tylko lokalnie.
