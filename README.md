# SUML-PRO — MLOps Project

Projekt zespołowy realizowany w ramach przedmiotu SUML / MLOps.
Celem projektu jest przygotowanie aplikacji wykorzystującej model Machine Learning
do predykcji ryzyka cukrzycy na podstawie danych Pima Indians Diabetes Dataset.

Projekt składa się z kilku części:

- pipeline'u Kedro do przetwarzania danych, trenowania modelu i ewaluacji,
- lokalnej bazy danych SQLite,
- REST API przygotowanego w FastAPI,
- aplikacji webowej Streamlit,
- śledzenia eksperymentów w Weights & Biases,
- modułu generowania danych syntetycznych z użyciem SDV.

## Forma aplikacji

Projekt można traktować jako aplikację webową z backendem API:

- `Streamlit` — frontend / dashboard użytkownika,
- `FastAPI` — backend udostępniający endpointy do predykcji,
- `SQLite` — lokalne źródło danych,
- `Kedro` — pipeline danych, trenowania i ewaluacji modelu.

Użytkownik korzysta z aplikacji przez przeglądarkę pod adresem:

```
http://localhost:8501
```

API działa pod adresem:

```
http://127.0.0.1:8000
```

Dokumentacja API (Swagger UI) jest dostępna pod:

```
http://127.0.0.1:8000/docs
```

## Tech Stack

| Warstwa                 | Narzędzie              | Rola                                       |
|-------------------------|------------------------|--------------------------------------------|
| Dane                    | SQLite                 | Lokalna baza danych                        |
| Pipeline                | Kedro 0.19.9           | Orkiestracja, preprocessing, trening       |
| Modelowanie baseline    | scikit-learn           | RandomForestClassifier                     |
| Modelowanie AutoML      | AutoGluon 1.5.0        | Automatyczny dobór najlepszego modelu      |
| Śledzenie eksperymentów | Weights & Biases       | Metryki, porównanie modeli                 |
| API                     | FastAPI + Pydantic     | Serwowanie predykcji, walidacja danych     |
| Dashboard               | Streamlit 1.58.0       | Interfejs użytkownika                      |
| Dane syntetyczne        | SDV 1.37.1             | Generowanie i ewaluacja danych             |
| Środowisko              | Python venv            | Izolacja zależności                        |

## Struktura projektu

```
SUML-PRO/
├── api/                              # aplikacja FastAPI
│   ├── __init__.py
│   ├── main.py                       # endpointy: /health, /model-info, /predict
│   ├── model_loader.py               # ładowanie modeli z dysku
│   └── schemas.py                    # walidacja Pydantic
├── app/
│   └── streamlit_app.py              # dashboard Streamlit (3 zakładki)
├── conf/
│   ├── base/
│   │   ├── catalog.yml               # definicje datasetów Kedro
│   │   ├── parameters.yml            # parametry modelu i preprocessingu
│   │   └── parameters_synthetic.yml  # parametry pipeline SDV
│   └── local/
│       └── credentials.yml           # lokalna konfiguracja bazy (nie commitować)
├── data/
│   ├── 01_raw/                       # baza SQLite: dataset.db
│   ├── 03_primary/                   # dane syntetyczne: synthetic_data.csv
│   ├── 05_model_input/               # dane po preprocessingu (pkl)
│   ├── 06_models/                    # zapisane modele: baseline_model.pkl, autogluon/
│   └── 08_reporting/                 # metryki: metrics.json, automl_metrics.json, synthetic_scores.json
├── notebooks/                        # EDA i wcześniejsze eksperymenty
├── scripts/
│   └── bootstrap_dataset_db.py       # pobranie danych i utworzenie SQLite
├── src/suml_projekt/
│   └── pipelines/
│       ├── data_processing/          # preprocessing, split, trening, ewaluacja
│       ├── automl/                   # trening AutoGluon
│       └── synthetic/                # generowanie danych syntetycznych SDV
├── tests/                            # testy jednostkowe
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Wymagania

Zalecana wersja Pythona:

```
Python 3.10, 3.11 lub 3.12
```

Przed uruchomieniem projektu należy mieć zainstalowane:

```
Python
pip
Git
```

Opcjonalnie (do uruchomienia przez Docker):

```
Docker
Docker Compose
```

---

## Szybki start — uruchomienie lokalne

Wszystkie komendy należy wykonywać z katalogu głównego projektu (`SUML-PRO`).

### 1. Sklonuj repozytorium

```bash
git clone <adres_repozytorium>
cd SUML-PRO
```

### 2. Utwórz i aktywuj środowisko wirtualne

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Zainstaluj zależności

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

> Instalacja AutoGluon może zająć kilka minut.

### 4. Utwórz lokalny plik `.env`

```bash
cp .env.example .env
```

Przykładowa zawartość `.env`:

```env
WANDB_API_KEY=your_key_here
WANDB_ENTITY=your_team_name
WANDB_PROJECT=suml-projekt
DATABASE_PATH=data/01_raw/dataset.db
DB_TABLE_NAME=diabetes
WANDB_MODE=offline
```

> Ustaw `WANDB_MODE=offline` jeżeli nie korzystasz z W&B online. Pliku `.env` nie commitować.

### 5. Utwórz lokalne credentials dla Kedro

Utwórz plik `conf/local/credentials.yml`:

```yaml
db_credentials:
  con: "sqlite:///data/01_raw/dataset.db"
```

> Plik `conf/local/credentials.yml` nie powinien być commitowany do repozytorium.

### 6. Utwórz bazę danych SQLite

```bash
python scripts/bootstrap_dataset_db.py
```

Skrypt pobiera publiczny plik CSV z danymi Pima Indians Diabetes Dataset i zapisuje go
jako bazę SQLite w `data/01_raw/dataset.db` z tabelą `diabetes`.

Oczekiwany output:

```
INFO SQLite ready: .../data/01_raw/dataset.db rows=768 table=diabetes
```

### 7. Uruchom pipeline Kedro — model baseline

```bash
python -m kedro run
```

Pipeline wykonuje preprocessing, split danych (train 70% / val 15% / test 15%),
trening RandomForestClassifier oraz ewaluację z logowaniem do W&B.

Po poprawnym wykonaniu powstają:

```
data/05_model_input/X_train.pkl, X_val.pkl, X_test.pkl, ...
data/06_models/baseline_model.pkl
data/08_reporting/metrics.json
```

### 8. Uruchom pipeline AutoGluon (opcjonalnie, zalecane)

```bash
python -m kedro run --pipeline automl
```

AutoGluon automatycznie dobiera najlepszy model spośród wielu algorytmów.
Trening trwa około 2 minut (time_limit=120s, presets=medium_quality).

Po poprawnym wykonaniu powstają:

```
data/06_models/autogluon/
data/08_reporting/automl_metrics.json
```

> Jeżeli AutoGluon jest wytrenowany, API automatycznie używa go zamiast modelu baseline.

### 9. Uruchom API FastAPI

W **pierwszym terminalu** uruchom:

```bash
python -m uvicorn api.main:app --reload
```

API powinno działać pod adresem `http://127.0.0.1:8000`.

Sprawdzenie działania:

```bash
curl http://127.0.0.1:8000/health
```

Oczekiwana odpowiedź:

```json
{
  "status": "ok",
  "baseline_model_loaded": true,
  "automl_model_loaded": true
}
```

### 10. Uruchom aplikację Streamlit

W **drugim terminalu**, z katalogu głównego projektu:

```bash
python -m streamlit run app/streamlit_app.py
```

Aplikacja będzie dostępna pod adresem `http://localhost:8501`.

> Ważne: uruchamiaj zawsze z katalogu `SUML-PRO`, nie z folderu `app/`.

---

## Endpointy API

### GET /health — health check

Sprawdza czy API działa i które modele są załadowane.

```bash
curl http://127.0.0.1:8000/health
```

Przykładowa odpowiedź:

```json
{
  "status": "ok",
  "baseline_model_loaded": true,
  "automl_model_loaded": true
}
```

### GET /model-info — informacje o modelu

Zwraca typ modelu i metryki walidacyjne.

```bash
curl http://127.0.0.1:8000/model-info
```

Przykładowa odpowiedź:

```json
{
  "baseline": {
    "type": "RandomForestClassifier",
    "metrics": {
      "accuracy": 0.739,
      "f1": 0.605,
      "precision": 0.590,
      "recall": 0.622
    }
  },
  "automl": {
    "type": "AutoGluon TabularPredictor",
    "metrics": {
      "f1": 0.567
    }
  }
}
```

### POST /predict — predykcja

Przyjmuje dane pacjenta i zwraca predykcję ryzyka cukrzycy.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 33
  }'
```

Przykładowa odpowiedź:

```json
{
  "prediction": 0,
  "probability": 0.36,
  "model": "RandomForestGini"
}
```

Znaczenie klasy:

```
0 — brak cukrzycy
1 — ryzyko cukrzycy
```

Walidacja danych wejściowych (zwracane kody błędów):

| Kod | Znaczenie                                      |
|-----|------------------------------------------------|
| 200 | Predykcja zakończona sukcesem                  |
| 422 | Błędne dane wejściowe (walidacja Pydantic)     |
| 503 | Model nie jest załadowany                      |
| 500 | Błąd wewnętrzny podczas predykcji              |

Zakresy akceptowanych wartości:

| Pole                     | Typ   | Zakres         |
|--------------------------|-------|----------------|
| Pregnancies              | int   | 0 – 20         |
| Glucose                  | float | > 0, ≤ 300     |
| BloodPressure            | float | 0 – 200        |
| SkinThickness            | float | 0 – 100        |
| Insulin                  | float | 0 – 1000       |
| BMI                      | float | > 0, ≤ 80      |
| DiabetesPedigreeFunction | float | 0.0 – 3.0      |
| Age                      | int   | 1 – 120        |

---

## Dashboard Streamlit

Aplikacja Streamlit zawiera trzy zakładki:

1. **Predykcja** — formularz do wprowadzenia danych pacjenta i uzyskania predykcji modelu przez API.
2. **Dane** — podgląd 100 pierwszych rekordów z lokalnej bazy SQLite oraz statystyki opisowe.
3. **Dane syntetyczne** — generowanie danych syntetycznych z użyciem SDV i porównanie ze statystykami oryginału.

Streamlit komunikuje się z API przez zmienną środowiskową `API_URL`:

```env
API_URL=http://127.0.0.1:8000
```

Domyślnie używa `http://127.0.0.1:8000`.

---

## Pipeline danych syntetycznych — SDV

```bash
python -m kedro run --pipeline=synthetic
```

Pipeline generuje dane syntetyczne i ewaluuje ich jakość, logując wyniki do W&B.

Wyniki zapisywane są do:

```
data/03_primary/synthetic_data.csv
data/08_reporting/synthetic_scores.json
```

Obliczane metryki:

- `diagnostic_score` — poprawność struktury danych syntetycznych (oczekiwane ~1.0),
- `quality_score` — podobieństwo statystyczne do danych rzeczywistych.

---

## Weights & Biases

Logowanie do W&B:

```bash
wandb login
```

Przykładowe eksperymenty z różnymi parametrami:

```bash
python -m kedro run --params="model.n_estimators=100,model.max_depth=10"
python -m kedro run --params="model.n_estimators=300,model.max_depth=10"
python -m kedro run --params="model.n_estimators=100,model.max_depth=20"
python -m kedro run --params="model.n_estimators=500,model.max_depth=20"
python -m kedro run --params="model.n_estimators=200,model.max_depth=null"
```

Tryb offline (bez połączenia z internetem):

```env
WANDB_MODE=offline
```

---

## Testy

Uruchomienie testów jednostkowych:

```bash
python -m pytest
```

---

## Uruchomienie przez Docker

Jeżeli projekt zawiera pliki `Dockerfile` oraz `docker-compose.yml`:

```bash
docker compose up --build
```

Po uruchomieniu:

```
Streamlit: http://localhost:8501
FastAPI:   http://localhost:8000
API docs:  http://localhost:8000/docs
```

> W Docker Compose Streamlit nie powinien odwoływać się do API przez `127.0.0.1`,
> tylko przez nazwę usługi. Ustaw w `.env`:
>
> ```env
> API_URL=http://api:8000
> ```

---

## Najczęstsze problemy

### Brak pliku `dataset.db`

Objaw:

```
sqlite3.OperationalError: unable to open database file
```

Rozwiązanie:

```bash
python scripts/bootstrap_dataset_db.py
```

### Streamlit nie może połączyć się z API

Objaw:

```
Nie można połączyć się z API pod http://127.0.0.1:8000
```

Rozwiązanie: uruchom FastAPI w osobnym terminalu:

```bash
python -m uvicorn api.main:app --reload
```

### API działa, ale model nie jest załadowany (503)

Rozwiązanie: uruchom pipeline treningowy i zrestartuj API:

```bash
python -m kedro run
python -m uvicorn api.main:app --reload
```

### Streamlit uruchomiony z folderu `app/`

Niepoprawnie:

```bash
cd app
python -m streamlit run streamlit_app.py
```

Poprawnie (z katalogu głównego projektu):

```bash
python -m streamlit run app/streamlit_app.py
```

### AutoGluon nie jest dostępny po `kedro run`

`kedro run` uruchamia tylko pipeline baseline. Aby wytrenować AutoGluon:

```bash
python -m kedro run --pipeline automl
```

---

## Bezpieczeństwo

Nie należy commitować do repozytorium:

```
.env
conf/local/credentials.yml
```

Klucze API, dane logowania i lokalne ścieżki powinny być przechowywane wyłącznie lokalnie.

---

## Autorzy

Projekt zespołowy przygotowany w ramach przedmiotu SUML / MLOps.

- Imię i nazwisko 1 — zakres prac
- Imię i nazwisko 2 — zakres prac
- Imię i nazwisko 3 — zakres prac
