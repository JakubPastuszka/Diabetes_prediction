"""Dashboard Streamlit dla projektu predykcja cukrzycy (Pima Indians).

Komunikuje się z FastAPI (sprint 5) przez HTTP. Trzy zakładki:
Predykcja, Dane, Dane syntetyczne.
"""

# --- 1. IMPORTY ---
import os
import sqlite3
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# SDV jest opcjonalne — wymagane tylko dla zakładki danych syntetycznych.
# Bez SDV zakładka wyświetla komunikat zamiast crashować aplikację.
try:
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

# --- 2. KONFIGURACJA ---
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
DB_PATH = Path("data/01_raw/dataset.db")
TABLE_NAME = "diabetes"

# --- 3. FUNKCJE CACHE'OWANE ---
@st.cache_data
def load_data() -> pd.DataFrame:
    """Wczytuje dane cukrzycy z SQLite. Cache'owane — zapytanie wykonuje się raz."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)


@st.cache_resource
def fit_synthesizer(real_data: pd.DataFrame):
    """Trenuje synthesizer SDV raz i cache'uje wytrenowany obiekt.

    Args:
        real_data: DataFrame z oryginalnymi danymi treningowymi.

    Returns:
        Wytrenowany GaussianCopulaSynthesizer lub None jeśli SDV niedostępne.
    """
    if not SDV_AVAILABLE:
        return None
    metadata = Metadata.detect_from_dataframe(real_data)
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(real_data)
    return synth


# --- 4. LAYOUT ---
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")
st.title("Diabetes Prediction Dashboard")
tab_pred, tab_data, tab_synth = st.tabs(["Predykcja", "Dane", "Dane syntetyczne"])

# --- 5a. ZAKŁADKA: PREDYKCJA ---
with tab_pred:
    st.header("Predykcja cukrzycy")
    st.caption("Wprowadź dane pacjenta — model oceni ryzyko cukrzycy (dane Pima Indians).")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Pregnancies — liczba ciąż", 0, 20, 2)
        glucose = st.slider("Glucose — stężenie glukozy (mg/dL)", 1.0, 300.0, 120.0, 1.0)
        blood_pressure = st.slider("BloodPressure — ciśnienie rozkurczowe (mm Hg)", 0.0, 200.0, 70.0, 1.0)
        skin_thickness = st.slider("SkinThickness — grubość fałdu skórnego (mm)", 0.0, 100.0, 25.0, 1.0)
    with col2:
        insulin = st.slider("Insulin — insulina 2h (mu U/ml)", 0.0, 1000.0, 80.0, 1.0)
        bmi = st.slider("BMI — wskaźnik masy ciała", 1.0, 80.0, 28.5, 0.1)
        dpf = st.slider("DiabetesPedigreeFunction — wskaźnik genetyczny", 0.0, 3.0, 0.35, 0.01)
        age = st.slider("Age — wiek (lata)", 1, 120, 33)

    if st.button("Przewiduj", type="primary"):
        payload = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                result = r.json()
                prob = result.get("probability")
                if result["prediction"] == 1:
                    st.error("Wynik: **wykryto ryzyko cukrzycy** (klasa 1)")
                else:
                    st.success("Wynik: **brak cukrzycy** (klasa 0)")
                if prob is not None:
                    st.metric("Prawdopodobieństwo cukrzycy", f"{prob * 100:.1f}%")
                st.caption(f"Model: {result['model']}")
            elif r.status_code == 422:
                st.error("Błędne dane wejściowe (walidacja Pydantic).")
                st.json(r.json())
            elif r.status_code == 503:
                st.error("API działa, ale model nie jest załadowany. "
                         "Uruchom `python -m kedro run` i zrestartuj API.")
            else:
                st.error(f"Błąd API ({r.status_code}): {r.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Nie można połączyć się z API pod {API_URL}. "
                     "Czy `uvicorn api.main:app` jest uruchomione?")

# --- 5b. ZAKŁADKA: DANE ---
with tab_data:
    st.header("Podgląd danych (Pima Indians)")
    df = load_data()
    st.write(f"Liczba rekordów: {len(df)}")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Statystyki opisowe")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Rozkład wybranej kolumny")
    column = st.selectbox("Kolumna", df.select_dtypes("number").columns)
    st.bar_chart(df[column].value_counts().sort_index())

# --- 5c. ZAKŁADKA: DANE SYNTETYCZNE ---
with tab_synth:
    st.header("Dane syntetyczne (SDV)")
    st.caption("Generowanie danych przypominających oryginał statystycznie, "
               "ale bez prawdziwych rekordów pacjentów.")

    if not SDV_AVAILABLE:
        st.warning(
            "Biblioteka SDV nie jest zainstalowana w tym środowisku. "
            "Aby uruchomić generowanie danych syntetycznych, użyj wersji development: "
            "`docker compose -f docker-compose.dev.yml up --build`"
        )
    else:
        df = load_data()

        n_samples = st.number_input(
            "Liczba rekordów do wygenerowania", 100, 10_000, 1000, step=100
        )

        if st.button("Generuj dane syntetyczne", type="primary"):
            with st.spinner("Trenowanie synthesizera i generowanie..."):
                synth = fit_synthesizer(df)
                st.session_state["synthetic"] = synth.sample(num_rows=int(n_samples))

        if "synthetic" in st.session_state:
            synthetic = st.session_state["synthetic"]
            st.success(f"Wygenerowano {len(synthetic)} rekordów.")

            col_real, col_synth = st.columns(2)
            with col_real:
                st.subheader("Oryginał — statystyki")
                st.dataframe(df.describe(), use_container_width=True)
            with col_synth:
                st.subheader("Syntetyczne — statystyki")
                st.dataframe(synthetic.describe(), use_container_width=True)

            st.subheader("Podgląd danych syntetycznych")
            st.dataframe(synthetic.head(50), use_container_width=True)
