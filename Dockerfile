FROM python:3.10-slim

# System dependencies for scikit-learn
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    apt-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# Lightweight requirements (no AutoGluon, no SDV) for fast production builds
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

# Create required data directories
RUN mkdir -p data/01_raw data/03_primary data/05_model_input data/06_models data/08_reporting

# Create Kedro local credentials file pointing to the SQLite database
RUN mkdir -p conf/local && \
    echo 'db_credentials:' > conf/local/credentials.yml && \
    echo '  con: "sqlite:///data/01_raw/dataset.db"' >> conf/local/credentials.yml

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
