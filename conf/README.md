# Folder `conf/` — po co to jest

Tu trzymamy konfigurację pod Kedro (i ewentualnie inne narzędzia).

## `conf/local/`

Na swoim kompie wrzucasz rzeczy **tylko dla siebie** albo **wrażliwe** — np. `credentials.yml` z hasłem / ścieżką do bazy.  
**Nie commitujemy** tego do gita (żeby przypadkiem nie wypchnąć sekretów).

Jak nie masz jeszcze `credentials.yml`, skopiuj przykład z repo:

`docs/sqlite_credentials.example.yml` → `conf/local/credentials.yml`

## `conf/base/`

Tu jest wspólna konfiguracja dla całej grupy: `catalog.yml`, `parameters.yml` itd. — bez haseł i bez ścieżek specyficznych tylko dla jednej maszyny.

## Jak nie wiesz co gdzie

Dokumentacja Kedro o konfiguracji:  
https://docs.kedro.org/en/stable/kedro_project_setup/configuration.html
