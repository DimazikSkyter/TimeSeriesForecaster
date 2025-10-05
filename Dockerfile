FROM python:3.12-slim AS base

ENV POETRY_VERSION=1.8.4 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry --version

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY README.md README.md
COPY TimeSeriesForecasterSchema.svg TimeSeriesForecasterSchema.svg
COPY TimeSeriesApp ./TimeSeriesApp

ARG INSTALL_DEV=true

RUN if [ "$INSTALL_DEV" = "true" ]; then \
        poetry install --with dev; \
    else \
        poetry install; \
    fi

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "TimeSeriesApp/forecaster_main/ui/app.py"]
