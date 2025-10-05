import json
import logging
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from forecaster_main.infrastructure.io_data import (
    LoadParams, CsvSource, PrometheusSource,
    ClickHouseSource, ClickhouseParams
)
from forecaster_main.predictor.models import PredictParams
from forecaster_main.predictor.predictor_service import SingleSeriesModelPredictor, ForecastParams
from forecaster_main.infrastructure.logging import LoggingParams

FORECASTER_FILE_PATH: str = os.getenv('FORECASTER_FILE_PATH',
                                      str(Path(__file__).parent / "forecaster_params.json"))
st.set_page_config(page_title="Metric Forecaster", layout="wide")
st.title("📈 Metric Forecaster Prototype")
st.caption("Загрузи временные ряды из источника и визуализируй их.")

# Инициализация session_state
models = ["trend"]
if "df" not in st.session_state:
    st.session_state["df"] = None

# Добавление сервиса анализа, для апдейта конфига нужно перезапускать сервис
with open(FORECASTER_FILE_PATH, "r", encoding="utf-8") as f:
    json_obj = json.load(f)
forecaster_params: ForecastParams = ForecastParams(**json_obj)
logging_level = os.getenv('LOGGING_LEVEL', 'INFO')
level = logging.getLevelName(logging_level.upper())
ssmp: SingleSeriesModelPredictor = SingleSeriesModelPredictor(forecaster_params, LoggingParams(level=level))

# --- Источники ---
source_type = st.selectbox(
    "Источник данных",
    ["CSV/Excel", "Prometheus/Victoria", "ClickHouse"]
)

# CSV / Excel
if source_type == "CSV/Excel":
    file = st.file_uploader("Загрузите CSV или Excel", type=["csv", "xlsx"])
    if file is not None:
        loader = CsvSource()
        params = LoadParams(path=file)
        df = loader.load(params)
        st.session_state["df"] = df

# Prometheus / Victoria
elif source_type == "Prometheus/Victoria":
    uri = st.text_input("URI", "http://localhost:8428")
    query = st.text_input("Query", "up")
    start = st.date_input("Start date")
    end = st.date_input("End date")
    if st.button("Загрузить"):
        loader = PrometheusSource()
        params = LoadParams(
            uri=uri,
            query=query,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
        )
        df = loader.load(params)
        st.session_state["df"] = df

# ClickHouse
elif source_type == "ClickHouse":
    host = st.text_input("Host", "localhost")
    port = st.number_input("Port", 8123)
    user = st.text_input("User", "default")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database", "default")
    table = st.text_input("Table", "timeseries")
    if st.button("Загрузить"):
        params = LoadParams(ch_table=table)
        ch_params = ClickhouseParams(
            ch_host=host,
            ch_port=port,
            ch_user=user,
            ch_password=password,
            ch_database=database,
        )
        loader = ClickHouseSource(ch_params)
        df = loader.load(params)
        st.session_state["df"] = df

# --- Отображение данных ---
if st.session_state["df"] is not None:
    df = st.session_state["df"]

    st.subheader("📊 Загруженные данные")
    st.dataframe(df.head())

    fig = px.line(df, x=df.index, y=df.columns, title="Метрики")
    st.plotly_chart(fig, use_container_width=True)

    # --- Forecast UI ---
    st.subheader("🔮 Forecast")

    horizon = st.number_input("Horizon (кол-во точек)", min_value=1, value=10)
    mode = st.radio("Откуда прогнозировать?", ["С конца ряда", "С номера индекса"])

    start_idx = None
    if mode == "С номера индекса":
        start_idx = st.number_input("Номер точки (0..N-1)", min_value=0, max_value=len(df) - 1, value=len(df) - 5)

    series_list = df.columns.tolist()
    selected_series = st.selectbox("Выберите временной ряд", series_list)
    st.write("Вы выбрали:", selected_series)

    st.markdown("**Выберите модели:**")
    use_sarimax = st.checkbox("SARIMAX", value=True)
    use_arima = st.checkbox("ARIMA")
    use_prophet = st.checkbox("Prophet")
    use_lstm = st.checkbox("LSTM")

    if use_sarimax:
        models.append("sarimax")
    if use_arima:
        models.append("arima")
    if use_prophet:
        models.append("prophet")
    if use_lstm:
        models.append("lstm")

    if selected_series and st.button("Выполнить прогноз"):
        if not models:
            st.warning("Выберите хотя бы одну модель!")
        chosen_series = df[selected_series]
        print(f"Current index {chosen_series.index.inferred_type}")
        if start_idx:
            chosen_series = chosen_series.iloc[:start_idx]

        if horizon > len(chosen_series) // 2:
            st.warning("Horizon больше половины длины ряда — прогноз может быть нестабильным.")

        forecast_df = ssmp.predict(chosen_series, PredictParams(models_names=models, horizon=horizon))

        # запускаем прогноз
        st.subheader("📈 Прогноз")
        st.dataframe(forecast_df)

        fig_fcst = px.line(forecast_df, title="Forecasts by models")
        st.plotly_chart(fig_fcst, use_container_width=True)
