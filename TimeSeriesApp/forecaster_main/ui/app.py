import json
import os

import streamlit as st
import pandas as pd
import plotly.express as px

from forecaster_main.infrastructure.io_data import (
    LoadParams, CsvSource, PrometheusSource,
    ClickHouseSource, ClickhouseParams
)
from forecaster_main.predictor.predictor_service import SingleSeriesModelPredictor, ForecastParams
from predictor.models import SARIMAXModel, SarimaxParams, ARIMAParams, ARIMAModel, ProphetModel, LSTMModel, LSTMParams, \
    ProphetParams, PredictParams

FORECASTER_FILE_PATH: str = os.getenv('FORECASTER_FILE_PATH', "./forecaster_params.json")

st.set_page_config(page_title="Metric Forecaster", layout="wide")
st.title("📈 Metric Forecaster Prototype")
st.caption("Загрузи временные ряды из источника и визуализируй их.")

# Инициализация session_state
if "df" not in st.session_state:
    st.session_state["df"] = None

# Добавление сервиса анализа, для апдейта конфига нужно перезапускать сервис
json_obj = json.loads(FORECASTER_FILE_PATH)
forecaster_params : ForecastParams = ForecastParams(**json_obj)
ssmp: SingleSeriesModelPredictor = SingleSeriesModelPredictor(forecaster_params)

# --- Источники ---
source_type = st.selectbox(
    "Источник данных",
    ["CSV/Excel", "Prometheus/Victoria", "ClickHouse"]
)

# CSV / Excel
if source_type == "CSV/Excel":
    file = st.file_uploader("Загрузите CSV или Excel", type=["csv", "xlsx"])
    if file is not None:
        if file.name.endswith("xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        if "timestamp" in df.columns:
            #df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.set_index("timestamp")

        loader = CsvSource()
        params = LoadParams(path=file.name)
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
        start_idx = st.number_input("Номер точки (0..N-1)", min_value=0, max_value=len(df)-1, value=len(df)-5)

    st.markdown("**Выберите модели:**")
    use_sarimax = st.checkbox("SARIMAX", value=True)
    use_arima = st.checkbox("ARIMA")
    use_prophet = st.checkbox("Prophet")
    use_lstm = st.checkbox("LSTM")

    if st.button("Выполнить прогноз"):
        # Берём первую метрику для примера

        ssmp.predict()

        series = df.iloc[:, 0]
        if start_idx:
            series = series.iloc[:start_idx]

        predictor = SingleSeriesModelPredictor(ForecastParams(acf_length=None))
#Переделать
        models = []
        if use_sarimax:
            models.append(SARIMAXModel(series, SarimaxParams()))
        if use_arima:
            models.append(ARIMAModel(series, ARIMAParams()))
        if use_prophet:
            models.append(ProphetModel(series, ProphetParams()))
        if use_lstm:
            models.append(LSTMModel(series, LSTMParams(window=5, horizon=horizon)))

        # подключаем модели
        for m in models:
            predictor.models[m.name] = m

        # запускаем прогноз
        results = {}
        for name, model in predictor.models.items():
            fcst = model.forecast(PredictParams(horizon=horizon))
            results[name] = fcst["yhat"]

        forecast_df = pd.DataFrame(results)
        st.subheader("📈 Прогноз")
        st.dataframe(forecast_df)

        fig_fcst = px.line(forecast_df, title="Forecasts by models")
        st.plotly_chart(fig_fcst, use_container_width=True)