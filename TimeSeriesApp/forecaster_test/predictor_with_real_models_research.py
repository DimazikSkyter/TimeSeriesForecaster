# predictor_with_real_models_research.py
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from forecaster_main.predictor.predictor_service import SingleSeriesModelPredictor, ForecastParams
from forecaster_main.predictor.models import PredictParams, SeriesParams
from forecaster_main.infrastructure.logging import LoggingParams
from forecaster_main.infrastructure.io_data import CsvSource, LoadParams

week_max_5y_path = Path(__file__).parent / "week_max_5y.csv"

def test_week_max_5y_for_all_models():

    print("Start new test 'test_week_max_5y_for_all_models'")

    df = CsvSource().load(LoadParams(path=str(week_max_5y_path)))

    series = df["value"]
    print(series.head())
    # 2. Делим данные
    train_size = int(len(series) * 0.75)
    train_series = series.iloc[:train_size]

    horizon = len(series) - train_size

    # 3. Собираем предсказатель
    predictor = SingleSeriesModelPredictor(
        ForecastParams(max_points=1000, window=20),
        LoggingParams()
    )

    predict_params = PredictParams(
        horizon=horizon,
        freq="W",  # недельные данные
        models_names=["sarimax", "arima", "prophet", "lstm", "trend"]
    )

    # 4. Предсказания
    result = predictor.predict(train_series, predict_params)

    # 5. Визуализация
    plt.figure(figsize=(14, 6))
    plt.plot(series.index, series, label="Original", linewidth=2)

    for col in result.columns:
        if col != "origin_series":
            plt.plot(result.index, result[col], label=col)

    plt.legend()
    plt.title("Forecast comparison on last 25% of data")
    plt.tight_layout()
    plt.show()
