import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.lite.python.util import trace_model_call
from utilsforecast.feature_engineering import trend

from forecaster_main.predictor.models import ARIMAModel, ProphetModel, LSTMModel, PredictParams, ARIMAParams, \
    ProphetParams, LSTMParams, SARIMAXModel
from forecaster_test.daily_load_generator import generate_series,weekly_max
from predictor.models import SarimaxParams, SeriesParams

seria_size = 1000
prediction_len = 120
lstm_units = 50
lstm_epochs = 20
# --- вспомогательные функции ---

def generate_trend_load():
    series = generate_series()
    week_series = weekly_max(series)
    print(week_series)
    train_size = round(len(week_series) * 0.8)
    return week_series[:train_size], week_series[(train_size - len(week_series)):]

def generate_arima_like(n=100, seed=42, tail_size=20):
    np.random.seed(seed)
    series = np.cumsum(np.random.randn(n + tail_size))  # случайное блуждание ~ ARIMA(2,1,2)
    return (pd.Series(series[:n], index=pd.RangeIndex(start=0, stop=n, step=1)),
            pd.Series(series[-tail_size:], index=pd.RangeIndex(start=n, stop=n + tail_size, step=1)))

def generate_log_seasonal(n=100):
    x = np.arange(1, n + 1)
    trend = np.log(x)
    season20 = np.sin(2 * np.pi * x / 20)
    season8 = 0.5 * np.sin(2 * np.pi * x / 8)
    series = trend + season20 + season8
    return pd.Series(series, index=pd.RangeIndex(start=0, stop=n, step=1))

def plot_results(original, forecast, true_future, title):
    plt.figure(figsize=(10, 4))
    # исходный ряд
    plt.plot(original.index, original.values, label="Original")
    # прогноз
    plt.plot(forecast.index, forecast["yhat"].values, label="Forecast")
    # реальные значения
    plt.plot(true_future.index, true_future.values, label="True future", linestyle="dashed")
    plt.title(title)
    plt.legend()
    plt.show()

# --- тесты ---

def test_sarimax_model_on_5y_trend():
    series, true_future = generate_trend_load()
    model = SARIMAXModel(SeriesParams(trend=None,
                                      seasonals=None,
                                      scores=None,
                                      source_series=series,
                                      final_resid_mae=None), SarimaxParams(order=(1,1,1), seasonal_order=(1,1,1,52), trend="t"))
    forecast = model.forecast(PredictParams(horizon=prediction_len))
    assert not forecast.empty
    plot_results(series, forecast, true_future,"SARIMAX on 5y trend-like data")

def test_arima_on_arima_like():
    series, true_future = generate_arima_like(n=seria_size, tail_size=prediction_len)
    model = ARIMAModel(series, ARIMAParams())
    forecast = model.forecast(PredictParams(horizon=prediction_len, freq="D"))
    assert not forecast.empty
    plot_results(series, forecast, true_future,"ARIMA on ARIMA-like data")

def test_arima_on_log_seasonal():
    series = generate_log_seasonal()
    true_future = generate_log_seasonal(n=len(series)+prediction_len).tail(prediction_len)
    model = ARIMAModel(series, ARIMAParams(sig_lag=20))
    forecast = model.forecast(PredictParams(horizon=prediction_len, freq="D"))
    assert not forecast.empty
    plot_results(series, forecast, true_future, "ARIMA on log+seasonal data")

def test_prophet_on_arima_like():
    series = generate_arima_like()
    true_future = generate_arima_like(n=len(series)+prediction_len).tail(prediction_len)
    df = series.copy()
    df.index = pd.date_range("2020-01-01", periods=len(series), freq="D")
    model = ProphetModel(df, ProphetParams())
    forecast = model.forecast(PredictParams(horizon=prediction_len, freq="D"))
    assert not forecast.empty
    plot_results(series, forecast, true_future, "Prophet on ARIMA-like data")

def test_prophet_on_log_seasonal():
    series = generate_log_seasonal()
    true_future = generate_log_seasonal(n=len(series)+prediction_len).tail(prediction_len)
    df = series.copy()
    df.index = pd.date_range("2020-01-01", periods=len(series), freq="D")
    model = ProphetModel(df, ProphetParams())
    forecast = model.forecast(PredictParams(horizon=prediction_len, freq="D"))
    assert not forecast.empty
    plot_results(series, forecast, true_future, "Prophet on log+seasonal data")

def test_lstm_on_5y_trend():
    series, true_future = generate_trend_load()
    model = LSTMModel(series, LSTMParams(window=10, horizon=prediction_len, units=lstm_units, epochs=lstm_epochs))
    forecast = model.forecast(PredictParams(horizon=prediction_len, freq="D"))
    assert not forecast.empty
    plot_results(series, forecast, true_future, "LSTM on 5y trend-like data")

def test_lstm_on_arima_like():
    series = generate_arima_like()
    true_future = generate_arima_like(n=len(series)+prediction_len).tail(prediction_len)
    model = LSTMModel(series, LSTMParams(window=10, horizon=prediction_len, units=lstm_units, epochs=lstm_epochs))
    forecast = model.forecast(PredictParams(horizon=prediction_len))
    assert not forecast.empty
    plot_results(series, forecast, true_future, "LSTM on ARIMA-like data")

def test_lstm_on_log_seasonal():
    series = generate_log_seasonal()
    true_future = generate_log_seasonal(n=len(series)+prediction_len).tail(prediction_len)
    model = LSTMModel(series, LSTMParams(window=10, horizon=prediction_len, units=lstm_units, epochs=lstm_epochs))
    forecast = model.forecast(PredictParams(horizon=prediction_len))
    assert not forecast.empty
    plot_results(series, forecast, true_future, "LSTM on log+seasonal data")
