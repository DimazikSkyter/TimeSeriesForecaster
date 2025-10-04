import os
from typing import List

import pytest
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")  # чтобы не открывать окна во время pytest
import matplotlib.pyplot as plt
from matplotlib import gridspec

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stl.mstl import MSTL

from forecaster_main.predictor.predictor_service import SingleSeriesModelPredictor, ForecastParams
from forecaster_main.predictor.models import PredictParams
from forecaster_main.infrastructure.logging import LoggingParams

# ---------- утилиты ----------

OUTDIR = "tests_output"
os.makedirs(OUTDIR, exist_ok=True)


def _safe_mstl(series: pd.Series, periods: list[int]):
    """ Универсальный вызов MSTL для разных версий statsmodels. """
    periods = sorted(int(p) for p in periods)
    try:
        return MSTL(series, periods=periods, stl_kwargs={'robust': True}).fit()
    except TypeError:
        try:
            return MSTL(series, periods=periods).fit()
        except Exception:
            return STL(series, period=periods[-1], robust=True).fit()


def _visualize_top3(series: pd.Series, params, title: str):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # 1) Оригинал
    params.source_series.plot(ax=axes[0], title=f"Original Series — {title}")

    # 2) Rolling variance
    series.rolling(window=20, min_periods=5).var().plot(ax=axes[1], color="orange")
    axes[1].set_title("Rolling Variance (window=20)")

    # 3) Тренд
    params.trend.plot(ax=axes[2], color="red")
    axes[2].set_title("Trend from analyze_series")

    # Y хотя бы [-1, 1]
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=min(ymin, -1), top=max(ymax, 1))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{title}_top3.png"))
    plt.close(fig)

# ---------- 2) ТОЛЬКО ПРЕДСКАЗАНИЯ (своя ось X, без history) ----------
def _visualize_forecast_only(series: pd.Series, predictor, title: str, truth_func=None,
                             models_names=None):
    if models_names is None:
        models_names = ["sarimax", "arima", "prophet", "lstm", "trend"]
    horizon = len(series) // 4
    predict_params = PredictParams(
        horizon=horizon,
        freq=None,
        models_names=models_names
    )
    df_pred_full = predictor.predict(series, predict_params)

    # берём только прогнозную часть (без origin_series и истории)
    df_pred = df_pred_full.drop(columns=["origin_series"], errors="ignore").tail(horizon)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # правильный индекс для будущего
    if isinstance(series.index, pd.RangeIndex):
        last_x = series.index[-1]
        new_index = pd.RangeIndex(start=last_x + 1, stop=last_x + 1 + horizon)
    else:
        # для дат — просто продолжаем индекс
        last_x = series.index[-1]
        new_index = pd.date_range(start=last_x, periods=horizon+1, freq=predict_params.freq)[1:]

    df_pred.index = new_index

    # true future
    future_truth = None
    if truth_func is not None:
        future_truth = truth_func(series.index[-1], horizon)
        future_truth.index = new_index

    # рисуем предсказания
    for col in df_pred.columns:
        df_pred[col].plot(ax=ax, label=f"pred {col}")

    if future_truth is not None:
        future_truth.plot(ax=ax, label="true future", linestyle="--")

    ax.set_title("Prediction vs Truth (future only)")
    ax.legend()

    ax.set_xlim(left=new_index.min(), right=new_index.max())

    # нормализация Y
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=min(ymin, -1), top=max(ymax, 1))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{title}_forecast.png"))
    plt.close(fig)
# ---------- фикстуры ----------

@pytest.fixture(scope="module")
def predictor():
    params = ForecastParams(
        max_points=1000,
        top_n=6,
        minimum_series_length=10
    )
    logging_params = LoggingParams(level="DEBUG")
    return SingleSeriesModelPredictor(params, logging_params)


@pytest.fixture(scope="module")
def series_var_dispers():
    rng = np.random.default_rng(42)
    n = 800
    base = np.linspace(0, n, n)
    noise = rng.standard_normal(n)
    scale = np.linspace(1.0, 10.0, n)
    y = base + noise * scale
    idx = pd.RangeIndex(start=800, stop=1600)
    series = pd.Series(y, index=idx)

    def gen_future_truth(last_idx, horizon):
        # индекс будущего: 1600..(1600+horizon-1)
        t = np.arange(last_idx + 1, last_idx + horizon + 1)

        # продолжение тренда: 800, 801, ..., 799 + horizon
        trend_future = np.arange(n, n + horizon)

        # шум с растущей дисперсией в будущем
        noise_future = rng.standard_normal(horizon)
        scale_future = np.linspace(scale[-1], scale[-1] * 1.5, horizon)

        y_future = trend_future + noise_future * scale_future
        return pd.Series(y_future, index=t)

    return series, gen_future_truth


@pytest.fixture(scope="module")
def series_two_periods():
    rng = np.random.default_rng(123)
    n = 416
    t = np.arange(n)
    def formula(tt):
        return (400.0
                + 60.0 * np.sin(2 * np.pi * tt / 12)
                + 140.0 * np.sin(2 * np.pi * tt / 52)
                + 2.5 * tt
                + rng.normal(scale=25.0, size=len(tt)))
    y = formula(t)
    idx = pd.RangeIndex(start=0, stop=n)
    series = pd.Series(y, index=idx)

    def gen_future_truth(last_idx, horizon):
        tt = np.arange(last_idx + 1, last_idx + horizon + 1)
        yy = (400.0
              + 60.0 * np.sin(2 * np.pi * tt / 12)
              + 140.0 * np.sin(2 * np.pi * tt / 52)
              + 2.5 * tt)
        return pd.Series(yy, index=tt)

    return series, gen_future_truth


@pytest.fixture(scope="module")
def series_arima():
    rng = np.random.default_rng(2024)
    ar = np.array([1.0, -0.5, 0.25])
    ma = np.array([1.0, 0.4, 0.3])
    arma = ArmaProcess(ar, ma)
    y = arma.generate_sample(nsample=400, scale=5.0, distrvs=rng.standard_normal)
    y = np.cumsum(y) + 50.0
    y = np.clip(y, 0.0, 1200.0)
    idx = pd.RangeIndex(start=0, stop=400)
    series = pd.Series(y, index=idx)

    def gen_future_truth(last_idx, horizon):
        tt = np.arange(last_idx + 1, last_idx + horizon + 1)
        yy = np.linspace(series.iloc[-1], series.iloc[-1] + 5 * horizon, horizon)
        return pd.Series(yy, index=tt)

    return series, gen_future_truth


@pytest.fixture(scope="module")
def series_asymptotic():
    n = 400
    t = np.arange(n)
    trend = 1600.0 - 1200.0 * np.exp(-t / 100.0)
    seasonal = 80.0 * np.sin(2 * np.pi * t / 16.0)
    y = 400.0 + (trend - trend[0]) + seasonal
    idx = pd.RangeIndex(start=0, stop=n)
    series = pd.Series(y, index=idx)

    def gen_future_truth(last_idx, horizon):
        tt = np.arange(last_idx + 1, last_idx + horizon + 1)
        trend_future = 1600.0 - 1200.0 * np.exp(-tt / 100.0)
        seasonal_future = 80.0 * np.sin(2 * np.pi * tt / 16.0)
        yy = 400.0 + (trend_future - trend_future[0]) + seasonal_future
        return pd.Series(yy, index=tt)

    return series, gen_future_truth


# ---------- тесты ----------

def test_var_dispers(predictor, series_var_dispers):
    series, truth_func = series_var_dispers
    params = predictor.analyze_series(series)
    _visualize_top3(series, params, "var_dispers")
    _visualize_forecast_only(series, predictor, "var_dispers", truth_func)


def test_two_periods(predictor, series_two_periods):
    series, truth_func = series_two_periods
    params = predictor.analyze_series(series)
    _visualize_top3(series, params, "two_periods")
    _visualize_forecast_only(series, predictor, "two_periods", truth_func)

def test_two_periods_lstm_only(predictor, series_two_periods):
    series, truth_func = series_two_periods
    params = predictor.analyze_series(series)
    _visualize_top3(series, params, "two_periods_lstm_only")
    _visualize_forecast_only(series, predictor, "two_periods_lstm_only", truth_func,
                             ["lstm"])


def test_arima(predictor, series_arima):
    series, truth_func = series_arima
    params = predictor.analyze_series(series)
    _visualize_top3(series, params, "arima_2_1_2")
    _visualize_forecast_only(series, predictor, "arima_2_1_2", truth_func)


def test_asymptotic(predictor, series_asymptotic):
    series, truth_func = series_asymptotic
    params = predictor.analyze_series(series)
    _visualize_top3(series, params, "asymptotic_16")
    _visualize_forecast_only(series, predictor, "asymptotic_16", truth_func)
