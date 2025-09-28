import os
import pytest
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("Agg")  # чтобы не открывать окна во время pytest
import matplotlib.pyplot as plt
from scipy.signal import periodogram

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stl.mstl import MSTL
from forecaster_main.predictor.predictor_service import SingleSeriesModelPredictor, ForecastParams
from forecaster_main.infrastructure.logging import LoggingParams
from test_models import seria_size

# ---------- утилиты ----------

OUTDIR = "tests_output"
os.makedirs(OUTDIR, exist_ok=True)

def _safe_mstl(series: pd.Series, periods: list[int]):
    """
    Универсальный вызов MSTL для разных версий statsmodels.
    Порядок попыток:
      1) MSTL(..., stl_kwargs={'robust': True})
      2) MSTL(... )
      3) Fallback: STL по крупнейшему периоду
    """
    periods = sorted(int(p) for p in periods)
    try:
        # новые версии умеют прокидывать параметры STL через stl_kwargs
        return MSTL(series, periods=periods, stl_kwargs={'robust': True}).fit()
    except TypeError:
        try:
            # более старые — без robust
            return MSTL(series, periods=periods).fit()
        except Exception:
            # совсем fallback — STL по крупнейшему периоду
            return STL(series, period=periods[-1], robust=True).fit()


def _visualize(series: pd.Series, params, title: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1) Оригинальный ряд из SeriesParams
    params.source_series.plot(ax=axes[0], title=f"Original Series (from SeriesParams) — {title}")

    # 2) Дисперсия (скользящее окно)
    rolling_var = params.source_series.rolling(window=20, min_periods=5).var()
    rolling_var.plot(ax=axes[1], color="orange", label="Rolling Variance")
    axes[1].legend(loc="best")
    axes[1].set_title("Rolling Variance (window=20)")

    # 3) Тренд из analyze_series
    params.trend.plot(ax=axes[2], color="red", label="analyze_series trend")
    axes[2].legend(loc="best")
    axes[2].set_title("Trend from analyze_series")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{title}.png"))
    plt.close(fig)

    print("SeriesParams:", params)



# ---------- фикстуры ----------

@pytest.fixture(scope="module")
def predictor():
    params = ForecastParams(
        max_points=1000,         # чтобы даунсэмпл не портил короткие ряды
        top_n=6,                 # хватит кандидатов
        minimum_series_length=10 # как в коде по умолчанию
    )
    logging_params = LoggingParams(level="INFO")
    return SingleSeriesModelPredictor(params, logging_params)


@pytest.fixture(scope="module")
def series_var_dispers():
    """
    1) ВР без тренда, меняется дисперсия, 800 точек.
       Индекс ~ [800..1600), чтобы было «y от 800 до 1200» по описанию.
    """
    rng = np.random.default_rng(42)
    n = 800
    base = np.zeros(n)
    noise = rng.standard_normal(n)
    scale = np.linspace(1.0, 10.0, n)   # нарастающая дисперсия
    y = base + noise * scale
    idx = pd.RangeIndex(start=800, stop=1600)  # 800 точек
    return pd.Series(y, index=idx)


@pytest.fixture(scope="module")
def series_two_periods():
    """
    2) Два периода ~12 и 52 шага (сезон и «год») + тренд + белый шум.
       Около 8 лет (n=416), значения в диапазоне примерно [400..1800].
    """
    rng = np.random.default_rng(123)
    n = 416
    t = np.arange(n)
    seasonal1 = 60.0 * np.sin(2 * np.pi * t / 12)     # ~12
    seasonal2 = 140.0 * np.sin(2 * np.pi * t / 52)    # ~52
    trend = 2.5 * t
    noise = rng.normal(scale=25.0, size=n)
    y = 400.0 + seasonal1 + seasonal2 + trend + noise
    idx = pd.RangeIndex(start=0, stop=n)
    return pd.Series(y, index=idx)


@pytest.fixture(scope="module")
def series_arima():
    """
    3) ARIMA(2,1,2): интегрируем ARMA(2,2), старт около 50,
       значения в [0..1200], ~400 точек.
    """
    rng = np.random.default_rng(2024)
    ar = np.array([1.0, -0.5, 0.25])  # phi(B) = 1 - 0.5 B + 0.25 B^2
    ma = np.array([1.0, 0.4, 0.3])    # theta(B) = 1 + 0.4 B + 0.3 B^2
    arma = ArmaProcess(ar, ma)
    y = arma.generate_sample(nsample=400, scale=5.0, distrvs=rng.standard_normal)
    y = np.cumsum(y) + 50.0  # d=1
    y = np.clip(y, 0.0, 1200.0)
    idx = pd.RangeIndex(start=0, stop=400)
    return pd.Series(y, index=idx)


@pytest.fixture(scope="module")
def series_asymptotic():
    """
    4) Ассимптотический тренд к 1600 + период 16, старт ~400, ~400 точек.
    """
    n = 400
    t = np.arange(n)
    trend = 1600.0 - 1200.0 * np.exp(-t / 100.0)
    seasonal = 80.0 * np.sin(2 * np.pi * t / 16.0)
    y = 400.0 + (trend - trend[0]) + seasonal  # старт ~400
    idx = pd.RangeIndex(start=0, stop=n)
    return pd.Series(y, index=idx)


# ---------- тесты ----------

def test_var_dispers(predictor, series_var_dispers):
    params = predictor.analyze_series(series_var_dispers)
    _visualize(series_var_dispers, params, "var_dispers")

#Возвращает 104, а де факто 52 и 12
def test_two_periods(predictor, series_two_periods):

    # freqs, power = periodogram(series_two_periods.diff().dropna(), scaling='spectrum')
    # freqs = freqs[1:]
    # power = power[1:]
    # periods = 1 / freqs
    # for f, pwr, per in zip(freqs, power, periods):
    #     print(f"freq={f:.4f}, power={pwr:.2f}, period≈{per:.1f} шагов")


    periods = predictor._get_periodogram_candidates(series_two_periods)
    print(f"Periodogram show periods {periods}")
    exit(11)

    params = predictor.analyze_series(series_two_periods)
    _visualize(series_two_periods, params, "two_periods")


def test_arima(predictor, series_arima):
    params = predictor.analyze_series(series_arima)
    _visualize(series_arima, params, "arima_2_1_2")


def test_asymptotic(predictor, series_asymptotic):
    params = predictor.analyze_series(series_asymptotic)
    _visualize(series_asymptotic, params, "asymptotic_16")