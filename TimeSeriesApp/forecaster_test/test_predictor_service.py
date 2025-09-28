# tests/test_predictor_service.py
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecaster_main.predictor.predictor_service import (
    SingleSeriesModelPredictor,
    ForecastParams,
    PredictParams,
    SeriesParams,
)
from forecaster_main.infrastructure.io_data import CsvSource, LoadParams
from forecaster_main.infrastructure.logging import LoggingParams

week_max_5y_path = Path(__file__).parent / "week_max_5y.csv"

# ---------- вспомогательные штуки ----------

class FakeModel:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def forecast(self, predict_params: PredictParams) -> pd.DataFrame:
        # вернём просто 0..h-1 и пусть длина ровно horizon
        y = np.arange(predict_params.horizon, dtype=float)
        # индекс нам не важен, т.к. в коде берутся только .values
        return pd.DataFrame({"yhat": y})


def make_series_numeric(n=10):
    return pd.Series(np.arange(n, dtype=float), index=pd.RangeIndex(n))


def make_series_daily(n=10, start="2025-01-01", freq="D"):
    idx = pd.date_range(start=start, periods=n, freq=freq)
    return pd.Series(np.arange(n, dtype=float), index=idx)


def dummy_series_params(series: pd.Series) -> SeriesParams:
    return SeriesParams(
        trend=None,
        seasonals={},
        scores={},
        source_series=series,
        final_resid_mae=float(np.mean(np.abs(series)))
    )

# ---------- _downsample ----------

def test_downsample_no_change():
    p = SingleSeriesModelPredictor(ForecastParams(max_points=100))
    s = make_series_numeric(20)
    out = p._downsample(s)
    assert len(out) == len(s)
    pd.testing.assert_series_equal(out, s)

def test_downsample_reduce_pairs():
    # 10 точек при max_points=5 -> усреднение попарно: (0+1)/2, (2+3)/2, ...
    p = SingleSeriesModelPredictor(ForecastParams(max_points=5))
    s = make_series_numeric(10)
    out = p._downsample(s)
    assert len(out) == 5
    expected = pd.Series([0.5, 2.5, 4.5, 6.5, 8.5], index=np.arange(5))
    pd.testing.assert_series_equal(out, expected)

# ---------- _get_periodogram_candidates ----------

def test_get_periodogram_candidates_short_freqs(monkeypatch):
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_numeric(5)

    # заставим periodogram вернуть длину 1 -> ранний выход []
    import forecaster_main.predictor.predictor_service as mod
    monkeypatch.setattr(mod, "periodogram", lambda series, scaling=None: (np.array([0.0]), np.array([0.0])))

    assert p._get_periodogram_candidates(s) == []

def test_get_periodogram_candidates_basic():
    p = SingleSeriesModelPredictor(ForecastParams())
    n = 200
    t = np.arange(n)
    # синус с периодом примерно 20
    s = pd.Series(np.sin(2 * np.pi * t / 20.0) + 0.1*np.random.RandomState(0).randn(n))
    cands = p._get_periodogram_candidates(s, top_n=3, min_period=2, max_period=100)
    assert 0 < len(cands) <= 3
    assert all(2 <= c <= 100 for c in cands)

# ---------- _normalize_periods ----------

def test_normalize_periods_daily():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_daily(30, freq="D")
    assert p._normalize_periods(s, [7, 30, 365]) == [7, 30, 365]

def test_normalize_periods_hourly():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_daily(48, freq="H")
    assert p._normalize_periods(s, [7, 30]) == [24, 24*7, 24*30, 24*365]

def test_normalize_periods_minute():
    p = SingleSeriesModelPredictor(ForecastParams())
    # 'T' == minutes
    idx = pd.date_range("2025-01-01", periods=120, freq="T")
    s = pd.Series(np.arange(120.0), index=idx)
    assert p._normalize_periods(s, [7, 30]) == [60, 60*24, 60*24*7, 60*24*30]

def test_normalize_periods_unknown_freq_returns_original():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_daily(20, freq="W")  # недельная частота -> попадём в "else: return candidate_periods"
    assert p._normalize_periods(s, [10, 20]) == [10, 20]

def test_normalize_periods_numeric_scaled():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_numeric(1000)  # scale = max(1000 // 365, 1) == 2
    assert p._normalize_periods(s, [7, 30, 52, 365]) == [14, 60, 104, 730]

# ---------- _prepare_index ----------

def test_prepare_index_extra_le_zero_returns_same():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_numeric(5)
    out = p._prepare_index(s, extra=0)
    assert out.equals(s.index)

def test_prepare_index_len_lt2_raises():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_numeric(1)
    with pytest.raises(ValueError):
        _ = p._prepare_index(s, extra=1)

def test_prepare_index_datetime():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_daily(3, freq="D")
    out = p._prepare_index(s, extra=2)
    # шаг = 1 день, ждём +1d и +2d к последней дате
    assert len(out) == 5
    assert out[-1] == s.index[-1] + 2 * (s.index[-1] - s.index[-2])

def test_prepare_index_numeric():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_numeric(3)  # шаг 1
    out = p._prepare_index(s, extra=3)
    assert list(out[-3:]) == [3, 4, 5]

def test_prepare_index_wrong_index_type_raises():
    p = SingleSeriesModelPredictor(ForecastParams())
    idx = pd.Index(["a", "b", "c"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    with pytest.raises(TypeError):
        _ = p._prepare_index(s, extra=1)

# ---------- init_models (async) ----------

@pytest.mark.asyncio
async def test_init_models_success(monkeypatch):
    p = SingleSeriesModelPredictor(ForecastParams(model_creation_timeout=2))
    # мок фабрики
    import forecaster_main.predictor.predictor_service as mod
    def fake_factory(name, series_params, model_params):
        return FakeModel(name)
    monkeypatch.setattr(mod, "models_factory", fake_factory)

    series_params = dummy_series_params(make_series_numeric(10))
    predict_params = PredictParams(horizon=5, models_names=["m1", "m2"])

    res = await p.init_models(series_params, predict_params)
    assert set(res.keys()) == {"m1", "m2"}
    assert all(isinstance(v, FakeModel) for v in res.values())

@pytest.mark.asyncio
async def test_init_models_timeout(monkeypatch):
    # очень маленький таймаут и "длинная" фабрика
    p = SingleSeriesModelPredictor(ForecastParams(model_creation_timeout=0.01))
    import forecaster_main.predictor.predictor_service as mod
    def slow_factory(name, series_params, model_params):
        time.sleep(0.1)  # вызов идёт в to_thread
        return FakeModel(name)
    monkeypatch.setattr(mod, "models_factory", slow_factory)

    series_params = dummy_series_params(make_series_numeric(10))
    predict_params = PredictParams(horizon=5, models_names=["m1", "m2"])

    with pytest.raises(TimeoutError):
        await p.init_models(series_params, predict_params)

# ---------- predict (sync оболочка) ----------

def test_predict_raises_when_no_models():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_numeric(10)
    with pytest.raises(ValueError, match="Minimum one model required"):
        p.predict(s, PredictParams(horizon=3, models_names=[]))

def test_predict_raises_when_series_empty():
    p = SingleSeriesModelPredictor(ForecastParams())
    s = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Series is empty"):
        p.predict(s, PredictParams(horizon=3, models_names=["m"]))

def test_predict_happy_path_with_mocks(monkeypatch):
    p = SingleSeriesModelPredictor(ForecastParams())

    s = make_series_daily(5, freq="D")

    # 1) не хотим тяжёлый анализ → замокаем
    monkeypatch.setattr(p, "analyze_series", lambda series: dummy_series_params(series))

    # 2) чтобы не упереться в несоответствие длин,
    #    вернём индекс длиной ровно horizon
    def fake_prepare_index(series, extra):
        return pd.RangeIndex(start=0, stop=extra)
    monkeypatch.setattr(p, "_prepare_index", fake_prepare_index)

    # 3) init_models → отдаём две фейковые модели
    async def fake_init_models(series_params, predict_params):
        return {"m1": FakeModel("m1"), "m2": FakeModel("m2")}
    monkeypatch.setattr(p, "init_models", fake_init_models)

    out = p.predict(s, PredictParams(horizon=4, models_names=["m1", "m2"]))
    assert list(out.columns) == ["m1", "m2"]
    assert len(out) == 4
    # значения в колонках — 0..3
    assert out["m1"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert out["m2"].tolist() == [0.0, 1.0, 2.0, 3.0]

def test_analyze_with_5y():
    csv_loader = CsvSource()
    df = csv_loader.load(LoadParams(path=str(week_max_5y_path)))
    series = df[["value"]]
    ssmp = SingleSeriesModelPredictor(ForecastParams(), LoggingParams())
    sp: SeriesParams = ssmp.analyze_series(series)
    print("Test for analyze for 5y is finished")


def test_analyze_series_no_valid_periods(monkeypatch):
    # заставим periodogram вернуть пустоту, чтобы остались только фикс-кандидаты
    p = SingleSeriesModelPredictor(ForecastParams(), LoggingParams())
    s = make_series_numeric(10)
    # сделаем такие кандидаты, чтобы все отсеялись (len<2*period)
    p.params.candidate_periods = (100, 200)

    import forecaster_main.predictor.predictor_service as mod
    monkeypatch.setattr(mod, "periodogram", lambda series, scaling=None: (np.array([0.0]), np.array([0.0])))

    sp = p.analyze_series(s)
    assert sp.trend is None
    assert sp.seasonals == {}
    assert sp.final_resid_mae == pytest.approx(float(np.mean(np.abs(sp.source_series))))

def test_analyze_series_with_fake_stl(monkeypatch):
    p = SingleSeriesModelPredictor(ForecastParams())
    s = make_series_numeric(50)

    # один небольшой период, чтобы точно зашёл в цикл
    p.params.candidate_periods = (5,)
    # не хотим реальный STL → подменим
    class FakeResult:
        def __init__(self, series):
            self.seasonal = pd.Series(np.zeros_like(series), index=series.index, dtype=float)
            self.trend = series.rolling(window=3, min_periods=1).mean()
            self.resid = series - self.trend

    class FakeSTL:
        def __init__(self, series, period: int, robust: bool = True):
            self.series = series
            self.period = period
            self.robust = robust
        def fit(self):
            return FakeResult(self.series)

    import forecaster_main.predictor.predictor_service as mod
    monkeypatch.setattr(mod, "STL", FakeSTL)

    sp = p.analyze_series(s)
    # был хотя бы один период
    assert 5 in sp.seasonals
    assert isinstance(sp.final_resid_mae, float)
    assert isinstance(sp.scores[5]["r2"], float)
