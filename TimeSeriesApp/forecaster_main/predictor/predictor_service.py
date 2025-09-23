import asyncio
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL

from forecaster_main.predictor.models import PredictParams, Model
from forecaster_main.predictor.models import models_factory, SeriesParams, ModelParams


@dataclass
class ForecastParams:
    max_points: int = 1000
    periodogram_max_period: int = 100
    candidate_periods: Tuple[int, ...] = field(default_factory=lambda: (7, 30, 52, 365))
    top_n: int = 5
    model_creation_timeout: int = 10
    window: int = 10

class SingleSeriesModelPredictor:
    """
    Generate multiply predictions for income single seria
    при запуске предикт сначало выполняется вычисление вспомогательных метрик для модели, затем идет
    """

    def __init__(self, params: ForecastParams):
        self.params = params
        self.downsampled_series = None
        self.models: Dict[str, Model] = {}

    def _downsample(self, series: pd.Series) -> pd.Series:
        s = series.copy()
        while len(s) > self.params.max_points:
            factor = int(np.ceil(len(s) / self.params.max_points))
            new_index = np.arange(len(s) // factor)
            s = pd.Series(
                s.values[:len(new_index) * factor].reshape(len(new_index), factor).mean(axis=1),
                index=new_index
            )
        return s

    def _get_periodogram_candidates(self, series: pd.Series, top_n=5, min_period=2, max_period=None):
        freqs, power = periodogram(series, scaling="spectrum")
        if len(freqs) < 2:
            return []
        periods = 1 / freqs[1:]
        powers = power[1:]
        if max_period is None:
            max_period = len(series) // 2
        idx = np.argsort(powers)[::-1]
        candidates = []
        for i in idx[:top_n]:
            p = int(round(periods[i]))
            if min_period <= p <= max_period:
                candidates.append(p)
        return candidates

    def _normalize_periods(self, series: pd.Series, candidate_periods):
        n = len(series)
        idx = series.index

        if isinstance(idx, pd.DatetimeIndex):
            freq = pd.infer_freq(idx) or "D"
            if freq.startswith("D"):
                return [7, 30, 365]
            elif freq.startswith("H"):
                return [24, 24 * 7, 24 * 30, 24 * 365]
            elif freq.startswith("T") or freq.startswith("min"):
                return [60, 60 * 24, 60 * 24 * 7, 60 * 24 * 30]
            else:
                return candidate_periods
        else:
            # если индекс числовой → берём относительные циклы
            scale = max(n // 365, 1)
            return [p * scale for p in candidate_periods]

    def _prepare_index(self, series: pd.Series, extra: int) -> pd.Index:
        index = series.index
        if extra <= 0:
            return index
        if len(index) < 2:
            raise ValueError("Нужно минимум 2 точки, чтобы определить шаг.")
        if isinstance(index, pd.DatetimeIndex):
            step = index[-1] - index[-2]
            tail = [index[-1] + step * i for i in range(1, extra + 1)]
            return index.append(pd.DatetimeIndex(tail))
        if pd.api.types.is_numeric_dtype(index.dtype):
            step = index[-1] - index[-2]
            tail = np.array([index[-1] + step * i for i in range(1, extra + 1)])
            return index.append(pd.Index(tail, dtype=index.dtype))
        raise TypeError("Поддерживаются только DatetimeIndex и числовые индексы.")

    async def init_models(self, series_params: SeriesParams, predict_params: PredictParams) -> dict[str, Model]:
        async def worker(model_name: str) -> tuple[str, Model]:
            model_params = ModelParams({**asdict(self.params), **asdict(predict_params)})
            model = await asyncio.to_thread(models_factory, model_name, series_params, model_params)
            return model_name, model

        tasks = [asyncio.create_task(worker(name)) for name in predict_params.models_names]
        done, pending = await asyncio.wait(tasks, timeout=self.params.model_creation_timeout)
        for t in pending:
            t.cancel()
        if pending:
            raise TimeoutError("Model creation exceeded timeout")
        return {name: model for name, model in (t.result() for t in done)}

    def predict(self, series: pd.Series, predict_params: PredictParams) -> pd.DataFrame:
        if not predict_params.models_names:
            raise ValueError("Minimum one model required.")
        if series.empty:
            raise ValueError("Series is empty.")
        series_params = self.analyze_series(series)
        self.models = asyncio.run(self.init_models(series_params, predict_params))
        index = self._prepare_index(series, predict_params.horizon)
        df = pd.DataFrame(index=index)
        df["origin_series"] = series.reindex(index)
        for name, model in self.models.items():
            fcst = model.forecast(predict_params)
            df[name] = fcst.reindex(index)["yhat"]
        return df

    def analyze_series(self, series: pd.Series):
        series_ds = self._downsample(series)
        self.downsampled_series = series_ds.copy()
        fixed_candidates = self._normalize_periods(series_ds, self.params.candidate_periods)
        periodogram_candidates = self._get_periodogram_candidates(series_ds, top_n=self.params.top_n)
        all_candidates = sorted(set(fixed_candidates + periodogram_candidates))
        seasonals = {}
        resid = series_ds.copy()
        trend = None
        scores = {}
        for period in all_candidates:
            if period < 3 or len(series_ds) < 2 * period:
                continue
            stl = STL(series_ds, period=int(period), robust=True)
            result = stl.fit()
            seasonals[period] = result.seasonal
            trend = result.trend
            resid = result.resid
            resid_mae = float(np.mean(np.abs(resid)))
            r2 = 1 - np.var(resid) / np.var(series_ds)
            scores[period] = dict(resid_mae=resid_mae, r2=r2)
        return SeriesParams(
            trend=trend,
            seasonals=seasonals,
            scores=scores,
            final_resid=resid,
            final_resid_mae=float(np.mean(np.abs(resid)))
        )
