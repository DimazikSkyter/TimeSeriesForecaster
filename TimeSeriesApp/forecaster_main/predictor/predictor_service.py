import asyncio
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stl.mstl import MSTL

from forecaster_main.infrastructure.logging import LoggingParams
from forecaster_main.predictor.models import PredictParams, Model
from forecaster_main.predictor.models import models_factory, SeriesParams, ModelParams


@dataclass
class ForecastParams:
    max_points: int = 1000
    periodogram_max_period: int = 100
    candidate_periods: Set[int] = field(default_factory=lambda: {7, 30, 52, 365})
    top_n: int = 5
    model_creation_timeout: int = 10
    window: int = 10
    minimum_series_length: int = 10


class SingleSeriesModelPredictor:
    """
    Generate multiply predictions for income single seria
    при запуске предикт сначало выполняется вычисление вспомогательных метрик для модели, затем идет
    """

    def __init__(self, params: ForecastParams,
                 logging_params: LoggingParams):
        self.logger = logging_params.configure(self.__class__.__name__)
        self.params = params
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

#Переделать, отфильтровать только периоды меньше 1/3 длины и больше M (передать из параметров), использовать diff(), добавить защиту от соседних точек, лимитируем топ top_n, периоды короче 5 точек не рассматриваем
    def _get_periodogram_candidates(self, series: pd.Series, top_n=5, min_period=2, max_period=None) -> Set[int]:
        freqs, power = periodogram(series.dropna(), scaling="spectrum")
        # if len(freqs) < 5:
        #     return set()
        freqs = freqs[1:]
        powers = power[1:]
        periods = 1 / freqs
        for f, pwr, per in zip(freqs, powers, periods):
            if pwr > 50:
                print(f"freq={f:.4f}, power={pwr:.2f}, period≈{per:.1f} шагов")
        if max_period is None:
            max_period = len(series) // 2
        idx = np.argsort(powers)[::-1]
        candidates = []
        for i in idx[:top_n]:
            p = int(round(periods[i]))
            if min_period <= p <= max_period:
                candidates.append(p)
        return set(candidates)

    def _normalize_periods(self, series: pd.Series, candidate_periods) -> Set[int]:
        n = len(series)
        idx = series.index

        if isinstance(idx, pd.DatetimeIndex):
            freq = pd.infer_freq(idx) or "D"
            if freq.startswith("D"):
                return {7, 30, 365}
            elif freq.startswith("H"):
                return {24, 24 * 7, 24 * 30, 24 * 365}
            elif freq.startswith("T") or freq.startswith("min"):
                return {60, 60 * 24, 60 * 24 * 7, 60 * 24 * 30}
            else:
                return candidate_periods
        else:
            # если индекс числовой → берём относительные циклы
            scale = max(n // 365, 1)
            return {p * scale for p in candidate_periods}

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

    def analyze_series(self, series: pd.Series) -> SeriesParams:
        top_n = getattr(self.params, "top_n", 8)
        season_strength_min = getattr(self.params, "season_strength_min", 0.6)
        min_r2_gain = getattr(self.params, "min_r2_gain", 0.02)
        min_period = 3
        max_period = max(min(int(len(series) * getattr(self.params, "max_period_frac", 0.4)), len(series) // 2),
                         min_period + 1)

        series_ds = self._downsample(series).dropna()
        series_len = len(series_ds)

        if self.is_series_small(series_ds):
            series_params = self.generate_small_series_params(series_ds)
            self.logger.info("Analyze small series completed the compute of series params %s", series_params)
            return series_params

        var_total = np.var(series_ds.values)
        fft_candidates: Set[int] = SingleSeriesModelPredictor.top_fft_periods(series_ds.values, k=top_n,
                                                                              min_p=min_period, max_p=max_period)

        fixed_candidates: Set[int] = self.params.candidate_periods.copy()

        all_candidates = sorted({p for p in set.union(fft_candidates, fixed_candidates) if
                                 min_period <= p <= max_period and series_len >= 2 * p})

        self.logger.info(f"Prepared candidates for cycles are {all_candidates}")

        chosen_periods: List[int] = []
        best_r2 = 0.0
        best_period = None
        best_result = None

        def _strength(resid, comp):
            # fs или ft = max(0, 1 - Var(remainder)/Var(remainder+component))
            v_res = np.var(resid)
            v_all = np.var(resid + comp) + 1e-12
            return float(max(0.0, 1.0 - v_res / v_all))

        # fixed_candidates: Set[int] = self._normalize_periods(series_ds, self.params.candidate_periods)
        # periodogram_candidates: Set[int] = self._get_periodogram_candidates(series_ds, top_n=self.params.top_n)
        # all_candidates = sorted(set.union(fixed_candidates, periodogram_candidates))

        for p in all_candidates:
            res = STL(series_ds, period=int(p), robust=True).fit()
            resid = res.resid.values
            r2 = float(max(0.0, 1.0 - np.var(resid) / (var_total + 1e-12)))
            fs = _strength(resid, res.seasonal.values)
            ft = _strength(resid, res.trend.values)

            # основной лучший период по силе сезонности + R²
            if fs > season_strength_min and r2 > best_r2 + 1e-6:
                best_r2 = r2
                best_period = int(p)
                best_result = res

            # много-сезонный отбор: добавляем, если сильный, не гармоника, и даёт прирост качества
            if fs >= season_strength_min and not self.is_harmonic(int(p), chosen_periods) and (
                    r2 - best_r2 >= min_r2_gain or len(chosen_periods) == 0):
                chosen_periods.append(int(p))
                best_r2 = max(best_r2, r2)

        if len(chosen_periods) > 1:
            res = MSTL(series_ds, periods=sorted(chosen_periods), robust=True).fit()
            trend = res.trend
            resid = res.resid
        elif best_result is not None:
            trend = best_result.trend
            resid = best_result.resid
            chosen_periods = [best_period] if best_period is not None else chosen_periods[:1]
        else:
            # сезонов не нашли — делаем робастный тренд скользящей медианой
            win = max(5, (series_len // 20) | 1)
            trend = series_ds.rolling(win, center=True, min_periods=win // 2).median().interpolate()
            resid = series_ds - trend
            chosen_periods = []

        # вернуть тренд в размер исходной серии (если было downsample),
        # но в данный момент возвращаем даунсемпленный временной ряд
        # trend_full = trend.reindex(series.index).interpolate() if not trend.index.equals(series.index) else trend

        trend = trend.astype(float)
        resid = resid.astype(float)

        x = np.arange(series_len)
        coeffs = np.polyfit(x, trend.values, deg=1)
        trend = pd.Series(np.polyval(coeffs, x), index=series_ds.index)

        r2_final = float(max(0.0, 1.0 - np.var(resid.values) / (var_total + 1e-12)))
        common_idx = series_ds.index.intersection(trend.index)
        corr = float(np.corrcoef(series_ds.loc[common_idx], trend.loc[common_idx])[0, 1])

        series_params: SeriesParams = SeriesParams(
            trend=trend,
            r2=r2_final,
            corr=corr,
            periods=set(chosen_periods),
            min_value=float(series_ds.min()),
            max_value=float(series_ds.max()),
            median_value=float(series_ds.median()),
            mean_value=float(series_ds.mean()),
            source_series=series_ds
        )
        self.logger.info(f"Series params successfully generated:\n{series_params}")
        return series_params

    def is_series_small(self, series: pd.Series):
        return len(series) < self.params.minimum_series_length

    def generate_small_series_params(self, series: pd.Series):
        """
        Consider that small series haven't cycles
        :return: @SeriesParams without periods
        """
        win = max(5, (len(series) // 5) | 1)
        trend = series.rolling(win, center=True, min_periods=win // 2).median().interpolate()
        resid = (series - trend).dropna()
        r2 = SingleSeriesModelPredictor.calculate_resid_determination_coef(series, resid)
        corr = float(np.corrcoef(series.loc[trend.index], trend)[0, 1])
        return SeriesParams(trend=trend, r2=r2, corr=corr, periods=set(),
                            source_series=series)  # .reindex(series.index).interpolate()

    @staticmethod
    def calculate_resid_determination_coef(series: pd.Series, resid: pd.Series) -> float:
        var_total = np.var(series.values)
        return float(max(0.0, 1.0 - np.var(resid.values) / (var_total + 1e-12)))

    @staticmethod
    def is_harmonic(p: int, chosen: List[int], tol: float = 0.12) -> bool:
        for q in chosen:
            r = p / q
            # близко к целому или обратному целому => гармоника
            near_int = abs(r - round(r)) < tol
            near_inv = abs(1 / r - round(1 / r)) < tol
            if near_int or near_inv:
                return True
        return False

    @staticmethod
    def top_fft_periods(x: np.ndarray, k: int, min_p: int, max_p: int) -> Set[int]:
        """Грубый, но быстрый отбор пиков спектра по FFT."""
        n = len(x)
        x = np.asarray(x, dtype=float)
        x = x - np.nanmean(x)
        spec = np.abs(np.fft.rfft(x)) ** 2
        # индекс 0 — постоянная составляющая, пропускаем
        powers = spec[1:]
        ks = np.arange(1, len(spec))
        periods = (n // ks).astype(int)
        mask = (periods >= min_p) & (periods <= max_p)
        periods = periods[mask]
        powers = powers[mask]
        if len(periods) == 0:
            return set()
        # топ k по мощности
        idx = np.argpartition(powers, -min(k, len(powers)))[-min(k, len(powers)):]
        cand = sorted(set(periods[idx]))
        # убираем почти-дубликаты (±10%)
        cand.sort()
        dedup = []
        for p in cand:
            if all(abs(p - q) / q > 0.1 for q in dedup):
                dedup.append(p)
        return set(dedup)
