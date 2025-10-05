from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.stl.mstl import MSTL

from forecaster_main.infrastructure.logging import LoggingParams
from forecaster_main.predictor.models import (
    Model,
    ModelParams,
    PredictParams,
    SeriesParams,
    models_factory,
)


@dataclass
class ForecastParams:
    max_points: int = 2000
    periodogram_max_period: int = 100
    candidate_periods: set[int] = field(default_factory=lambda: {7, 30, 52, 365})
    top_n: int = 5
    model_creation_timeout: int = 120
    window: int = 10
    minimum_series_length: int = 10
    min_cycle_length: int = 5
    min_power: float = 100
    min_season_strength: float = 0.6
    min_r2_stl: float = 0.8
    max_cycles_in_ts: int = 2
    trend_smoothing: str | None = None


class SingleSeriesModelPredictor:
    """
    Generate multiply predictions for income single seria
    при запуске предикт сначало выполняется вычисление вспомогательных метрик для модели, затем идет
    """

    def __init__(self, params: ForecastParams, logging_params: LoggingParams):
        self.logger_params = logging_params
        self.logger = logging_params.configure(self.__class__.__name__)
        self.params = params
        self.models: dict[str, Model] = {}

    def _downsample(self, series: pd.Series) -> pd.Series:
        s = series.copy()
        while len(s) > self.params.max_points:
            factor = int(np.ceil(len(s) / self.params.max_points))
            new_index = np.arange(len(s) // factor)
            s = pd.Series(
                s.values[: len(new_index) * factor]
                .reshape(len(new_index), factor)
                .mean(axis=1),
                index=new_index,
            )
        return s

    # Переделать, отфильтровать только периоды меньше 1/3 длины и больше M (передать из параметров), использовать diff(),
    # добавить защиту от соседних точек, лимитируем топ top_n, периоды короче 5 точек не рассматриваем
    def _get_periodogram_candidates(
        self,
        series: pd.Series,
        top_n: int = 5,
        max_period: int | None = None,
        proximity: float = 0.1,
    ) -> set[int]:
        """
        Поиск кандидатов по periodogram(series.diff()).
        - фильтр по min_cycle_length и max_period
        - агрегация близких периодов (±proximity)
        - возврат top_n лучших по мощности
        """

        x = series.diff().dropna().to_numpy(dtype=float)
        if x.size < self.params.min_cycle_length * 3:
            return set()

        freqs, power = periodogram(x, scaling="spectrum")
        freqs, power = freqs[1:], power[1:]
        periods = 1.0 / freqs

        n = len(x)
        if max_period is None:
            max_period = max(min(n // 5, n // 2), self.params.min_cycle_length + 1)

        mask = (periods >= self.params.min_cycle_length) & (periods <= max_period)
        periods, power = periods[mask], power[mask]

        if len(periods) == 0:
            return set()

        # сортируем по мощности
        cand = sorted(zip(power, periods), key=lambda x: x[0], reverse=True)
        self.logger.debug(f"Candidates are {list(cand)}")

        def combine(c1, c2):
            p1, per1 = c1
            p2, per2 = c2
            total_pwr = p1 + p2
            avg_per = (p1 * per1 + p2 * per2) / total_pwr
            return total_pwr, avg_per

        merged: list[tuple[float, float]] = []
        for pwr, per in cand:
            if pwr < self.params.min_power:
                continue
            merged_flag = False
            for i, (mpwr, mper) in enumerate(merged):
                if abs(mper - per) / per <= proximity:
                    merged[i] = combine((mpwr, mper), (pwr, per))
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append((pwr, per))

        merged.sort(key=lambda x: x[0], reverse=True)
        final = [round(per) for _, per in merged[:top_n]]

        # убрать дубликаты (11 и 12 -> оставим 12)
        dedup: list[int] = []
        for p in sorted(final):
            if dedup and abs(p - dedup[-1]) <= 1:  # близко в 1 шаг
                continue
            dedup.append(p)

        return set(dedup)

    def _normalize_periods(
        self, series: pd.Series, candidate_periods: set[int]
    ) -> set[int]:
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
            datetime_tail = [index[-1] + step * i for i in range(1, extra + 1)]
            return index.append(pd.DatetimeIndex(datetime_tail))
        if pd.api.types.is_numeric_dtype(index.dtype):
            step = index[-1] - index[-2]
            numeric_tail = np.array(
                [index[-1] + step * i for i in range(1, extra + 1)],
                dtype=index.dtype,
            )
            return index.append(pd.Index(numeric_tail, dtype=index.dtype))
        raise TypeError("Поддерживаются только DatetimeIndex и числовые индексы.")

    async def init_models(
        self, series_params: SeriesParams, predict_params: PredictParams
    ) -> dict[str, Model]:
        async def worker(model_name: str) -> tuple[str, Model | None]:
            model_params = ModelParams(
                {**asdict(self.params), **asdict(predict_params)}
            )
            try:
                model = await asyncio.to_thread(
                    models_factory,
                    model_name,
                    series_params,
                    model_params,
                    self.logger_params,
                )
                return model_name, model
            except Exception as e:
                self.logger.exception(f"Model {model_name} skipped due to error")
                return model_name, None

        tasks = [
            asyncio.create_task(worker(name)) for name in predict_params.models_names
        ]
        done, pending = await asyncio.wait(
            tasks, timeout=self.params.model_creation_timeout
        )
        for t in pending:
            t.cancel()
        if pending:
            raise TimeoutError("Model creation exceeded timeout")

        return {
            name: model
            for name, model in (t.result() for t in done)
            if model is not None
        }

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
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):  # more options can be specified also
            self.logger.debug(
                "The final predict df\n###########################\n%s", df
            )
        return df

    def analyze_series(self, series: pd.Series) -> SeriesParams:
        top_n = getattr(self.params, "top_n", 8)
        min_period = 3
        max_period = max(
            min(
                int(len(series) * getattr(self.params, "max_period_frac", 0.4)),
                len(series) // 2,
            ),
            min_period + 1,
        )

        series_ds = self._downsample(series).dropna()
        series_len = len(series_ds)

        if self.is_series_small(series_ds):
            small_series_params = self.generate_small_series_params(series_ds)
            self.logger.info(
                "Analyze small series completed the compute of series params %s",
                small_series_params,
            )
            return small_series_params

        var_total = np.var(series_ds.values)
        periodogram_candidates: set[int] = self._get_periodogram_candidates(
            series_ds, top_n=top_n, max_period=max_period, proximity=0.1
        )

        fixed_candidates: set[int] = (
            self.params.candidate_periods.copy()
        )  # todo пока исключили, подумать, как вернуть

        # self.logger.info("Prepared prior candidates for cycles are %s and fixed candidates are %s",
        #                  periodogram_candidates, fixed_candidates) todo нужно будет раскомментировать, но сделать дебаг

        chosen_periods: list[int] = []
        stl_with_period: list[tuple[DecomposeResult, int]] = (
            self._check_and_accept_candidates(series_ds, periodogram_candidates)
        )
        self.logger.info("Chosen period is %s", stl_with_period)

        if len(stl_with_period) > 1:
            chosen_periods = [stl_and_period[1] for stl_and_period in stl_with_period]
            res = MSTL(series_ds, periods=chosen_periods).fit()
            trend = pd.Series(res.trend, index=series_ds.index, dtype=float)
            resid = pd.Series(res.resid, index=series_ds.index, dtype=float)
        elif len(stl_with_period) == 1:
            stl_result = stl_with_period[0][0]
            trend = pd.Series(stl_result.trend, index=series_ds.index, dtype=float)
            resid = pd.Series(stl_result.resid, index=series_ds.index, dtype=float)
            chosen_periods = [stl_with_period[0][1]]
        else:
            win = max(5, (series_len // 20) | 1)
            trend = (
                series_ds.rolling(win, center=True, min_periods=win // 2)
                .median()
                .interpolate()
            )
            resid = series_ds - trend
            chosen_periods = []

        # вернуть тренд в размер исходной серии (если было downsample),
        # но в данный момент возвращаем даунсемпленный временной ряд
        # trend_full = trend.reindex(series.index).interpolate() if not trend.index.equals(series.index) else trend

        if self.params.trend_smoothing:
            deg = int(self.params.trend_smoothing.replace("polyfit", ""))
            x = np.arange(series_len)
            coeffs = np.polyfit(x, trend.values, deg=deg)
            trend = pd.Series(np.polyval(coeffs, x), index=series_ds.index)

        trend = trend.astype(float)
        resid = resid.astype(float)

        common_idx = series_ds.index.intersection(trend.index)
        r2_final = float(max(0.0, 1.0 - np.var(resid.values) / (var_total + 1e-12)))
        corr = float(
            np.corrcoef(series_ds.loc[common_idx], trend.loc[common_idx])[0, 1]
        )

        series_params: SeriesParams = SeriesParams(
            trend=trend,
            r2=r2_final,
            corr=corr,
            periods=set(chosen_periods),
            min_value=float(series_ds.min()),
            max_value=float(series_ds.max()),
            median_value=float(series_ds.median()),
            mean_value=float(series_ds.mean()),
            source_series=series_ds,
        )
        self.logger.info(f"Series params successfully generated:\n{series_params}")
        return series_params

    def is_series_small(self, series: pd.Series) -> bool:
        return len(series) < self.params.minimum_series_length

    def generate_small_series_params(self, series: pd.Series) -> SeriesParams:
        """
        Consider that small series haven't cycles
        :return: @SeriesParams without periods
        """
        win = max(5, (len(series) // 5) | 1)
        trend = (
            series.rolling(win, center=True, min_periods=win // 2)
            .median()
            .interpolate()
        )
        resid = (series - trend).dropna()
        r2 = SingleSeriesModelPredictor.calc_r2(series, resid)
        corr = float(np.corrcoef(series.loc[trend.index], trend)[0, 1])
        return SeriesParams(
            trend=trend,
            r2=r2,
            corr=corr,
            periods=set(),
            source_series=series,
            min_value=float(series.min()),
            max_value=float(series.max()),
            median_value=float(series.median()),
            mean_value=float(series.mean()),
        )

    @staticmethod
    def calc_r2(
        total_series: pd.Series | np.ndarray | Sequence[float],
        conditional_series: pd.Series | np.ndarray | Sequence[float],
    ) -> float:
        total = np.asarray(total_series, dtype=float)
        cond = np.asarray(conditional_series, dtype=float)
        var_total = np.var(total) + 1e-12
        var_cond = np.var(cond)
        return float(max(0.0, 1.0 - var_cond / var_total))

    @staticmethod
    def is_harmonic(p: int, chosen: Sequence[int], tol: float = 0.12) -> bool:
        for q in chosen:
            r = p / q
            # близко к целому или обратному целому => гармоника
            near_int = abs(r - round(r)) < tol
            near_inv = abs(1 / r - round(1 / r)) < tol
            if near_int or near_inv:
                return True
        return False

    def _check_and_accept_candidates(
        self, series: pd.Series, candidates: set[int]
    ) -> list[tuple[DecomposeResult, int]]:
        success_candidates: list[tuple[DecomposeResult, int, float]] = []
        for candidate in candidates:
            res: DecomposeResult = STL(series, period=int(candidate), robust=True).fit()
            resid = res.resid.values
            r2 = SingleSeriesModelPredictor.calc_r2(series, resid)
            fs = self.calc_r2(res.seasonal.values + resid, resid)
            ft = self.calc_r2(res.trend.values + resid, resid)

            if fs > self.params.min_season_strength and r2 > self.params.min_r2_stl:
                success_candidates.append((res, candidate, r2))

        ordered = sorted(success_candidates, key=lambda x: x[2])
        trimmed = ordered[: self.params.max_cycles_in_ts]
        return [(res, candidate) for res, candidate, _ in trimmed]

    # @staticmethod
    # def top_fft_periods(x: np.ndarray, k: int, min_p: int, max_p: int) -> Set[int]:
    #     """Грубый, но быстрый отбор пиков спектра по FFT."""
    #     n = len(x)
    #     x = np.asarray(x, dtype=float)
    #     x = x - np.nanmean(x)
    #     spec = np.abs(np.fft.rfft(x)) ** 2
    #     # индекс 0 — постоянная составляющая, пропускаем
    #     powers = spec[1:]
    #     ks = np.arange(1, len(spec))
    #     periods = (n // ks).astype(int)
    #     mask = (periods >= min_p) & (periods <= max_p)
    #     periods = periods[mask]
    #     powers = powers[mask]
    #     if len(periods) == 0:
    #         return set()
    #     # топ k по мощности
    #     idx = np.argpartition(powers, -min(k, len(powers)))[-min(k, len(powers)):]
    #     cand = sorted(set(periods[idx]))
    #     # убираем почти-дубликаты (±10%)
    #     cand.sort()
    #     dedup = []
    #     for p in cand:
    #         if all(abs(p - q) / q > 0.1 for q in dedup):
    #             dedup.append(p)
    #     return set(dedup)
