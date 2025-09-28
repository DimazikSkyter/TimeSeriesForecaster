import time
from dataclasses import dataclass, asdict, field
from typing import Protocol, Optional, Any, Set

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from prophet import Prophet
from statsmodels.api import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecaster_main.infrastructure.logging import LoggingParams


@dataclass
class PredictParams:
    horizon: int = 14
    freq: Optional[str] = None
    models_names: list[str] = field(default_factory=list)


@dataclass
class SarimaxParams:
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12)
    trend: str = None


@dataclass
class ARIMAParams:
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] | None = None  # например (P, D, Q, s)


@dataclass
class ProphetParams:
    pass


@dataclass
class LSTMParams:
    window: int
    horizon: int
    units: int = 50
    epochs: int = 20


@dataclass
class TrendParams:
    pass


@dataclass
class SeriesParams:  # обновите под ваш проект, если нужно
    trend: pd.Series
    r2: float  # Percent of noise variance in total variance
    corr: float
    periods: Set[int]
    source_series: pd.Series = field(repr=False)
    min_value: float
    max_value: float
    median_value: float
    mean_value: float


@dataclass
class ModelParams:
    raw: dict[str, Any]

    def to(self, cls):
        """Преобразовать в конкретный dataclass параметров"""
        return cls(**{k: v for k, v in self.raw.items() if k in cls.__dataclass_fields__})


class Model(Protocol):
    @property
    def name(self) -> str: ...

    def forecast(self, prediction: PredictParams) -> pd.DataFrame: ...

    def avg_predict(self, results: pd.DataFrame) -> pd.Series:
        """
        :param results: result predictions
        :return: avg for each row (models mean for each point)
        """
        return results.mean(axis=1)


# todo сжать ARIMA и SARIMAX в общую модель
class SARIMAXModel(Model):
    def __init__(self, series_params: SeriesParams,
                 model_params: SarimaxParams,
                 logging_params: LoggingParams):
        self.series = series_params.source_series
        self.order = model_params.order
        self.seasonal_order = model_params.seasonal_order
        self.logger = logging_params.configure(self.__class__.__name__)

        self.logger.info(f"Try to fit model {self.name} with model params {model_params}")
        start_time = time.perf_counter()

        self.model_fit = SARIMAX(
            self.series,
            order=self.order,
            trend=model_params.trend,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"Fit model {self.name} successfully in {elapsed:.2f} seconds")

    @property
    def name(self) -> str:
        return f"SARIMAX{self.order}x{self.seasonal_order}"

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        fcst = self.model_fit.forecast(steps=prediction.horizon)
        fcst_index = pd.date_range(
            start=self.series.index[-1] + (self.series.index[1] - self.series.index[0]),
            periods=prediction.horizon,
            freq=self.series.index.freq or pd.infer_freq(self.series.index)
        )
        return pd.DataFrame({"yhat": fcst}, index=fcst_index)


class ARIMAModel(Model):
    def __init__(self, series_params: SeriesParams,
                 model_params: ARIMAParams,
                 logging_params: LoggingParams):
        self.series = series_params.source_series
        self.params = model_params
        self.logger = logging_params.configure(self.__class__.__name__)

        self.logger.info(f"Try to fit model {self.name} with model params {model_params}")
        start_time = time.perf_counter()

        if model_params.seasonal_order:
            self.model_fit = SARIMAX(
                self.series,
                order=model_params.order,
                seasonal_order=model_params.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
        else:
            from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
            self.model_fit = SM_ARIMA(self.series, order=model_params.order).fit()

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"Fit model {self.name} successfully in {elapsed:.2f} seconds")

    @property
    def name(self) -> str:
        if self.params.seasonal_order:
            return f"SARIMAX{self.params.order}x{self.params.seasonal_order}"
        else:
            return f"ARIMA{self.params.order}"

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        fcst = self.model_fit.forecast(steps=prediction.horizon)
        return pd.DataFrame({"yhat": fcst})


class ProphetModel(Model):

    def __init__(self, series_params: SeriesParams,
                 model_params: ProphetParams,
                 logging_params: LoggingParams):
        self.logger = logging_params.configure(self.__class__.__name__)

        self.logger.info(f"Try to fit model {self.name} with model params {model_params}")
        start_time = time.perf_counter()
        df = pd.DataFrame({
            "ds": series_params.source_series.index.tz_localize(None),
            "y": series_params.source_series.values,
        })
        self.model = Prophet(**asdict(model_params))
        self.model.fit(df)

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"Fit model {self.name} successfully in {elapsed:.2f} seconds")

    @property
    def name(self) -> str:
        return "PROPHET"

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=prediction.horizon, freq=prediction.freq)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat"]].tail(prediction.horizon).set_index("ds")


class LSTMModel(Model):

    def __init__(self, series_params: SeriesParams, model_params: LSTMParams):
        self.window = model_params.window
        self.horizon = model_params.horizon
        self.series = series_params.source_series

        # подготавливаем данные (X = окна, y = следующая точка)
        values = self.series.values.reshape(-1, 1)
        X, y = [], []
        for i in range(len(values) - self.window):
            X.append(values[i:i + self.window])
            y.append(values[i + self.window])
        X, y = np.array(X), np.array(y)

        # модель
        model = Sequential()
        model.add(LSTM(model_params.units, activation="relu", input_shape=(self.window, 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")

        model.fit(X, y, epochs=model_params.epochs, verbose=0)
        self.model = model
        self.last_window = values[-self.window:]  # сохраним последние данные

    @property
    def name(self) -> str:
        return "LSTM"

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        preds = []
        window = self.last_window.copy()

        for _ in range(prediction.horizon):
            X = window.reshape(1, self.window, 1)
            yhat = self.model.predict(X, verbose=0)[0, 0]
            preds.append(yhat)
            # сдвигаем окно
            window = np.vstack([window[1:], [yhat]])
        index = pd.date_range(
            start=self.series.index[-1] + (self.series.index[1] - self.series.index[0]),
            periods=prediction.horizon,
            freq=self.series.index.freq or pd.infer_freq(self.series.index)
        )
        return pd.DataFrame({"yhat": preds}, index=index)


class TrendModel(Model):
    def __init__(self, series_params: SeriesParams, model_params: TrendParams):
        y = series_params.source_series.values
        # используем числовой индекс (0, 1, 2, …) вместо дат
        x = np.arange(len(y))
        X = add_constant(x)  # добавляем константу для интерсепта
        self.series = series_params.source_series
        self.model = OLS(y, X).fit()
        self.last_index = x[-1]
        self.freq = series_params.source_series.index.freq or pd.infer_freq(series_params.source_series.index)
        self.start_date = series_params.source_series.index[-1]

    @property
    def name(self):
        return "TREND_LR"

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        future_x = np.arange(self.last_index + 1, self.last_index + prediction.horizon + 1)
        X_future = add_constant(future_x)
        preds = self.model.predict(X_future)

        index = pd.date_range(
            start=self.series.index[-1] + (self.series.index[1] - self.series.index[0]),
            periods=prediction.horizon,
            freq=self.series.index.freq or pd.infer_freq(self.series.index)
        )
        return pd.DataFrame({"yhat": preds}, index=index)


def models_factory(name: str, series_params, model_params: ModelParams) -> Model:
    registry = {
        "sarimax": (SARIMAXModel, SarimaxParams),
        "arima": (ARIMAModel, ARIMAParams),
        "prophet": (ProphetModel, ProphetParams),
        "lstm": (LSTMModel, LSTMParams),
        "trend": (TrendModel, TrendParams),
    }

    try:
        model_cls, param_cls = registry[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown model: {name}")

    return model_cls(series_params, model_params.to(param_cls))
