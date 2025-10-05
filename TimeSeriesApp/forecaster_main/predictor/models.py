from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.regularizers import regularizers
from prophet import Prophet
from statsmodels.api import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecaster_main.infrastructure.logging import LoggingParams

T = TypeVar("T")


@dataclass
class PredictParams:
    horizon: int = 14
    freq: str | None = None
    models_names: list[str] = field(default_factory=list)
    window: int | None = None


@dataclass
class SarimaxParams:
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12)
    trend: str | None = None


@dataclass
class ARIMAParams:
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] | None = None  # например (P, D, Q, s)


@dataclass
class ProphetParams:
    pass


@dataclass
class LSTMParams:
    horizon: int
    window: int = 20
    units: int = 100
    epochs: int = 200


@dataclass
class TrendParams:
    pass


@dataclass
class SeriesParams:  # обновите под ваш проект, если нужно
    trend: pd.Series = field(repr=False)
    r2: float  # Percent of noise variance in total variance
    corr: float
    periods: set[int]
    source_series: pd.Series = field(repr=False)
    min_value: float
    max_value: float
    median_value: float
    mean_value: float


@dataclass
class ModelParams:
    raw: dict[str, Any]

    def to(self, cls: type[T]) -> T:
        """Преобразовать в конкретный dataclass параметров"""
        valid_fields = getattr(cls, "__dataclass_fields__", {})
        return cls(**{k: v for k, v in self.raw.items() if k in valid_fields})


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
    def __init__(
        self,
        series_params: SeriesParams,
        model_params: SarimaxParams,
        logging_params: LoggingParams,
    ):
        self.series = series_params.source_series
        self.order = model_params.order
        self.logger = logging_params.configure(self.__class__.__name__)
        if series_params.periods:
            best_period = max(series_params.periods)
            self.seasonal_order = (1, 1, 1, best_period)
        else:
            self.seasonal_order = model_params.seasonal_order

        self.logger.info(
            f"Try to fit model {self.name} with model params {model_params}"
        )
        start_time = time.perf_counter()

        self.model_fit = SARIMAX(
            self.series,
            order=self.order,
            trend=model_params.trend,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"Fit model {self.name} successfully in {elapsed:.2f} seconds")

    @property
    def name(self) -> str:
        return f"SARIMAX{self.order}x{self.seasonal_order}"

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        fcst = self.model_fit.forecast(steps=prediction.horizon)
        fcst_index = get_index(self.series, prediction.horizon)
        return pd.DataFrame({"yhat": fcst}, index=fcst_index)


class ARIMAModel(Model):
    def __init__(
        self,
        series_params: SeriesParams,
        model_params: ARIMAParams,
        logging_params: LoggingParams,
    ):
        self.series = series_params.source_series
        self.params = model_params
        self.logger = logging_params.configure(self.__class__.__name__)

        self.logger.info(
            f"Try to fit model {self.name} with model params {model_params}"
        )
        start_time = time.perf_counter()

        if model_params.seasonal_order:
            self.model_fit = SARIMAX(
                self.series,
                order=model_params.order,
                seasonal_order=model_params.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
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
        self.logger.info(f"Forecast with ARIMA horizon={prediction.horizon}")
        start_time = time.perf_counter()
        preds = self.model_fit.forecast(steps=prediction.horizon)
        fcst_index = get_index(self.series, prediction.horizon)
        forecast = pd.DataFrame({"yhat": preds.values}, index=fcst_index)
        elapsed = time.perf_counter() - start_time
        self.logger.info(f"Forecast successfully generated in {elapsed:.2f} seconds")
        return forecast


class ProphetModel(Model):

    def __init__(
        self,
        series_params: SeriesParams,
        model_params: ProphetParams,
        logging_params: LoggingParams,
    ):
        idx = series_params.source_series.index
        if not isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise ValueError(
                "ProphetModel requires a DatetimeIndex or PeriodIndex as series index"
            )

        self.logger = logging_params.configure(self.__class__.__name__)

        self.logger.info(
            f"Try to fit model {self.name} with model params {model_params}"
        )
        start_time = time.perf_counter()
        df = pd.DataFrame(
            {
                "ds": idx.tz_localize(None),
                "y": series_params.source_series.values,
            }
        )
        self.model = Prophet(**asdict(model_params))
        self.model.fit(df)

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"Fit model {self.name} successfully in {elapsed:.2f} seconds")

    @property
    def name(self) -> str:
        return "PROPHET"

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        future = self.model.make_future_dataframe(
            periods=prediction.horizon, freq=prediction.freq if prediction.freq else "D"
        )
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat"]].tail(prediction.horizon).set_index("ds")


class LSTMModel(Model):

    def __init__(
        self,
        series_params: SeriesParams,
        model_params: LSTMParams,
        logging_params: LoggingParams,
    ):
        periods = {p for p in series_params.periods if p is not None}
        based_window = model_params.window or 20
        if periods:
            self.window = max(based_window, max(periods) + 1)
        else:
            self.window = based_window
        self.series = series_params.source_series
        self.epochs = model_params.epochs

        self.logger = logging_params.configure(self.__class__.__name__)

        self.logger.info(
            "Try to fit model %s model params model_params %s", self.name, model_params
        )

        if series_params.trend is not None:
            trend = series_params.trend.reindex(self.series.index).interpolate()
            values = (self.series.values - trend.values).astype("float32")
            self.trend: pd.Series | None = trend
        else:
            values = self.series.values.astype("float32")
            self.trend = None

        self.mu: float = float(values.mean())
        self.sigma: float = float(values.std() + 1e-8)
        values_n = ((values - self.mu) / self.sigma).reshape(-1, 1)

        X, y = [], []
        for i in range(len(values_n) - self.window):
            X.append(values_n[i : i + self.window])
            y.append(values_n[i + self.window])
        X_arr, y_arr = np.array(X), np.array(y)

        model = self._model()

        split = int(len(X_arr) * 0.8)
        cb = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        model.fit(
            X_arr[:split],
            y_arr[:split],
            epochs=self.epochs,
            batch_size=32,
            validation_data=(X_arr[split:], y_arr[split:]),
            shuffle=False,
            verbose=0,
            callbacks=[cb],
        )

        self.model = model
        self.last_window_n: np.ndarray = values_n[-self.window :]

        self.logger.info("Model successfully fited")
        self._print_model(model)

    def _print_model(self, model):
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        summary_str = "\n".join(string_list)
        self.logger.info(
            "\n"
            + "=" * 60
            + "\nMODEL SUMMARY\n"
            + "=" * 60
            + "\n"
            + summary_str
            + "\n"
            + "=" * 60
        )

    @property
    def name(self) -> str:
        return "LSTM"

    def _model(self):
        model = Sequential()
        model.add(
            LSTM(
                32,
                activation="tanh",
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                return_sequences=True,
                input_shape=(self.window, 1),
            )
        )
        model.add(Dropout(0.2))
        model.add(
            LSTM(
                16,
                activation="tanh",
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                return_sequences=False,
            )
        )
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
        return model

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        """
        Recursive forecast for prediction.horizon steps forward.
        """
        preds_resid: list[float] = []
        window_n = self.last_window_n.copy()

        for _ in range(prediction.horizon):
            X = window_n.reshape(1, self.window, -1)
            yhat_n = float(self.model.predict(X, verbose=0)[0, 0])
            preds_resid.append(yhat_n)
            window_n = np.vstack([window_n[1:], [yhat_n]])
        preds_resid_arr = np.array(preds_resid, dtype="float32")
        preds_resid_arr = preds_resid_arr * self.sigma + self.mu
        if self.trend is not None:
            slope = float(self.trend.iloc[-1] - self.trend.iloc[-2])
            trend_future = np.arange(1, prediction.horizon + 1) * slope + float(
                self.trend.iloc[-1]
            )
        else:
            trend_future = np.zeros_like(preds_resid_arr)

        yhat = preds_resid_arr + trend_future
        self.logger.info("yhat is %s", yhat)
        return pd.DataFrame(
            {"yhat": yhat}, index=get_index(self.series, prediction.horizon)
        )


class TrendModel(Model):
    def __init__(
        self,
        series_params: SeriesParams,
        model_params: TrendParams,
        logging_params: LoggingParams,
    ):
        y = series_params.source_series.values
        # используем числовой индекс (0, 1, 2, …) вместо дат
        x = np.arange(len(y))
        X = add_constant(x)  # добавляем константу для интерсепта
        self.series = series_params.source_series
        self.model = OLS(y, X).fit()
        self.last_index = x[-1]
        self.start_date = series_params.source_series.index[-1]

    @property
    def name(self):
        return "TREND_LR"

    def forecast(self, prediction: PredictParams) -> pd.DataFrame:
        future_x = np.arange(
            self.last_index + 1, self.last_index + prediction.horizon + 1
        )
        X_future = add_constant(future_x)
        preds = self.model.predict(X_future)

        fcst_index = get_index(self.series, prediction.horizon)
        return pd.DataFrame({"yhat": preds}, index=fcst_index)


def models_factory(
    name: str,
    series_params: SeriesParams,
    model_params: ModelParams,
    logging_params: LoggingParams,
) -> Model:
    registry = {
        "sarimax": (SARIMAXModel, SarimaxParams),
        "arima": (ARIMAModel, ARIMAParams),
        "prophet": (ProphetModel, ProphetParams),
        "lstm": (LSTMModel, LSTMParams),
        "trend": (TrendModel, TrendParams),
    }

    try:
        model_cls, param_cls = registry[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc

    try:
        return model_cls(series_params, model_params.to(param_cls), logging_params)
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to create model {name}: {e}\n{tb}") from e


def get_index(series: pd.Series, horizon: int) -> pd.Index:
    idx = series.index

    # DatetimeIndex
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or pd.infer_freq(idx)
        if freq is not None:
            step = idx[-1] - idx[-2]
            return pd.date_range(start=idx[-1] + step, periods=horizon, freq=freq)
        # если freq неизвестен, хотя бы по шагу
        step = idx[-1] - idx[-2]
        inferred_future = [idx[-1] + step * i for i in range(1, horizon + 1)]
        return pd.DatetimeIndex(inferred_future)

    # PeriodIndex
    if isinstance(idx, pd.PeriodIndex):
        return pd.period_range(start=idx[-1] + 1, periods=horizon, freq=idx.freq)

    # RangeIndex
    if isinstance(idx, pd.RangeIndex):
        step = idx.step or 1
        start = idx[-1] + step
        stop = start + horizon * step
        return pd.RangeIndex(start=start, stop=stop, step=step)

    # Numeric Index
    if pd.api.types.is_numeric_dtype(idx):
        step = idx[-1] - idx[-2] if len(idx) > 1 else 1
        numeric_future = np.arange(
            idx[-1] + step,
            idx[-1] + step * (horizon + 1),
            step,
        )
        return pd.Index(numeric_future)

    # fallback
    return pd.RangeIndex(start=0, stop=horizon)
