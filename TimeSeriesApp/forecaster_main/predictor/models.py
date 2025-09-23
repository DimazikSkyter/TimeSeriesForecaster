from dataclasses import dataclass, asdict, field
from typing import Protocol, Optional, Any

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
class SeriesParams:
    trend: any
    seasonals: any
    scores: any
    final_resid: any
    final_resid_mae: any


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
    def __init__(self, series_params: SeriesParams, model_params: SarimaxParams):
        self.series = series_params.final_resid
        self.order = model_params.order
        self.seasonal_order = model_params.seasonal_order

        self.model_fit = SARIMAX(
            self.series,
            order=self.order,
            trend=model_params.trend,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

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
    def __init__(self, series_params: SeriesParams, model_params: ARIMAParams):
        self.series = series_params.final_resid
        self.params = model_params
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

    def __init__(self, series_params: SeriesParams, model_params: ProphetParams):
        df = pd.DataFrame({
            "ds": series_params.final_resid.index.tz_localize(None),
            "y": series_params.final_resid.values,
        })
        self.model = Prophet(**asdict(model_params))
        self.model.fit(df)

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
        self.series = series_params.final_resid

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


def models_factory(name: str, series_params, model_params: ModelParams) -> Model:
    registry = {
        "sarimax": (SARIMAXModel, SarimaxParams),
        "arima": (ARIMAModel, ARIMAParams),
        "prophet": (ProphetModel, ProphetParams),
        "lstm": (LSTMModel, LSTMParams),
    }

    try:
        model_cls, param_cls = registry[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown model: {name}")

    return model_cls(series_params, model_params.to(param_cls))
