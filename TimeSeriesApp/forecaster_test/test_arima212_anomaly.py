# test_arima212_anomaly.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

# --- 1) генерация ARIMA(2,1,2) с «аномалией»: после 80% добавляется тренд 2*x ---
def simulate_arima_212_with_trend_break(
        n=200, h=20, phi=(0.5, -0.2), theta=(0.4, 0.3),
        change_at=0.6, slope=3.0, seed=0
):
    """
    Первые floor(n*change_at) точек: ARIMA(2,1,2)
    Начиная с change_idx: ARIMA(2,1,2) + линейный тренд 2*x (непрерывно состыковано)
    """
    rng = np.random.default_rng(seed)
    ar = np.r_[1, -np.array(phi)]
    ma = np.r_[1,  np.array(theta)]
    arma = ArmaProcess(ar, ma)

    # генерим дифференцированный ARMA и интегрируем → ARIMA(2,1,2)
    diff = arma.generate_sample(nsample=n + h, distrvs=rng.standard_normal)
    y = np.cumsum(diff)

    # точка излома
    change_idx = int(np.floor(n * change_at))
    x = np.arange(n + h)

    # строим тренд, начинающийся ровно в change_idx и «склеенный» по уровню
    trend = np.zeros_like(y)
    trend[change_idx:] = slope * (x[change_idx:] - x[change_idx])

    # финальный ряд «с изломом»
    y_broken = y + trend

    hist = pd.Series(y_broken[:n])
    fut  = pd.Series(y_broken[n:])
    return hist, fut, change_idx

# --- 2) прогнозы ---
def forecast_multi_step(y_hist: pd.Series, order=(2,1,2), h=20):
    fit = SM_ARIMA(
        y_hist, order=order,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(method='statespace')
    pm = fit.get_forecast(steps=h).predicted_mean
    return pd.Series(pm, index=range(len(y_hist), len(y_hist)+h))

def forecast_recursive(y_hist: pd.Series, h=20, order=(2,1,2)):
    """
    Продакшн-режим: подставляем собственные прогнозы вместо фактов (truth недоступна).
    """
    history = list(y_hist.values)
    preds = []
    for i in range(h):
        fit = SM_ARIMA(
            history, order=order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(method='statespace')
        yhat = fit.forecast(steps=1)[0]
        preds.append(yhat)
        history.append(yhat)  # recursive
    return pd.Series(preds, index=range(len(y_hist), len(y_hist)+h))


n, h = 100, 50
hist, fut, change_idx = simulate_arima_212_with_trend_break(
    n=n, h=h, change_at=0.8, slope=2.0, seed=42
)
fc_multi = forecast_multi_step(hist, order=(2,1,2), h=h)
fc_recur = forecast_recursive(hist, h=h, order=(2,1,2))
# график
plt.figure(figsize=(12,5))
plt.plot(hist.index, hist.values, label="Original (with break)")
plt.axvline(change_idx, color="gray", linestyle=":", label="break @ 80%")
plt.plot(fc_multi.index, fc_multi.values, "-.", label="Forecast multi-step")
plt.plot(fc_recur.index, fc_recur.values, ".-", label="Forecast recursive")
plt.plot(range(n, n+h), fut.values, "--", label="True future")
plt.title("ARIMA(2,1,2) с изломом: после 80% добавлен тренд 2*x")
plt.legend(); plt.tight_layout(); plt.show()
