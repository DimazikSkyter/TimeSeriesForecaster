from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

def simulate_arima_212(n=100, h=10, phi=(0.5, -0.2), theta=(0.4, 0.3), seed=0):
    rng = np.random.default_rng(seed)
    ar = np.r_[1, -np.array(phi)]
    ma = np.r_[1,  np.array(theta)]
    arma = ArmaProcess(ar, ma)
    diff = arma.generate_sample(nsample=n+h, distrvs=rng.standard_normal)
    y = np.cumsum(diff)  # I(1) → ARIMA(2,1,2)
    return pd.Series(y[:n]), pd.Series(y[n:])

def rolling_forecast_with_truth(y_hist: pd.Series, y_future: pd.Series, order=(2,1,2)):
    preds = []
    history = list(y_hist.values)
    for t in range(len(y_future)):
        fit = SM_ARIMA(history, order=order,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(method='statespace')
        yhat = fit.forecast(steps=1)[0]
        preds.append(yhat)
        # КЛЮЧ: обновляем историей ИСТИННЫМ значением, а не прогнозом
        history.append(yhat)
    idx = range(len(y_hist), len(y_hist) + len(y_future))
    return pd.Series(preds, index=idx)

def multi_step_forecast(y_hist: pd.Series, order=(2,1,2), h=10):
    fit = SM_ARIMA(y_hist, order=order,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(method='statespace')
    return pd.Series(fit.get_forecast(steps=h).predicted_mean,
                     index=range(len(y_hist), len(y_hist)+h))

# --- demo ---
hist, fut = simulate_arima_212()
fc_multi = multi_step_forecast(hist, order=(2,1,2), h=len(fut))          # будет «линия»
fc_roll  = rolling_forecast_with_truth(hist, fut, order=(2,1,2))         # «живая» кривая

plt.figure(figsize=(12,5))
plt.plot(hist.index, hist.values, label="Original")
plt.plot(fc_multi.index, fc_multi.values, label="Forecast multi-step")
plt.plot(fc_roll.index,  fc_roll.values,  label="Forecast rolling (1-step, with truth)")
plt.plot(range(len(hist), len(hist)+len(fut)), fut.values, "--", label="True future")
plt.title("ARIMA(2,1,2): multi-step (flat) vs rolling one-step (realistic)")
plt.legend(); plt.show()
