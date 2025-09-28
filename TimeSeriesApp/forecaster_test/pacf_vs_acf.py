import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import periodogram  # если нет, можно заменить на np.fft
from forecaster_test.daily_load_generator import generate_series, weekly_max


def test_pacf():
    # Параметры
    S = 52  # проверь: 52 для годовой недельной сезонности; поставь 56 если уверен
    ACF_LAGS = 200

    series = generate_series()
    week_series = weekly_max(series)  # длина ~261

    # 1. Дифференцирование для устранения тренда
    series_diff = week_series.diff().dropna()
    freqs, power = periodogram(series_diff, scaling='spectrum')
    periods = 1 / freqs[1:]
    powers = power[1:]

    best_period = periods[np.argmax(powers)]
    print(f"Основной период после diff(1): {best_period:.1f} недель")
    # 2. ACF/PACF на разностях
    nlags = min(100, len(series_diff)//2 - 1)

    fig, axes = plt.subplots(2, 1, figsize=(12,6))

    acf_vals = acf(series_diff, nlags=nlags, fft=True)
    axes[0].stem(range(len(acf_vals)), acf_vals)
    axes[0].axhline(0, color='black')
    axes[0].set_title("ACF (diff)")

    pacf_vals = pacf(series_diff, nlags=nlags, method="ywm")
    axes[1].stem(range(len(pacf_vals)), pacf_vals)
    axes[1].axhline(0, color='black')
    axes[1].set_title("PACF (diff)")

    plt.tight_layout()
    plt.show()


    # Спектральный анализ (на исходном ряде)
    freqs, power = periodogram(week_series, scaling='spectrum')

    # Перевод частот в периоды
    periods = 1 / freqs[1:]   # пропускаем freqs[0] = 0
    powers = power[1:]

    best_period = periods[np.argmax(powers)]
    print(f"Максимум спектра ≈ {best_period:.1f} недель")

    plt.figure(figsize=(10,4))
    plt.plot(periods, powers)
    plt.xlim(0, 100)  # смотрим периоды до 100 недель
    plt.xlabel("Период (недели)")
    plt.ylabel("Спектральная мощность")
    plt.title("Periodogram")
    plt.show()

    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(week_series, model="additive", period=52)
    result.plot()
    plt.show()