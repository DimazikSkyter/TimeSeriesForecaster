import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_series(
        years=5,
        points_per_day=24,
        trend_coef=1.4,            # y = a * x  (x в днях)
        start_date="2020-06-01",
        base_level=200.0,          # базовый уровень (защита от отрицательных значений)
        daily_amp=250.0,           # амплитуда дневной сезонности
        yearly_amp_scale=0.35,     # модуляция амплитуды по году
        noise_frac0=0.002,         # начальный уровень шума
        noise_frac_growth=0.00002, # рост шума с течением времени
        peak_mid_target=1200.0,    # целевой пик в середине ряда
        seed=42
):
    rng = np.random.default_rng(seed)
    days = 365 * years
    n = days * points_per_day
    t = np.arange(n)

    # x в днях
    x_days = t / points_per_day

    # тренд
    trend = trend_coef * x_days

    # дневная сезонность (24ч, максимум в середине дня)
    daily = np.sin(2 * np.pi * t / points_per_day - np.pi/2)

    # годовая модуляция амплитуды
    yearly = 1.0 + yearly_amp_scale * np.sin(2 * np.pi * t / (365 * points_per_day) - np.pi/2)

    # амплитуда со временем растёт
    amplitude = daily_amp * yearly * (1.0 + 0.002 * x_days)

    # базовый ряд
    y = base_level + trend + amplitude * daily

    # шум (гетероскедастический, растёт со временем)
    rel_noise = noise_frac0 + noise_frac_growth * x_days
    eps = rng.normal(0.0, 1.0, size=n)
    y = y * (1.0 + rel_noise * eps)

    # защита от отрицательных
    y = np.maximum(y, 0.0)

    # масштабирование так, чтобы пик в середине ≈ target
    mid = n // 2
    window = slice(max(0, mid - 14*points_per_day), min(n, mid + 14*points_per_day))
    local_max = y[window].max() if np.any(y[window] > 0) else y.max()
    scale = peak_mid_target / local_max if local_max > 0 else 1.0
    y = np.maximum(y * scale, 0.0)

    # временной индекс
    dates = pd.date_range(start_date, periods=n, freq=f"{24//points_per_day}H")
    return pd.Series(y, index=dates, name="value")

def weekly_max(series: pd.Series) -> pd.Series:
    """
    Сжимает ряд, оставляя максимальные значения раз в неделю.
    """
    return series.resample("W").max()

# генерация
s = generate_series()

s_weekly = weekly_max(s)
print("Сжатый ряд:", s_weekly.shape)


def test_daily_load_graph():
    plt.figure(figsize=(18,5))
    #plt.plot(s.index, s.values)
    plt.plot(s_weekly.index, s_weekly.values, "r.-", label="weekly max")
    plt.title("Synthetic series: 5 years, daily seasonality, yearly modulation, growing variance")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

    s_weekly.to_csv("week_max_5y.csv", header=True)
