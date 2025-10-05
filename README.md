# TimeSeriesForecaster

TimeSeriesForecaster — прототип визуализатора временных рядов и сервиса прогнозирования, построенный на Streamlit. Проект позволяет загружать данные из различных источников (CSV/Excel, Prometheus/VictoriaMetrics, ClickHouse), анализировать их и строить прогнозы с использованием набора моделей (SARIMAX, ARIMA, Prophet, LSTM, линейный тренд).

## Основные возможности

- загрузка временных рядов из файлов или внешних источников;
- автоматическая нормализация индекса и преобразование числовых колонок;
- анализ ряда и подбор подходящих сезонных периодов;
- прогнозирование с использованием нескольких моделей и визуализация результатов;
- запуск через Streamlit, CLI и Docker.

## Технологический стек

- Python 3.12+
- Streamlit, Plotly — пользовательский интерфейс и визуализация
- Pandas, NumPy, Statsmodels, Prophet, TensorFlow — анализ и моделирование
- Poetry — управление зависимостями
- Pytest — тестирование
- MyPy, Black, Isort — контроль качества кода

## Структура репозитория

```text
TimeSeriesForecaster/
├── Dockerfile                     # Контейнер для запуска приложения и проверок
├── Makefile                       # Часто используемые команды разработки
├── README.md                      # Документация проекта
├── TimeSeriesApp/                 # Исходный код приложения
│   ├── forecaster_main/           # Основной пакет (логика, UI, визуализация)
│   └── forecaster_test/           # Набор тестовых данных и экспериментов
├── pyproject.toml / poetry.lock   # Управление зависимостями через Poetry
└── TimeSeriesForecasterSchema.svg # Схема архитектуры
```

## Требования

- Python 3.12 или новее
- [Poetry](https://python-poetry.org/docs/#installation) версии 1.8+
- Docker (опционально, для контейнерного запуска)

## Установка и локальный запуск

```bash
poetry install --with dev
poetry run streamlit run TimeSeriesApp/forecaster_main/ui/app.py
```

После запуска откройте `http://localhost:8501` в браузере и выберите источник данных.

### Работа с Makefile

Для удобства добавлен `Makefile`, объединяющий основные команды разработки:

| Команда               | Описание |
|-----------------------|----------|
| `make install`        | Установка зависимостей (основные + dev) через Poetry |
| `make format`         | Автоформатирование кода (`isort`, затем `black`) |
| `make lint`           | Проверка качества кода: `isort --check`, `black --check`, `mypy` |
| `make mypy`           | Отдельный запуск MyPy на пакете `forecaster_main` |
| `make test`           | Запуск юнит-тестов Pytest |
| `make check`          | Комплексная проверка (линтеры + тесты) |
| `make docker-build`   | Сборка Docker-образа `timeseries-forecaster` |
| `make docker-run`     | Запуск контейнера с публикацией UI на `8501` порту |
| `make docker-test`    | Выполнение тестов внутри уже собранного контейнера |

## Типизация и стиль кода

Проект покрыт статической типизацией с помощью MyPy. Конфигурация расположена в `pyproject.toml` и проверяет пакет `TimeSeriesApp/forecaster_main`.

- Запуск: `make mypy`
- Правила: включены `check_untyped_defs`, `disallow_incomplete_defs`, `ignore_missing_imports`

Для единообразия стиля используются `black` и `isort` (профиль Black). Конфигурация также хранится в `pyproject.toml`.

## Тесты

Юнит-тесты расположены в каталоге `TimeSeriesApp/forecaster_test`. Для их запуска выполните:

```bash
make test
```

По умолчанию будут использованы зависимости, установленные через Poetry.

## Docker

В репозитории находится готовый `Dockerfile`, позволяющий запустить приложение и выполнять проверки.

Сборка образа:

```bash
make docker-build
```

Запуск Streamlit-приложения в контейнере:

```bash
make docker-run
```

Аргумент сборки `INSTALL_DEV` (значение по умолчанию `true`) управляет установкой dev-зависимостей. Для облегчённого образа можно собрать его так:

```bash
docker build -t timeseries-forecaster --build-arg INSTALL_DEV=false .
```

Запуск тестов внутри контейнера (использует dev-зависимости):

```bash
make docker-test
```

## Настройки логирования

Уровень логирования можно переопределить переменной окружения `LOGGING_LEVEL`. Пример для Docker:

```bash
docker run --rm -p 8501:8501 -e LOGGING_LEVEL=DEBUG timeseries-forecaster
```

## Дополнительно

- Конфигурация параметров прогнозирования лежит в `TimeSeriesApp/forecaster_main/ui/forecaster_params.json`.
- Диаграмма архитектуры (`TimeSeriesForecasterSchema.svg`) помогает понять высокоуровневую схему приложения.
- Для экспериментов и воспроизводимости тестовых данных см. каталог `forecaster_test`.

Приятной работы с TimeSeriesForecaster!