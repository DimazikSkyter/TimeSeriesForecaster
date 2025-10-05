from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any, Protocol

import clickhouse_connect
import pandas as pd
import requests  # type: ignore[import-untyped]
from requests import RequestException

from forecaster_main.infrastructure.logging import LoggingParams


# TODO добавить фильтрацию колонок
# TODO 2 вынести в коммон
@dataclass
class ClickhouseParams:
    ch_host: str | None = None
    ch_port: int = 8123
    ch_user: str = "default"
    ch_password: str = ""
    ch_database: str = "default"


@dataclass
class LoadParams:
    # CSV / Excel
    path: str | BytesIO | None = None
    sep: str = ","
    datetime_col: str = "timestamp"

    # Prometheus / Victoria
    query: str | None = None
    uri: str | None = None
    step: str = "1m"
    timeout_sec: int = 15
    start: datetime | pd.Timestamp | None = None
    end: datetime | pd.Timestamp | None = None

    # ClickHouse
    ch_table: str | None = None

    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SaveParams:
    # CSV / Excel
    path: str | None = None
    sep: str = ","

    # ClickHouse
    ch_table: str | None = None

    options: dict[str, Any] = field(default_factory=dict)


class DataSource(Protocol):
    def load(self, params: LoadParams) -> pd.DataFrame: ...

    def save(self, df: pd.DataFrame, params: SaveParams) -> None: ...


class CsvSource(DataSource):
    def __init__(self, logging_params: LoggingParams | None = None):
        self._logging_params = logging_params or LoggingParams()
        self.logger = self._logging_params.configure(self.__class__.__name__)

    def load(self, params: LoadParams) -> pd.DataFrame:
        self.logger.info(f"Loading file with params={params}")
        if not params.path:
            raise ValueError("CSV/Excel path must be provided")

        if hasattr(params.path, "read"):
            df = pd.read_csv(params.path, sep=params.sep)
        elif isinstance(params.path, str):
            if params.path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(params.path)
            else:
                df = pd.read_csv(params.path, sep=params.sep)
        else:
            raise ValueError("Invalid path: must be file path or file-like object")

        # timestamp
        if params.datetime_col in df.columns:
            raw_ts = df[params.datetime_col]
            ts = None

            if pd.api.types.is_string_dtype(raw_ts.dtype):
                raw_ts = raw_ts.str.strip()
                pattern_numeric = re.compile("^([0-9]+)+$")
                pattern_date = re.compile(r"^\d{4}-\d{2}-\d{2}$")
                pattern_datetime = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
                if pattern_numeric.match(raw_ts.iloc[0]):
                    raw_ts = pd.to_numeric(raw_ts, errors="coerce")
                elif params.options.get("timestamp_format"):
                    ts = pd.to_datetime(
                        raw_ts,
                        format=params.options["timestamp_format"],
                        errors="coerce",
                        utc=True,
                    )
                elif pattern_date.match(raw_ts.iloc[0]):
                    ts = pd.to_datetime(
                        raw_ts, format="%Y-%m-%d", errors="coerce", utc=True
                    )
                elif pattern_datetime.match(raw_ts.iloc[0]):
                    ts = pd.to_datetime(
                        raw_ts, format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True
                    )

            if pd.api.types.is_numeric_dtype(raw_ts):
                # определяем unit по порядку величины
                max_val = raw_ts.max()
                if max_val > 1e14:  # наносекунды
                    unit = "ns"
                elif max_val > 1e11:  # миллисекунды
                    unit = "ms"
                elif max_val > 1e9:  # секунды с долями
                    unit = "s"
                else:
                    unit = "s"
                ts = pd.to_datetime(raw_ts, unit=unit, errors="coerce", utc=True)
            elif pd.api.types.is_datetime64_any_dtype(raw_ts):
                ts = pd.to_datetime(raw_ts, errors="coerce", utc=True)
            elif ts is None:
                raise ValueError(f"Unsupported data type: {raw_ts.dtype}")

            df = df.drop(columns=[params.datetime_col])
            df.index = ts
        else:
            # если timestamp нет → обычный RangeIndex
            df.index = pd.RangeIndex(start=1, stop=len(df) + 1)

        # фильтры по датам
        if params.options.get("start"):
            df = df[df.index >= params.options.get("start")]
        if params.options.get("end"):
            df = df[df.index <= params.options.get("end")]

        # все числовые колонки → float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        self.logger.debug(f"Loaded dataframe by CsvSource shape={df.shape}")
        return df

    def save(self, df: pd.DataFrame, params: SaveParams) -> None:
        self.logger.info(f"Try to save data with params={params}")
        if not params.path:
            raise ValueError("CSV/Excel path must be provided")
        subtype = params.path.split(".")[-1].lower()

        if subtype == "csv":
            df.to_csv(params.path, sep=params.sep)
        elif subtype in ("xlsx", "xls"):
            df.to_excel(params.path)
        else:
            raise ValueError("Only csv, xlsx, xls supported")

        self.logger.info(f"File successfully saved at path={params.path}")


class PrometheusSource(DataSource):
    def __init__(self, logging_params: LoggingParams | None = None):
        self._logging_params = logging_params or LoggingParams()
        self.logger = self._logging_params.configure(self.__class__.__name__)

    def load(self, params: LoadParams) -> pd.DataFrame:
        self.logger.info(f"Loading file with params={params}")
        if not (params.query and params.uri and params.start and params.end):
            raise ValueError("uri, query, start, end required for Prometheus")

        url = f"{params.uri}/api/v1/query_range"
        try:
            response = requests.get(
                url,
                params={
                    "query": params.query,
                    "start": int(pd.Timestamp(params.start).timestamp()),
                    "end": int(pd.Timestamp(params.end).timestamp()),
                    "step": params.step,
                },
                timeout=params.timeout_sec,
            )
            response.raise_for_status()
        except RequestException as exc:
            self.logger.error("Failed to load data with url %s", url, exc_info=exc)
            raise

        payload = response.json()
        result = payload.get("data", {}).get("result", [])

        frames: list[pd.DataFrame] = []
        for series in result:
            metric_name = series.get("metric", {}).get("__name__", "value")
            tags = ",".join(
                [
                    f"{k}={v}"
                    for k, v in series.get("metric", {}).items()
                    if k != "__name__"
                ]
            )
            col_name = metric_name if not tags else f"{metric_name}{{{tags}}}"

            values = series["values"]  # [[ts, value], ...]
            df = pd.DataFrame(values, columns=["timestamp", col_name])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(float)

            df = df.set_index("timestamp")
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, axis=1)
        self.logger.info(f"Successfully load timeseries from source ")
        return merged

    def save(self, df: pd.DataFrame, params: SaveParams) -> None:
        raise NotImplementedError("Prometheus does not support saving")


# todo autoclose?
class ClickHouseSource(DataSource):
    def __init__(
        self, params: ClickhouseParams, logging_params: LoggingParams | None = None
    ):
        self._logging_params = logging_params or LoggingParams()
        self.logger = self._logging_params.configure(self.__class__.__name__)
        self.clickhouse_params = params
        self.client: Any = clickhouse_connect.get_client(
            host=params.ch_host,
            port=params.ch_port,
            username=params.ch_user,
            password=params.ch_password,
            database=params.ch_database,
        )

    def load(self, params: LoadParams) -> pd.DataFrame:
        if not params.ch_table:
            raise ValueError("ClickHouse  table required")
        self.logger.info(
            f"Try to load data from clickhouse {self.clickhouse_params} with params={params}"
        )
        query = f"""
            SELECT timestamp, timeseries_name, timeseries_tags, time_series_value
            FROM {params.ch_table}
        """
        if params.start and params.end:
            start_ts = pd.Timestamp(params.start).isoformat()
            end_ts = pd.Timestamp(params.end).isoformat()
            query += f" WHERE timestamp BETWEEN '{start_ts}' AND '{end_ts}'"

        df = self.client.query_df(query)

        # комбинируем name+tags в имя колонки
        df["col_name"] = df["timeseries_name"] + df["timeseries_tags"].fillna("")
        pivoted = df.pivot(
            index="timestamp", columns="col_name", values="time_series_value"
        )
        pivoted.index = pd.to_datetime(pivoted.index)
        self.logger.info(
            f"Data successfully loaded from clickhouse for {len(pivoted.index)} rows"
        )
        return pivoted

    def save(self, df: pd.DataFrame, params: SaveParams) -> None:
        self.logger.info(
            f"Trying to save data in clickhouse {self.clickhouse_params} with params={params}"
        )
        if not params.ch_table:
            raise ValueError("ClickHouse table required")

        records: list[tuple[pd.Timestamp, str, str, float]] = []
        for col in df.columns:
            for ts, val in df[col].dropna().items():
                timestamp = pd.Timestamp(ts)
                records.append((timestamp, col, "", float(val)))

        if not records:
            self.logger.info("No records to save in clickhouse")
            return

        self.client.insert(
            params.ch_table,
            records,
            column_names=[
                "timestamp",
                "timeseries_name",
                "timeseries_tags",
                "time_series_value",
            ],
        )

        self.logger.info(
            f"Successfully saved data in clickhouse for {len(records)} rows"
        )
