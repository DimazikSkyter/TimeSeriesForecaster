from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Protocol

import clickhouse_connect
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame


# TODO добавить фильтрацию колонок
# TODO 2 вынести в коммон
@dataclass
class ClickhouseParams:
    ch_host: Optional[str] = None
    ch_port: int = 8123
    ch_user: str = "default"
    ch_password: str = ""
    ch_database: str = "default"

@dataclass
class LoadParams:
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None

    # CSV / Excel
    path: Optional[str] = None
    sep: str = ","
    datetime_col: str = "timestamp"

    # Prometheus / Victoria
    query: Optional[str] = None
    uri: Optional[str] = None
    step: str = "1m"
    timeout_sec: int = 15

    # ClickHouse
    ch_table: str = None

    options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SaveParams:
    # CSV / Excel
    path: Optional[str] = None
    sep: str = ","

    # ClickHouse
    ch_table: str = None

    options: Dict[str, Any] = field(default_factory=dict)


class DataSource(Protocol):
    def load(self, params: LoadParams) -> pd.DataFrame: ...

    def save(self, df: pd.DataFrame, params: SaveParams) -> None: ...


class CsvSource(DataSource):
    def load(self, params: LoadParams) -> pd.DataFrame:
        if not params.path:
            raise ValueError("CSV/Excel path must be provided")

        if params.path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(params.path)
        else:
            df = pd.read_csv(params.path, sep=params.sep)

        # timestamp
        if params.datetime_col in df.columns:
            raw_ts = df[params.datetime_col]

            if pd.api.types.is_numeric_dtype(raw_ts):
                # определяем unit по порядку величины
                max_val = raw_ts.max()
                if max_val > 1e14:      # наносекунды
                    unit = "ns"
                elif max_val > 1e11:    # миллисекунды
                    unit = "ms"
                elif max_val > 1e9:     # секунды с долями
                    unit = "s"
                else:
                    unit = "s"
                ts = pd.to_datetime(raw_ts, unit=unit, errors="coerce", utc=True)
            else:
                ts = pd.to_datetime(raw_ts, errors="coerce", utc=True)

            df = df.drop(columns=[params.datetime_col])
            df.index = ts
        else:
            # если timestamp нет → обычный RangeIndex
            df.index = pd.RangeIndex(start=1, stop=len(df) + 1)

        # фильтры по датам
        if params.start:
            df = df[df.index >= params.start]
        if params.end:
            df = df[df.index <= params.end]

        # все числовые колонки → float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        return df

    def save(self, df: pd.DataFrame, params: SaveParams) -> None:
        if not params.path:
            raise ValueError("CSV/Excel path must be provided")
        subtype = params.path.split(".")[-1].lower()

        if subtype == "csv":
            df.to_csv(params.path, sep=params.sep)
        elif subtype in ("xlsx", "xls"):
            df.to_excel(params.path)
        else:
            raise ValueError("Only csv, xlsx, xls supported")


class PrometheusSource(DataSource):
    def load(self, params: LoadParams) -> pd.DataFrame:
        if not (params.query and params.uri and params.start and params.end):
            raise ValueError("uri, query, start, end required for Prometheus")

        url = f"{params.uri}/api/v1/query_range"
        response = requests.get(url, params={
            "query": params.query,
            "start": int(params.start.timestamp()),
            "end": int(params.end.timestamp()),
            "step": params.step,
        }, timeout=params.timeout_sec)
        response.raise_for_status()
        result = response.json()["data"]["result"]

        frames = []
        for series in result:
            metric_name = series.get("metric", {}).get("__name__", "value")
            tags = ",".join([f"{k}={v}" for k, v in series.get("metric", {}).items() if k != "__name__"])
            col_name = metric_name if not tags else f"{metric_name}{{{tags}}}"

            values = series["values"]  # [[ts, value], ...]
            df = pd.DataFrame(values, columns=["timestamp", col_name])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(float)

            df = df.set_index("timestamp")
            frames.append(df)

        merged = pd.concat(frames, axis=1)
        return merged

    def save(self, df: pd.DataFrame, params: SaveParams) -> None:
        raise NotImplementedError("Prometheus does not support saving")


#todo autoclose?
class ClickHouseSource(DataSource):
    def __init__(self, params: ClickhouseParams):
        self.client = clickhouse_connect.get_client(
            host=params.ch_host,
            port=params.ch_port,
            username=params.ch_user,
            password=params.ch_password,
            database=params.ch_database
        )

    def load(self, params: LoadParams) -> pd.DataFrame:
        if not params.ch_table:
            raise ValueError("ClickHouse  table required")

        query = f"""
            SELECT timestamp, timeseries_name, timeseries_tags, time_series_value
            FROM {params.ch_table}
        """
        if params.start and params.end:
            query += f" WHERE timestamp BETWEEN '{params.start}' AND '{params.end}'"

        df = self.client.query_df(query)

        # комбинируем name+tags в имя колонки
        df["col_name"] = df["timeseries_name"] + df["timeseries_tags"].fillna("")
        pivoted = df.pivot(index="timestamp", columns="col_name", values="time_series_value")
        pivoted.index = pd.to_datetime(pivoted.index)
        return pivoted

    def save(self, df: pd.DataFrame, params: SaveParams) -> None:
        if not params.ch_table:
            raise ValueError("ClickHouse table required")

        records = []
        for col in df.columns:
            for ts, val in df[col].dropna().items():
                records.append((ts, col, "", float(val)))
        self.client.insert(params.ch_table, records,
                      column_names=["timestamp", "timeseries_name", "timeseries_tags", "time_series_value"])
