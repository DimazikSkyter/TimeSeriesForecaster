from pathlib import Path

import requests_mock
import clickhouse_connect
import pandas as pd
import pytest
import subprocess
import time
from forecaster_main.infrastructure.io_data import CsvSource, ClickHouseSource, ClickhouseParams, LoadParams, \
    SaveParams, PrometheusSource

csv_path = Path(__file__).parent / "data.csv"
excel_path = Path(__file__).parent / "data.xlsx"


def test_csv_load_and_save():
    loader = CsvSource()
    params = LoadParams(path=str(csv_path))
    df_loaded = loader.load(params)

    assert "metric1" in df_loaded.columns
    assert "metric2" in df_loaded.columns
    assert isinstance(df_loaded.index, pd.DatetimeIndex)

    loader.save(df_loaded, SaveParams(path=str(excel_path)))
    df_saved = pd.read_excel(excel_path)
    assert not df_saved.empty



@pytest.fixture(scope="session")
def clickhouse_container():
    container_name = "test-clickhouse"
    subprocess.run([
        "docker", "run", "--rm", "-d",
        "--name", container_name,
        "-p", "8123:8123", "-p", "9000:9000",
        "clickhouse/clickhouse-server:23.8"
    ], check=True)

    time.sleep(10)  # ждем старт
    yield
    subprocess.run(["docker", "stop", container_name], check=True)


@pytest.mark.clickhouse
def test_clickhouse_load_and_save(clickhouse_container):
    params: ClickhouseParams = ClickhouseParams()
    source = ClickHouseSource(params)

    client = clickhouse_connect.get_client(
        host="localhost",
        username="default",
        password="",
        database="default"
    )
    client.command("""
        CREATE TABLE timeseries (
            timestamp DateTime,
            timeseries_name String,
            timeseries_tags String,
            time_series_value Float64
        ) ENGINE=MergeTree()
        ORDER BY timestamp
    """)

    # создаем df
    ts = pd.date_range("2024-01-01", periods=10, freq="H")
    df = pd.DataFrame({
        "metric1": range(1, 11),
        "metric2": [i * 10 for i in range(1, 11)]
    }, index=ts)

    # сохраняем
    source.save(df, SaveParams(ch_table="timeseries"))

    # загружаем
    df_loaded = source.load(LoadParams(ch_table="timeseries"))
    assert "metric1" in df_loaded.columns
    assert "metric2" in df_loaded.columns
    assert len(df_loaded) == 10


def test_victoria_load_mock():
    uri = "http://victoria-metrics:8428"
    query = "test_metric"
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-01 01:00:00")

    mock_response = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"__name__": "test_metric", "job": "demo"},
                    "values": [
                        [start.timestamp(), "1"],
                        [end.timestamp(), "2"]
                    ]
                }
            ]
        }
    }

    with requests_mock.Mocker() as m:
        m.get(f"{uri}/api/v1/query_range", json=mock_response)
        loader = PrometheusSource()
        df = loader.load(LoadParams(
            uri=uri, query=query, start=start, end=end
        ))

    assert "test_metric{job=demo}" in df.columns
    assert df.iloc[0, 0] == 1.0
    assert df.iloc[1, 0] == 2.0