from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go

def build_timeseries_figure(df: pd.DataFrame) -> go.Figure:
    if "is_forecast" not in df.columns:
        df = df.assign(is_forecast=False)

    observed = df[~df["is_forecast"]]
    predicted = df[df["is_forecast"]]

    fig = go.Figure()
    if not observed.empty:
        fig.add_trace(go.Scatter(
            x=observed.index, y=observed["value"],
            mode="lines", name="Observed"
        ))

    if not predicted.empty:
        fig.add_trace(go.Scatter(
            x=predicted.index, y=predicted["value"],
            mode="lines", name="Forecast",
            line=dict(dash="dash")
        ))

        # --- Вертикальная линия БЕЗ add_vline ---
        x0 = pd.to_datetime(predicted.index[0]).to_pydatetime()
        y_min = float(pd.concat([observed["value"], predicted["value"]]).min())
        y_max = float(pd.concat([observed["value"], predicted["value"]]).max())

        fig.add_trace(go.Scatter(
            x=[x0, x0], y=[y_min, y_max],
            mode="lines",
            name="Forecast start",
            line=dict(width=1, dash="dot"),
            showlegend=False
        ))
        fig.add_annotation(
            x=x0, y=y_max, text="Forecast start",
            showarrow=True, arrowhead=1, yshift=10,
            xref="x", yref="y"
        )
        # --- конец исправления ---

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Value",
        legend_title_text="Series",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig
