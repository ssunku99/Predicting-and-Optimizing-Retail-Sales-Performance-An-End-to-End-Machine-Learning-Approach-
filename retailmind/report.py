"""Static HTML report generator — self-contained `index.html` for static hosting
(GitHub Pages, Netlify, S3, etc).

Addresses the professor's request to deploy on GitHub Pages / Netlify. Streamlit
needs a Python server which those platforms don't provide. This module renders
the pipeline outputs into a single HTML file that runs entirely client-side via
inline Plotly charts.

Usage:
    from retailmind import RetailPipeline
    from retailmind.report import save_report

    p = RetailPipeline.from_files('train.csv', auxiliary_paths=['store.csv'])
    p.run(sample_entities=50)
    save_report(p, out_dir='docs/')           # → docs/index.html

Then push `docs/` to GitHub and enable Pages from the `/docs` folder.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html


# ============= helpers =============

def _fig_html(fig, div_id: str) -> str:
    return to_html(
        fig, include_plotlyjs="cdn", full_html=False, div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


def _table_html(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "<p><em>No data.</em></p>"
    return df.head(max_rows).to_html(index=False, classes="rm-table", float_format=lambda x: f"{x:,.2f}")


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div><div class="kpi-value">{html.escape(value)}</div></div>'


def _section(title: str, body: str, anchor: Optional[str] = None) -> str:
    a = f' id="{anchor}"' if anchor else ""
    return f'<section{a}><h2>{html.escape(title)}</h2>{body}</section>'


# ============= chart builders =============

def _build_total_sales_chart(canon: pd.DataFrame) -> str:
    total = canon.groupby("date", as_index=False)["sales"].sum()
    fig = px.line(total, x="date", y="sales", title="Total daily sales over time")
    fig.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40))
    return _fig_html(fig, "total-sales")


def _build_dow_chart(seasonality: dict) -> str:
    dow = seasonality.get("dow_mean_sales", {})
    if not dow:
        return ""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df = pd.DataFrame({"dow": [d for d in order if d in dow],
                       "mean_sales": [dow[d] for d in order if d in dow]})
    fig = px.bar(df, x="dow", y="mean_sales", title="Day-of-week seasonality")
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=50, b=40))
    return _fig_html(fig, "dow")


def _build_forecast_chart(canon: pd.DataFrame, forecast: pd.DataFrame,
                          baseline: Optional[pd.DataFrame] = None,
                          n_entities: int = 4) -> str:
    if forecast is None or forecast.empty:
        return ""
    top = (canon.groupby("entity_id")["sales"].sum()
                .reindex(forecast["entity_id"].unique()).nlargest(n_entities).index)
    fig = go.Figure()
    for ent in top:
        h = canon[canon["entity_id"] == ent].tail(90)
        f = forecast[forecast["entity_id"] == ent]
        fig.add_trace(go.Scatter(x=h["date"], y=h["sales"], name=f"{ent} history",
                                  mode="lines", legendgroup=str(ent)))
        fig.add_trace(go.Scatter(x=f["date"], y=f["yhat"], name=f"{ent} forecast",
                                  mode="lines+markers", legendgroup=str(ent),
                                  line=dict(dash="dot")))
    fig.update_layout(title=f"Forecast vs history (top {n_entities} entities)",
                      height=400, margin=dict(l=40, r=20, t=50, b=40),
                      hovermode="x unified")
    return _fig_html(fig, "forecast")


def _build_anomaly_chart(canon: pd.DataFrame, anomalies: pd.DataFrame) -> str:
    if anomalies is None or anomalies.empty:
        return ""
    top_ent = anomalies["entity_id"].value_counts().head(1).index[0]
    series = canon[canon["entity_id"] == top_ent]
    pts = anomalies[anomalies["entity_id"] == top_ent]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["date"], y=series["sales"], mode="lines",
                              name=f"{top_ent} sales"))
    fig.add_trace(go.Scatter(x=pts["date"], y=pts["value"], mode="markers",
                              name="anomaly", marker=dict(color="red", size=10,
                                                          symbol="x")))
    fig.update_layout(title=f"Anomalies for entity '{top_ent}'",
                      height=320, margin=dict(l=40, r=20, t=50, b=40))
    return _fig_html(fig, "anomalies-chart")


def _build_drivers_chart(driver_report) -> str:
    if driver_report is None:
        return ""
    imp = driver_report.importance.head(15).copy()
    imp["importance_norm"] = imp["importance"] / imp["importance"].max()
    fig = px.bar(imp, x="importance_norm", y="feature",
                 color="direction_label", orientation="h",
                 title="Top sales drivers (normalised LightGBM gain)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500, margin=dict(l=120, r=20, t=50, b=40))
    return _fig_html(fig, "drivers")


# ============= main entry =============

REPORT_CSS = """
:root { --bg:#fff; --fg:#111; --muted:#666; --accent:#1d4ed8; --card:#f7f7f8; --border:#e5e7eb; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 0; background: var(--bg); color: var(--fg); line-height: 1.5; }
header { background: linear-gradient(120deg, #1d4ed8, #0ea5e9); color: white; padding: 36px 24px; }
header h1 { margin: 0 0 8px; font-size: 28px; }
header p { margin: 0; opacity: 0.9; }
nav { background: var(--card); border-bottom: 1px solid var(--border); padding: 12px 24px;
      position: sticky; top: 0; z-index: 10; overflow-x: auto; white-space: nowrap; }
nav a { color: var(--accent); margin-right: 20px; text-decoration: none; font-size: 14px; }
nav a:hover { text-decoration: underline; }
main { max-width: 1100px; margin: 0 auto; padding: 24px; }
section { margin-bottom: 36px; }
section h2 { font-size: 20px; border-bottom: 2px solid var(--border); padding-bottom: 6px; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 12px; margin: 16px 0; }
.kpi { background: var(--card); padding: 14px 16px; border-radius: 8px; border: 1px solid var(--border); }
.kpi-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 22px; font-weight: 600; margin-top: 4px; }
table.rm-table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
table.rm-table th, table.rm-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }
table.rm-table th { background: var(--card); position: sticky; top: 0; }
table.rm-table tr:hover { background: #fafafa; }
.callout { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 4px; margin: 12px 0; }
.muted { color: var(--muted); font-size: 13px; }
footer { margin: 60px 24px 24px; color: var(--muted); font-size: 12px; text-align: center; border-top: 1px solid var(--border); padding-top: 18px; }
code { background: var(--card); padding: 2px 5px; border-radius: 3px; font-size: 13px; }
"""


def render_report(pipeline, dataset_name: str = "Dataset") -> str:
    """Render the pipeline's results to a complete self-contained HTML string."""
    if pipeline.canonical is None or pipeline.eda_report is None:
        raise RuntimeError("Pipeline has not been run yet. Call pipeline.run() first.")

    canon = pipeline.canonical
    eda_r = pipeline.eda_report
    o = eda_r["overview"]

    # --- KPIs ---
    kpis = "".join([
        _kpi("Rows", f"{o['rows']:,}"),
        _kpi("Entities", f"{o['entities']:,}"),
        _kpi("Date span", f"{o['span_days']} days"),
        _kpi("Total sales", f"${o['total_sales']:,.0f}"),
        _kpi("Mean daily sales", f"{o['mean_daily_sales']:,.0f}"),
    ])
    kpi_block = f'<div class="kpi-row">{kpis}</div>'

    # --- EDA ---
    eda_html = kpi_block + _build_total_sales_chart(canon) + _build_dow_chart(eda_r.get("seasonality", {}))
    if eda_r.get("promo_lift"):
        pl = eda_r["promo_lift"]
        eda_html += (f'<div class="callout"><strong>Promo lift:</strong> '
                     f'{pl.get("naive_lift_pct", 0):.1f}% (mean sales on promo '
                     f'{pl["mean_sales_on_promo"]:,.0f} vs off-promo '
                     f'{pl["mean_sales_off_promo"]:,.0f})</div>')

    # --- Forecast ---
    fc_html = ""
    if pipeline.forecast_model and pipeline.forecast_model.cv_metrics:
        m = pipeline.forecast_model.cv_metrics.get("mean", {})
        fc_html += '<div class="kpi-row">'
        for k, label in [("rmse", "RMSE"), ("mae", "MAE"),
                         ("smape", "SMAPE"), ("mape", "MAPE")]:
            if k in m:
                val = f"{m[k]:,.1f}" + ("%" if k in ("smape", "mape") else "")
                fc_html += _kpi(f"CV {label}", val)
        fc_html += '</div>'
        fc_html += '<p class="muted">Walk-forward time-series cross-validation (3 folds).</p>'
    fc_html += _build_forecast_chart(canon, pipeline.forecast)
    fc_html += "<h3>Forecast preview (head)</h3>" + _table_html(pipeline.forecast)

    # --- Anomalies ---
    anom_html = _build_anomaly_chart(canon, pipeline.anomalies)
    anom_html += "<h3>Top anomalies</h3>" + _table_html(pipeline.anomalies)

    # --- Drivers ---
    drv_html = ""
    if pipeline.driver_report:
        dr = pipeline.driver_report
        drv_html += f'<div class="kpi-row">'
        drv_html += _kpi("R² (holdout)", f"{dr.r2:.3f}")
        drv_html += _kpi("RMSE (holdout)", f"{dr.rmse:,.1f}")
        drv_html += _kpi("Holdout rows", f"{dr.holdout_size:,}")
        drv_html += _kpi("Model", dr.model_name)
        drv_html += "</div>"
    drv_html += _build_drivers_chart(pipeline.driver_report)
    if pipeline.driver_report:
        drv_html += "<h3>Driver importance table</h3>" + _table_html(pipeline.driver_report.importance, 20)

    # --- Recommendations ---
    rec_html = "<h3>Reorder-point recommendations</h3>" + _table_html(pipeline.recommendations, 25)

    # --- Schema mapping summary ---
    schema_lines = ["<pre>" + html.escape(pipeline.mapping.report()) + "</pre>"]
    schema_html = "".join(schema_lines)

    body = (
        _section("Schema mapping (auto-detected)", schema_html, "schema") +
        _section("Exploratory data analysis", eda_html, "eda") +
        _section("Forecasting", fc_html, "forecast") +
        _section("Anomaly detection", anom_html, "anomalies") +
        _section("Sales drivers", drv_html, "drivers") +
        _section("Order recommendations", rec_html, "recommend")
    )

    nav = """
    <nav>
      <a href="#schema">Schema</a>
      <a href="#eda">EDA</a>
      <a href="#forecast">Forecast</a>
      <a href="#anomalies">Anomalies</a>
      <a href="#drivers">Drivers</a>
      <a href="#recommend">Recommendations</a>
    </nav>
    """

    full = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetailMind — {html.escape(dataset_name)} report</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
  <header>
    <h1>RetailMind — {html.escape(dataset_name)}</h1>
    <p>Universal retail analytics pipeline · same code, any dataset.</p>
  </header>
  {nav}
  <main>{body}</main>
  <footer>
    Generated by RetailMind v2 · Data Science Practicum 2 · Sai Teja Sunku
  </footer>
</body>
</html>"""
    return full


def save_report(pipeline, out_dir: str | Path = "docs",
                filename: str = "index.html",
                dataset_name: str = "Dataset") -> Path:
    """Render and save the report. Returns the written path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(render_report(pipeline, dataset_name=dataset_name), encoding="utf-8")
    return path
