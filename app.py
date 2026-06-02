"""RetailMind — Universal Retail Analytics (Streamlit UI)

Upload ANY retail sales dataset (CSV / Excel) and get:
  • Automatic schema mapping to a canonical retail schema
  • EDA summary
  • Forecasts (LightGBM walk-forward CV + seasonal-naïve baseline)
  • Anomaly detection (IsolationForest + STL residual + IQR)
  • Sales-driver regression with feature importance
  • Inventory order recommendations (reorder-point with safety stock)
  • Natural-language Q&A over the results

Run with:
    streamlit run app.py

Design note: every heavy computation is cached in ``st.session_state.results``
so clicking the chat / download buttons doesn't trigger a full retrain.
"""

from __future__ import annotations

import io
import json
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from retailmind.ingest import load_file, load_dataset
from retailmind.mapper import SchemaMapper
from retailmind.schema import ColumnRole, CanonicalSchema
from retailmind.canonical import canonicalize
from retailmind import eda
from retailmind.forecast import train_lgbm, predict_future, seasonal_naive_forecast
from retailmind.anomaly import detect_all
from retailmind.regression import fit_driver_model
from retailmind.recommend import recommend_orders, RecommendationParams
from retailmind.assistant import ask, groq_status


st.set_page_config(page_title="RetailMind", page_icon="🛒", layout="wide")


# ============= helpers =============

@st.cache_data(show_spinner=False)
def cached_load(path: str, aux_paths: tuple[str, ...]) -> pd.DataFrame:
    if aux_paths:
        return load_dataset(path, auxiliary_paths=list(aux_paths))
    return load_file(path)


def write_uploaded(uploaded) -> str:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    return tmp.name


def render_schema_editor(raw: pd.DataFrame, schema: CanonicalSchema) -> CanonicalSchema:
    """Editable column→role table."""
    role_choices = [r.value for r in ColumnRole]
    rows = []
    for col in raw.columns:
        rows.append({"column": col, "role": schema.role_of(col).value,
                     "sample": str(raw[col].dropna().head(2).tolist())})
    edited = st.data_editor(
        pd.DataFrame(rows),
        column_config={
            "column": st.column_config.TextColumn("Column", disabled=True),
            "role": st.column_config.SelectboxColumn("Role", options=role_choices, required=True),
            "sample": st.column_config.TextColumn("Sample values", disabled=True),
        },
        hide_index=True, width="stretch",
        key="schema_editor",
    )
    new_map = {row["column"]: ColumnRole(row["role"]) for _, row in edited.iterrows()}
    return CanonicalSchema(mapping=new_map)


def render_schema_wizard(raw: pd.DataFrame, schema: CanonicalSchema) -> CanonicalSchema:
    """Wizard fallback shown only when validation fails: dropdowns for the
    missing roles, with smart per-column suggestions. The user literally
    cannot get stuck — every required role has a dropdown of every column.
    """
    st.markdown("##### Quick fix")
    st.caption("Pick the column that holds each required field. Suggestions are pre-selected.")
    cols = list(raw.columns)
    mapping = dict(schema.mapping)

    # Suggest DATE (datetime-like cols first)
    date_candidates = [c for c in cols if pd.api.types.is_datetime64_any_dtype(raw[c])]
    if not date_candidates:
        date_candidates = [c for c in cols if "date" in c.lower() or "time" in c.lower()]
    date_default = (mapping_inv := {v: k for k, v in mapping.items()}).get(ColumnRole.DATE) or (date_candidates[0] if date_candidates else cols[0])

    # Suggest SALES (numeric with high variance, positive, not an ID)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
    def _sales_score(c):
        s = raw[c].dropna()
        if s.empty: return -1
        if (s < 0).mean() > 0.05: return -1
        if "id" in c.lower() or "code" in c.lower(): return -1
        if s.nunique() <= 2: return -1
        return float(s.median())
    ranked_sales = sorted(numeric_cols, key=_sales_score, reverse=True)
    sales_default = mapping_inv.get(ColumnRole.SALES) or (ranked_sales[0] if ranked_sales else cols[0])

    # Suggest ENTITY_ID (low-cardinality object/int)
    ent_candidates = [c for c in cols if 1 < raw[c].nunique() <= max(50, int(np.sqrt(len(raw))))]
    ent_default = mapping_inv.get(ColumnRole.ENTITY_ID)

    c1, c2, c3 = st.columns(3)
    with c1:
        date_pick = st.selectbox("📅 Date column", cols,
                                  index=cols.index(date_default) if date_default in cols else 0,
                                  key="wiz_date")
    with c2:
        sales_pick = st.selectbox("💰 Sales column", cols,
                                   index=cols.index(sales_default) if sales_default in cols else 0,
                                   key="wiz_sales")
    with c3:
        ent_options = ["(none — single global series)"] + cols
        ent_default_idx = ent_options.index(ent_default) if ent_default in cols else 0
        ent_pick = st.selectbox("🏪 Store / outlet column (optional)", ent_options,
                                 index=ent_default_idx, key="wiz_entity")

    # Build the schema from the wizard picks, preserving other roles
    new_mapping = {c: ColumnRole.AUX for c in cols}
    new_mapping[date_pick] = ColumnRole.DATE
    new_mapping[sales_pick] = ColumnRole.SALES
    if ent_pick != "(none — single global series)":
        new_mapping[ent_pick] = ColumnRole.ENTITY_ID
    # Preserve other detected roles unless overwritten
    for c, r in mapping.items():
        if c not in (date_pick, sales_pick) and (ent_pick == "(none — single global series)" or c != ent_pick):
            if r not in (ColumnRole.IGNORE, ColumnRole.AUX, ColumnRole.DATE, ColumnRole.SALES, ColumnRole.ENTITY_ID):
                new_mapping[c] = r
    return CanonicalSchema(mapping=new_mapping)


# ============= session state init =============

if "results" not in st.session_state:
    st.session_state.results = {}
if "config_hash" not in st.session_state:
    st.session_state.config_hash = None


def _hash_config(main_path, aux_paths, freq, horizon, max_entities, schema_mapping):
    """Compute a stable hash of all inputs affecting the pipeline output."""
    import hashlib
    h = hashlib.md5()
    h.update(str(main_path).encode())
    h.update(str(sorted(aux_paths)).encode())
    h.update(freq.encode())
    h.update(str(horizon).encode())
    h.update(str(max_entities).encode())
    h.update(json.dumps({k: v.value for k, v in schema_mapping.items()},
                         sort_keys=True).encode())
    return h.hexdigest()


def run_full_pipeline(raw, schema, freq, horizon, max_entities, progress_cb=None) -> dict:
    """Run every stage with smart inference. Returns dict for the tabs."""
    from retailmind.pipeline import RetailPipeline
    from retailmind.mapper import MappingResult

    def step(msg, pct):
        if progress_cb: progress_cb(msg, pct)

    # Construct pipeline with user-edited schema; smart inference will layer on top.
    pipe = RetailPipeline(
        raw=raw,
        mapping=MappingResult(schema=schema),
        freq=freq, horizon=horizon,
        max_entities_for_full_forecast=max_entities,
    )

    step("Profiling data + smart inference…", 5)
    pipe.profile_and_infer()
    step("Canonicalizing…", 15)
    pipe.canonicalize_()
    step("EDA…", 25)
    pipe.eda_()
    step("Training LightGBM (walk-forward CV)…", 40)
    pipe.forecast_(sample_entities=max_entities, cv_folds=3)
    step("Detecting anomalies…", 70)
    pipe.anomalies_(sample_entities=max_entities)
    step("Fitting driver regression…", 85)
    pipe.drivers_(sample_entities=max_entities)
    step("Computing reorder recommendations…", 95)
    pipe.recommendations_()
    step("Done.", 100)

    canon = pipe.canonical
    # df_sub used by visualisation: top-N by sales
    if canon["entity_id"].nunique() > max_entities:
        top = canon.groupby("entity_id")["sales"].sum().nlargest(max_entities).index
        df_sub = canon[canon["entity_id"].isin(top)].copy()
    else:
        df_sub = canon

    return {
        "pipe": pipe,
        "canon": canon, "df_sub": df_sub,
        "eda_report": pipe.eda_report,
        "model": pipe.forecast_model, "forecast": pipe.forecast,
        "baseline": pipe.baseline_forecast,
        "anomalies": pipe.anomalies, "drivers": pipe.driver_report,
        "recommendations": pipe.recommendations,
        "horizon": horizon,
        "chosen_freq": pipe.chosen_freq,
        "decisions": pipe.decision_log.to_list(),
        "decisions_md": pipe.decision_log.render(),
        "stratified": pipe.stratified_summary(),
        "profile": pipe.profile_.to_dict() if pipe.profile_ else {},
    }


class _PipelineShim:
    """Duck-typed for retailmind.assistant.ask(). Reads from a results dict."""
    def __init__(self, results, mapping):
        self.canonical = results["canon"]
        self.eda_report = results["eda_report"]
        self.forecast_model = results["model"]
        self.forecast = results["forecast"]
        self.anomalies = results["anomalies"]
        self.driver_report = results["drivers"]
        self.recommendations = results["recommendations"]
        self.mapping = mapping
        self.horizon = results["horizon"]


# ============= sidebar =============

st.sidebar.title("RetailMind")
st.sidebar.caption("Built for messy retail CSVs.")

mode = st.sidebar.radio("Data source", ["Upload my own", "Use bundled samples"])

main_path: Optional[str] = None
aux_paths: list[str] = []

if mode == "Upload my own":
    up = st.sidebar.file_uploader("Upload main sales file (CSV / XLSX / Parquet)",
                                  type=["csv", "xlsx", "xls", "parquet", "tsv", "txt"])
    aux_ups = st.sidebar.file_uploader(
        "Optional: auxiliary lookup files",
        type=["csv", "xlsx"], accept_multiple_files=True,
    )
    if up:
        main_path = write_uploaded(up)
        aux_paths = [write_uploaded(a) for a in (aux_ups or [])]
else:
    sample = st.sidebar.selectbox("Pick a sample",
                                   ["Rossmann (train.csv + store.csv)"])
    main_path, aux_paths = "train.csv", ["store.csv"]

freq = st.sidebar.selectbox("Aggregation frequency",
                            ["auto", "D", "W", "MS"], index=0,
                            help="'auto' lets the profiler pick based on data density. "
                                 "D=daily, W=weekly, MS=month-start.")
horizon = st.sidebar.slider("Forecast horizon", 7, 90, 28, step=1)
max_entities = st.sidebar.slider("Max entities to forecast", 5, 50, 25,
                                  help="Cap on entities used in recursive forecast + recommendations")
service_level = st.sidebar.slider("Service level (recommendations)", 0.80, 0.99, 0.95, step=0.01)
lead_time = st.sidebar.slider("Lead time (days)", 1, 30, 7)

run_btn = st.sidebar.button("▶ Run pipeline", type="primary", disabled=main_path is None)


# ============= main =============

st.title("RetailMind")
st.write("Drop in a retail CSV — get a forecast, anomalies, drivers, and an order plan back.")

if main_path is None:
    st.info("Upload a file or pick a sample from the sidebar to get started.")
    st.stop()

# Always show schema mapping at the top
with st.spinner("Loading & inferring schema…"):
    raw = cached_load(main_path, tuple(aux_paths))
    auto_result = SchemaMapper().infer(raw)

st.subheader("Schema mapping")
st.caption(f"Loaded **{raw.shape[0]:,} rows × {raw.shape[1]} columns**. "
            f"Edit any role that's wrong — every downstream module reads only canonical roles.")

# Show wizard if validation fails — guaranteed-success entry point
val_errors = auto_result.schema.validate()
if val_errors:
    edited_schema = render_schema_wizard(raw, auto_result.schema)
    with st.expander("Advanced: show full column table"):
        edited_schema = render_schema_editor(raw, edited_schema)
else:
    with st.expander("Show / edit mapping", expanded=False):
        edited_schema = render_schema_editor(raw, auto_result.schema)

# Show any non-blocking warnings — use yellow for data-quality issues, blue for info
_QUALITY_KEYWORDS = (
    # data-layout / encoding problems
    "comma-separated", "multi-value", "cannot be used", "encoding",
    # temporal oddities
    "future", "pre-order", "test record",
    # financial data-quality
    "returns/refunds",
    # structural assumptions that the user must verify
    "each row is one unit",      # unit_price promoted to sales (qty unknown)
    "units rather than dollars", # quantity promoted to sales (no price column)
    "auto-selected",             # numeric best-guess fallback for sales column
    "last-resort",               # absolute fallback — almost certainly wrong
)
non_error_warnings = [w for w in auto_result.warnings
                       if not w.startswith("Required role missing")]
for w in non_error_warnings:
    if any(kw in w.lower() for kw in _QUALITY_KEYWORDS):
        st.warning(w)
    else:
        st.info(w)

errors_now = edited_schema.validate()
if errors_now:
    for e in errors_now:
        st.error(e)
    st.stop()

if not run_btn and not st.session_state.results:
    st.success("Schema looks good. Hit **Run pipeline** in the sidebar.")
    st.stop()


# ============= run pipeline (cached by config hash) =============

new_hash = _hash_config(main_path, aux_paths, freq, horizon, max_entities, edited_schema.mapping)
need_recompute = run_btn or new_hash != st.session_state.config_hash or not st.session_state.results

if need_recompute:
    progress = st.progress(0)
    status = st.empty()

    def cb(msg, pct):
        status.text(msg)
        progress.progress(pct)

    try:
        st.session_state.results = run_full_pipeline(raw, edited_schema, freq, horizon,
                                                       max_entities, progress_cb=cb)
        st.session_state.config_hash = new_hash
    except Exception as e:
        st.error(f"Pipeline failed: {type(e).__name__}: {e}")
        st.stop()
    finally:
        progress.empty(); status.empty()

R = st.session_state.results
canon = R["canon"]

# Smart-inference banner — surface what the pipeline auto-decided
if R.get("decisions"):
    with st.expander(f"**{len(R['decisions'])} smart-inference decisions** "
                     f"(chosen freq: `{R.get('chosen_freq', 'D')}`)", expanded=True):
        st.markdown(R["decisions_md"])

# Profile / quality
if R.get("profile"):
    prof = R["profile"]
    q = prof.get("quality_score", 0)
    qcolor = "🟢" if q >= 80 else ("🟡" if q >= 50 else "🔴")
    st.caption(f"{qcolor} Data-quality score: **{q}/100** · "
                f"{prof.get('pct_missing', 0):.1f}% missing · "
                f"{prof.get('duplicate_rows', 0):,} duplicate rows")

st.subheader("Canonical preview")
st.dataframe(canon.head(10), width="stretch")
st.caption(f"Canonical shape: **{canon.shape[0]:,} rows × {canon.shape[1]} cols** · "
           f"{canon['entity_id'].nunique()} entities · "
           f"{canon['date'].min().date()} → {canon['date'].max().date()}")


tab_eda, tab_fcst, tab_anom, tab_drv, tab_rec, tab_chat = st.tabs(
    ["EDA", "Forecast", "Anomalies", "Drivers", "Recommendations", "Ask"]
)

# ----- EDA -----
with tab_eda:
    report = R["eda_report"]
    o = report["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{o['rows']:,}")
    c2.metric("Entities", o["entities"])
    c3.metric("Date span", f"{o['span_days']} days")
    c4.metric("Total sales", f"{o['total_sales']:,.0f}")

    total = canon.groupby("date", as_index=False)["sales"].sum()
    st.plotly_chart(px.line(total, x="date", y="sales", title="Total sales over time"),
                     width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top 10 entities by total sales**")
        st.dataframe(eda.entity_stats(canon, top=10).reset_index(), width="stretch")
    with col_b:
        st.markdown("**Day-of-week seasonality**")
        dow = pd.Series(report["seasonality"]["dow_mean_sales"]).reset_index()
        dow.columns = ["dow", "mean_sales"]
        dow["dow"] = pd.Categorical(dow["dow"],
            categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ordered=True)
        st.plotly_chart(px.bar(dow.sort_values("dow"), x="dow", y="mean_sales"),
                         width="stretch")
    if report.get("promo_lift"):
        st.markdown("**Promotion impact (naïve mean lift)**")
        st.json(report["promo_lift"])

# ----- Forecast -----
with tab_fcst:
    model, fcst, baseline = R["model"], R["forecast"], R["baseline"]
    st.markdown("### Cross-validation metrics (walk-forward, 3 folds)")
    m = model.cv_metrics.get("mean", {}) if model.cv_metrics else {}
    if m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SMAPE", f"{m.get('smape', 0):.1f}%",
                  delta=f"-{m.get('baseline_smape', 0) - m.get('smape', 0):.1f}% vs baseline",
                  delta_color="inverse")
        c2.metric("RMSE", f"{m.get('rmse', 0):,.0f}",
                  delta=f"-{m.get('baseline_rmse', 0) - m.get('rmse', 0):,.0f} vs baseline",
                  delta_color="inverse")
        c3.metric("MAE", f"{m.get('mae', 0):,.0f}")
        lift = m.get("rmse_lift_pct", 0)
        c4.metric("RMSE improvement vs naive", f"{lift:+.1f}%",
                  help="How much better LightGBM is than a 7-day seasonal-naïve baseline on the same holdout.")
        if model.log_target:
            st.caption("ℹ️ Auto-applied log-transform (target was right-skewed).")
        st.caption("Baseline = seasonal-naïve (yhat[t] = sales[t-7]). "
                    "Lift % is the share of baseline RMSE that LightGBM eliminates.")
    with st.expander("Full per-fold details"):
        st.json(model.cv_metrics)

    # Per-entity stratified breakdown
    ss = R.get("stratified", {})
    if ss:
        st.markdown("### Per-entity breakdown")
        c1, c2, c3 = st.columns(3)
        c1.metric("% entities beating baseline",
                  f"{ss.get('pct_beating_baseline', 0):.1f}%",
                  help="Share of entities where LightGBM SMAPE < seasonal-naïve SMAPE on the last 14 obs.")
        c2.metric("Volume-weighted SMAPE",
                  f"{ss.get('volume_weighted_smape_lgbm', 0):.1f}%",
                  delta=f"{ss.get('volume_weighted_smape_baseline', 0) - ss.get('volume_weighted_smape_lgbm', 0):+.1f}% vs baseline",
                  delta_color="inverse")
        c3.metric("Entities evaluated", ss.get('n_entities_evaluated', 0))
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top 5 — best-predicted**")
            if ss.get("best_entities"):
                st.dataframe(pd.DataFrame(ss["best_entities"]), width="stretch", hide_index=True)
        with col_b:
            st.markdown("**Bottom 5 — hardest to predict**")
            if ss.get("worst_entities"):
                st.dataframe(pd.DataFrame(ss["worst_entities"]), width="stretch", hide_index=True)

    ents = sorted(fcst["entity_id"].unique())
    pick = st.selectbox("Entity", ents, key="fc_ent")
    df_sub = R["df_sub"]
    hist = df_sub[df_sub["entity_id"] == pick][["date", "sales"]].rename(columns={"sales": "y"})
    fc1 = fcst[fcst["entity_id"] == pick][["date", "yhat"]]
    bl1 = baseline[baseline["entity_id"] == pick][["date", "yhat"]].rename(columns={"yhat": "yhat_baseline"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["y"], name="history", mode="lines"))
    fig.add_trace(go.Scatter(x=fc1["date"], y=fc1["yhat"], name="LightGBM", mode="lines+markers"))
    if not bl1.empty:
        fig.add_trace(go.Scatter(x=bl1["date"], y=bl1["yhat_baseline"], name="Seasonal-naive",
                                  mode="lines", line=dict(dash="dash")))
    fig.update_layout(title=f"Forecast — entity {pick}", xaxis_title="date", yaxis_title="sales")
    st.plotly_chart(fig, width="stretch")
    st.download_button("⬇ Download forecast CSV", fcst.to_csv(index=False).encode(),
                       file_name="forecast.csv")

# ----- Anomalies -----
with tab_anom:
    anoms = R["anomalies"]
    if anoms is None or anoms.empty:
        st.success("No anomalies flagged.")
    else:
        st.markdown(f"**{len(anoms)} anomaly rows** (across methods, sorted by score)")
        st.dataframe(anoms.head(50), width="stretch")
        ent_pick = st.selectbox("Visualize anomalies for entity",
                                sorted(anoms["entity_id"].unique()), key="an_ent")
        df_sub = R["df_sub"]
        ent_series = df_sub[df_sub["entity_id"] == ent_pick]
        ent_anoms = anoms[anoms["entity_id"] == ent_pick]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ent_series["date"], y=ent_series["sales"], name="sales", mode="lines"))
        fig.add_trace(go.Scatter(x=ent_anoms["date"], y=ent_anoms["value"], name="anomaly",
                                  mode="markers", marker=dict(color="red", size=10)))
        st.plotly_chart(fig, width="stretch")
        st.download_button("⬇ Download anomalies CSV", anoms.to_csv(index=False).encode(),
                           file_name="anomalies.csv")

# ----- Drivers -----
with tab_drv:
    # AutoML toggle: lets the reviewer click and re-fit using FLAML's
    # search across LGBM / XGBoost / RandomForest / ExtraTrees.
    st.markdown("#### Model")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        use_automl = st.checkbox(
            "Use AutoML (FLAML searches across LightGBM, XGBoost, RandomForest, ExtraTrees)",
            value=False,
            help="Default is a single hand tuned LightGBM (fast). FLAML runs a 30 second "
                 "search across 4 model families with the same walk forward CV and reports "
                 "the winner. Often improves R² by 5 to 20 points.",
        )
    with col_b:
        budget = st.number_input("Search budget (sec)", min_value=15, max_value=180,
                                  value=30, step=15, disabled=not use_automl)

    if use_automl and st.button("Re-fit with AutoML", type="primary"):
        from retailmind.automl import fit_flaml_driver_model
        with st.spinner(f"FLAML searching across 4 model families for {budget}s..."):
            df_sub = R.get("df_sub", R["canon"])
            new_rep = fit_flaml_driver_model(df_sub, time_budget=int(budget), cv_folds=2)
            st.session_state.results["drivers"] = new_rep
            st.rerun()

    rep = R["drivers"]
    if "flaml" in rep.model_name.lower():
        st.success(f"AutoML winner: **{rep.model_name}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (walk-forward CV)", f"{rep.r2:+.3f}",
              help=f"Average across {rep.cv_folds} time-ordered folds. "
                   f"Honest evaluation — random k-fold would inflate this.")
    if rep.baseline_r2 is not None:
        c2.metric("Baseline R²", f"{rep.baseline_r2:+.3f}",
                  help="Seasonal-naïve baseline R² on the same holdout.")
        c3.metric("Lift over baseline", f"{rep.r2_lift_vs_baseline:+.3f}",
                  delta_color="normal",
                  help="LightGBM R² minus baseline R². Positive = model adds signal.")
    c4.metric("Holdout rows", f"{rep.holdout_size:,}")

    if rep.log_target_used:
        st.caption("ℹ️ Auto-applied log-transform (target was right-skewed). "
                    "Metrics are still reported on the original sales scale.")

    # Model-quality alert takes priority — show it at the top, suppress the
    # success banner (lift can still be positive when both model and baseline
    # are negative, which would produce a misleading ✅ alongside the ⚠️).
    _use_baseline = getattr(rep, "use_baseline", False)
    if _use_baseline and rep.why_baseline:
        st.error(
            f"⚠️ **Model quality alert:** {rep.why_baseline}"
        )
    elif rep.r2_lift_vs_baseline is not None and rep.r2_lift_vs_baseline > 0:
        # Only show the success banner when the model is genuinely good
        st.success(
            f"The model **beats the seasonal-naïve baseline by R² lift "
            f"{rep.r2_lift_vs_baseline:+.3f}**. Absolute R² depends on how "
            f"forecastable the dataset is (small / noisy / single-entity data "
            f"is genuinely hard); what matters is the **improvement over a naïve "
            f"baseline** — and the model is delivering that."
        )
    elif rep.r2 < 0.3:
        st.info(
            "Low R² is common for daily transactional data. Try the sidebar's "
            "frequency selector — switch to **W** (weekly) or **MS** (monthly). "
            "Smoothing usually doubles R² on noisy daily series."
        )

    st.markdown("**Top features (leakage-safe)**")
    imp = rep.importance.copy()
    imp["importance_norm"] = imp["importance"] / imp["importance"].max()
    fig = px.bar(imp.head(20), x="importance_norm", y="feature",
                 color="direction_label", orientation="h",
                 title="Feature importance (gain) + direction")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(rep.importance, width="stretch")

# ----- Recommendations -----
with tab_rec:
    from retailmind.recommend import summary_metrics

    st.markdown("### What to order")
    st.caption(
        "Based on the forecast, here's what to order, when, and why — in plain English. "
        "Tweak the sliders if your delivery schedule is different."
    )

    # --- Controls (per-tab refinement on top of the sidebar) ---
    col_a, col_b = st.columns([1, 2])
    with col_a:
        review_period = st.slider("Check stock every (days)", 1, 30, 7,
                                   help="How often you re-check shelves and place orders.")
    with col_b:
        st.caption(
            f"Using **{lead_time}-day delivery** and aiming to avoid stock-outs "
            f"**{service_level:.0%}** of the time. (Adjust both in the left sidebar.)"
        )

    # --- On-hand inventory input (tucked behind expander — most users skip it) ---
    with st.expander("Already have stock on the shelves? Enter it here.", expanded=False):
        st.caption(
            "Tell us what you have for each item. Anything left at zero is treated as "
            "empty shelves (worst case)."
        )
        forecast_entities = sorted(R["forecast"]["entity_id"].unique())
        oh_df = pd.DataFrame({"entity_id": [str(e) for e in forecast_entities], "on_hand": 0.0})
        edited_oh = st.data_editor(
            oh_df,
            column_config={
                "entity_id": st.column_config.TextColumn("Item", disabled=True),
                "on_hand": st.column_config.NumberColumn("Units in stock",
                                                          min_value=0, step=10, format="%.1f"),
            },
            hide_index=True, width="stretch", key="on_hand_editor",
        )

    # --- Compute recommendations with the inventory model ---
    params = RecommendationParams(
        lead_time_days=lead_time,
        service_level=service_level,
        review_period_days=review_period,
    )
    recs = recommend_orders(R["forecast"], on_hand=edited_oh, params=params)
    sm = summary_metrics(recs)

    # --- Headline banner — one sentence the user can act on immediately ---
    if sm:
        if sm["n_urgent"] > 0:
            st.error(
                f"**{sm['n_urgent']} item(s) need ordering right now.**  "
                f"Total to order across everything: **{int(sm['total_order_qty']):,} units**."
            )
        elif sm["n_reorder_soon"] > 0:
            st.warning(
                f"**{sm['n_reorder_soon']} item(s) will run low soon.** "
                f"Plan an order in the next few days."
            )
        else:
            st.success("**All items are well-stocked.** Nothing to order right now.")

        # 3 status cards
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Order now", sm["n_urgent"],
                  help="On-hand stock will run out before the next delivery arrives.")
        c2.metric("🟡 Order soon", sm["n_reorder_soon"],
                  help="Stock lasts past the next delivery but not past the next check.")
        c3.metric("🟢 Well-stocked", sm["n_stocked"],
                  help="Plenty to cover the whole cycle.")

    # --- Per-item narrative cards — the most important new piece ---
    if not recs.empty:
        urgent_recs = recs[recs["urgency"] == "🔴 urgent"]
        soon_recs   = recs[recs["urgency"] == "🟡 reorder soon"]
        ok_recs     = recs[recs["urgency"] == "🟢 stocked"]

        if not urgent_recs.empty:
            st.markdown("#### Order these now")
            for _, row in urgent_recs.iterrows():
                qty   = int(round(row["recommended_order_qty"]))
                hand  = int(round(row["on_hand"]))
                dmd   = row["mean_daily_demand"]
                cover = row["days_of_cover"]
                cover_phrase = ("you'd run out before the next delivery arrives"
                                 if cover < lead_time
                                 else f"you'd run out in about {cover:.0f} days")
                st.markdown(
                    f"**{row['entity_id']}** — order **{qty:,} units** within the next "
                    f"**{review_period} days**.  \n"
                    f"<span style='color:#64748B'>You have **{hand} units** on hand. "
                    f"At a forecasted demand of **{dmd:.1f} units/day**, {cover_phrase} "
                    f"(delivery takes {lead_time} days).</span>",
                    unsafe_allow_html=True,
                )
                st.divider()

        if not soon_recs.empty:
            st.markdown("#### Plan to order in the next few days")
            for _, row in soon_recs.iterrows():
                qty  = int(round(row["recommended_order_qty"]))
                hand = int(round(row["on_hand"]))
                dmd  = row["mean_daily_demand"]
                st.markdown(
                    f"**{row['entity_id']}** — order **{qty:,} units** within "
                    f"**{review_period} days**.  \n"
                    f"<span style='color:#64748B'>You have {hand} on hand · "
                    f"demand ≈ {dmd:.1f} units/day.</span>",
                    unsafe_allow_html=True,
                )
                st.divider()

        if not urgent_recs.empty or not soon_recs.empty:
            if not ok_recs.empty:
                ok_names = ", ".join(str(e) for e in ok_recs["entity_id"].head(5))
                more = f" and {len(ok_recs) - 5} more" if len(ok_recs) > 5 else ""
                st.caption(f"✓ Well-stocked, no action: {ok_names}{more}.")
        elif not ok_recs.empty:
            st.caption("✓ Every item is well-stocked. Check back after your next review.")

    # --- Full details table (collapsed by default) ---
    if not recs.empty:
        with st.expander("Show the full numbers for every item", expanded=False):
            st.caption(
                "Power-user view. Same data, no narrative. "
                "Useful if you want to paste it into a spreadsheet."
            )
            rec_display = recs.rename(columns={
                "entity_id":              "Item",
                "urgency":                "Status",
                "on_hand":                "In stock now",
                "days_of_cover":          "Days of stock left",
                "mean_daily_demand":      "Forecasted demand / day",
                "demand_std":             "Demand variability",
                "lead_time":              "Lead time (days)",
                "review_period":          "Review period (days)",
                "service_level":          "Service level",
                "reorder_point":          "Reorder when stock hits",
                "target_stock_level":     "Order up to this level",
                "recommended_order_qty":  "Order this many",
            })
            st.dataframe(rec_display, width="stretch")
            st.download_button("Download as CSV", recs.to_csv(index=False).encode(),
                                file_name="recommendations.csv")

    # --- Per-item chart (collapsed by default for cleaner first view) ---
    if not recs.empty:
        with st.expander("See it on a chart", expanded=False):
            st.caption("Pick an item to see how the daily forecast lines up with stock levels.")
            pick = st.selectbox("Item",
                                 recs["entity_id"].tolist(), key="rec_ent")
            f = R["forecast"][R["forecast"]["entity_id"].astype(str) == pick]
            r_row = recs[recs["entity_id"] == pick].iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=f["date"], y=f["yhat"], name="Daily forecast",
                                      mode="lines+markers", line=dict(color="#1f77b4")))
            fig.add_hline(y=r_row["reorder_point"], line_dash="dot", line_color="orange",
                           annotation_text="Reorder when stock drops here",
                           annotation_position="right")
            fig.add_hline(y=r_row["target_stock_level"], line_dash="dash", line_color="green",
                           annotation_text="Order up to this level",
                           annotation_position="right")
            fig.add_hline(y=r_row["on_hand"], line_dash="dashdot", line_color="blue",
                           annotation_text="What you have now",
                           annotation_position="right")
            fig.update_layout(title=f"{pick} — forecast vs stock targets",
                               xaxis_title="Date", yaxis_title="Units",
                               height=380, margin=dict(l=40, r=80, t=50, b=40))
            st.plotly_chart(fig, width="stretch")

# ----- Chat / Q&A assistant -----
with tab_chat:
    st.markdown("### Ask RetailMind")

    # Server-side Groq key from Streamlit Secrets — never exposed to the browser.
    # If present, visitors get richer answers without needing their own key, capped
    # at a per-session quota so a single tab can't drain the demo budget.
    try:
        _server_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    except Exception:
        _server_key = None
    SHARED_KEY_MODE = bool(_server_key)
    SHARED_MAX_MSGS = 15

    if SHARED_KEY_MODE:
        _sent = st.session_state.get("chat_msg_count", 0)
        st.caption(f"Powered by Groq · {max(0, SHARED_MAX_MSGS - _sent)}/{SHARED_MAX_MSGS} "
                   "questions left in this session.")
    else:
        st.caption("Natural-language Q&A over the pipeline outputs. Works **offline** by default "
                   "(no API key). Paste a Groq key for richer answers.")

    shim = _PipelineShim(R, mapping=auto_result)

    st.markdown("#### Ask anything about your data")
    st.caption("Type your own question in the box below, **or** click an example to start.")

    examples = [
        "Give me an overview of the dataset",
        "How accurate is the forecast?",
        "What drives sales the most?",
        "Show me the top anomalies",
        "How much should I order?",
        "What's the day-of-week pattern?",
        "Which 5 stores should I stock more for?",
        "Write a 3-sentence summary for the store owner",
        "Compare the model to the baseline",
    ]
    st.markdown("**Quick-start examples** (click any → it fills the box):")
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}", width="stretch"):
            st.session_state["chat_q"] = ex

    q = st.text_area(
        "Your question",
        value=st.session_state.get("chat_q", ""),
        placeholder="e.g. Which days of the week have the highest sales? What's the best-performing entity? How should I prepare for next month?",
        height=80,
        key="chat_input",
        help="Free-form — ask anything about EDA, forecast, anomalies, drivers, or recommendations.",
    )
    if SHARED_KEY_MODE:
        # Shared server-side key — no UI for mode / paste; fixed to Groq + cheap model.
        mode_pick = "groq"
        session_key = _server_key
        groq_model_to_use = "llama-3.1-8b-instant"
    else:
        mode_pick = st.radio("Mode", ["auto", "offline", "groq"], horizontal=True, index=0,
                             help="auto = groq if a key is available, else offline. "
                                  "groq = force richer LLM answers (needs key below). "
                                  "offline = pure pipeline data, no LLM.")

        # In-app key field (session-only, never written to disk)
        if mode_pick in ("auto", "groq"):
            with st.expander("Groq API key (paste here for richer answers — never stored)",
                              expanded=(mode_pick == "groq")):
                pasted = st.text_input(
                    "Paste your Groq API key (starts with `gsk_`)",
                    value=st.session_state.get("groq_key", ""),
                    type="password",
                    key="groq_key_input",
                    help="Get a free key in 30 seconds at console.groq.com/keys "
                         "— no credit card required.",
                )
                if pasted:
                    st.session_state["groq_key"] = pasted
                if st.session_state.get("groq_key"):
                    st.caption("Key set for this session. Click Ask below.")
                st.markdown("**How to get one (free, 30 seconds):**")
                st.markdown("1. Open [console.groq.com/keys](https://console.groq.com/keys) "
                             "and sign in with Google/GitHub.\n"
                             "2. Click **Create API Key**, copy it.\n"
                             "3. Paste it above.")

        session_key = st.session_state.get("groq_key") or None
        groq_model_to_use = "llama-3.3-70b-versatile"
        gready, gstatus = groq_status(session_key)
        if mode_pick == "groq":
            if gready:
                st.success(gstatus)
            else:
                st.warning(f"{gstatus}\n\nClick Ask anyway and you'll get the offline answer "
                            "with a clear note about why Groq wasn't used.")
        elif mode_pick == "auto":
            if gready:
                st.caption("auto-mode will use Groq (key detected).")
            else:
                st.caption("auto-mode will use offline (no Groq key set).")

    if st.button("Ask", type="primary", key="ask_btn"):
        # Per-session quota check (only when using the shared server-side key)
        if SHARED_KEY_MODE and st.session_state.get("chat_msg_count", 0) >= SHARED_MAX_MSGS:
            st.warning(
                f"You've used all {SHARED_MAX_MSGS} questions in this session. "
                "Refresh the page to start a new one."
            )
        else:
            try:
                with st.spinner("Thinking…"):
                    answer = ask(shim, q, mode=mode_pick, api_key=session_key,
                                 groq_model=groq_model_to_use)
                st.markdown("---")
                st.markdown(answer)
                if SHARED_KEY_MODE:
                    st.session_state["chat_msg_count"] = (
                        st.session_state.get("chat_msg_count", 0) + 1
                    )
            except Exception as e:
                st.error(f"Assistant error — {type(e).__name__}: {e}")
                with st.expander("Debug info"):
                    st.write({
                        "has_canonical": shim.canonical is not None,
                        "has_eda_report": shim.eda_report is not None,
                        "has_forecast_model": shim.forecast_model is not None,
                        "has_forecast": shim.forecast is not None,
                        "has_anomalies": shim.anomalies is not None,
                        "has_driver_report": shim.driver_report is not None,
                        "has_recommendations": shim.recommendations is not None,
                    })

st.sidebar.success("Pipeline complete.")
st.sidebar.caption("⚡ Results cached — clicking buttons (incl. Ask) won't retrain.")
