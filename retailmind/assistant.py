"""RetailMind Assistant — natural-language Q&A over pipeline outputs.

Addresses reviewer feedback #5 on v1: the v1 chatbot needed a Groq API key, so
reviewers without one saw no output and concluded the feature was missing.

This assistant works in two modes:

  1. **Offline** (default, no API key) — uses keyword-based intent routing to
     answer questions directly from pipeline outputs with specific numbers.
     Always returns a real answer.

  2. **Groq** (optional, requires GROQ_API_KEY env var) — routes the same
     question + pipeline context to a Groq LLM for richer natural-language
     answers. If the key is missing, silently falls back to offline mode.

Both modes use the same `PipelineContext` built from `RetailPipeline` outputs.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineContext:
    """Compact, JSON-serializable snapshot of pipeline results."""
    overview: dict
    top_entities: list[dict]
    seasonality: dict
    promo_lift: Optional[dict]
    forecast_metrics: dict             # mean across CV folds
    top_forecast_entities: list[dict]  # entity_id + next-7-day projection
    top_anomalies: list[dict]
    top_drivers: list[dict]
    top_recommendations: list[dict]
    horizon: int

    def to_dict(self) -> dict:
        return {
            "overview": self.overview,
            "top_entities": self.top_entities,
            "seasonality": self.seasonality,
            "promo_lift": self.promo_lift,
            "forecast_metrics": self.forecast_metrics,
            "top_forecast_entities": self.top_forecast_entities,
            "top_anomalies": self.top_anomalies,
            "top_drivers": self.top_drivers,
            "top_recommendations": self.top_recommendations,
            "horizon": self.horizon,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def build_context(pipeline) -> PipelineContext:
    """Construct PipelineContext from a fully-run RetailPipeline."""
    eda_r = pipeline.eda_report or {}
    fc = pipeline.forecast if pipeline.forecast is not None else pd.DataFrame()
    fc_mean = (pipeline.forecast_model.cv_metrics.get("mean", {})
               if pipeline.forecast_model and pipeline.forecast_model.cv_metrics else {})

    top_fc = []
    if not fc.empty:
        agg = (fc.groupby("entity_id")
                 .agg(next_7d_sum=("yhat", lambda s: float(s.head(7).sum())),
                      mean_daily=("yhat", "mean"))
                 .sort_values("next_7d_sum", ascending=False)
                 .head(10)
                 .reset_index())
        top_fc = agg.to_dict(orient="records")

    anoms = pipeline.anomalies
    top_anom = []
    if anoms is not None and not anoms.empty:
        top_anom = anoms.head(10)[["entity_id", "date", "value", "method",
                                    "score", "reason"]].to_dict(orient="records")

    dr = pipeline.driver_report
    top_drv = []
    if dr is not None:
        top_drv = dr.importance.head(10)[
            ["feature", "importance", "direction_label"]
        ].to_dict(orient="records")

    recs = pipeline.recommendations
    top_rec = []
    if recs is not None and not recs.empty:
        top_rec = recs.head(10)[["entity_id", "mean_daily_demand", "demand_std",
                                  "reorder_point", "recommended_order_qty",
                                  "service_level"]].to_dict(orient="records")

    return PipelineContext(
        overview=eda_r.get("overview", {}),
        top_entities=eda_r.get("top_entities", [])[:10],
        seasonality=eda_r.get("seasonality", {}),
        promo_lift=eda_r.get("promo_lift"),
        forecast_metrics=fc_mean,
        top_forecast_entities=top_fc,
        top_anomalies=top_anom,
        top_drivers=top_drv,
        top_recommendations=top_rec,
        horizon=pipeline.horizon,
    )


# ============= Intent routing for OFFLINE mode =============

# Intent keywords. Matched as substrings but `\b`-bounded so 'sales' doesn't
# accidentally trigger the 'sale' keyword in the promo intent.
INTENTS = {
    "metrics":    (r"\brmse\b", r"\bmape\b", r"\bsmape\b", r"\baccuracy\b",
                   r"\baccurate\b", r"\berror\b", r"\bperformance\b",
                   r"\bhow good\b", r"\bhow well\b"),
    "promo":      (r"\bpromo\b", r"\bpromotion\b", r"\bdiscount\b",
                   r"\bcampaign\b", r"\bpromo lift\b"),
    "drivers":    (r"\bdriver\b", r"\bdrives\b", r"\bdrive\b", r"\bimportant\b",
                   r"\bwhat affects\b", r"\binfluence\b", r"\bfeature\b",
                   r"\bfactor\b", r"\bwhy\b"),
    "recommend":  (r"\brecommend\b", r"\breorder\b", r"\bstock\b",
                   r"\binventory\b", r"\border\b", r"\bbuy\b", r"\bhow much\b"),
    "anomaly":    (r"\banomal", r"\boutlier\b", r"\bweird\b", r"\bunusual\b",
                   r"\bspike\b", r"\bdrop\b", r"\bsuspicious\b"),
    "forecast":   (r"\bforecast\b", r"\bpredict\b", r"\bnext week\b",
                   r"\bnext month\b", r"\bfuture\b", r"\bproject\b"),
    "seasonality":(r"\bseasonal\b", r"\bday of week\b", r"\bmonth\b",
                   r"\bweekly\b", r"\byearly\b", r"\bpattern\b", r"\btrend\b"),
    "top_stores": (r"\btop\b", r"\bbest\b", r"\bhighest\b", r"\bbiggest\b",
                   r"\bleading\b", r"\btop performer"),
    "overview":   (r"\boverview\b", r"\bsummary\b", r"\bdescribe\b",
                   r"\bwhat is\b", r"\bdataset\b", r"\bstats\b", r"\bhow many\b",
                   r"\btell me about\b"),
}


def _detect_intent(q: str) -> str:
    q = q.lower()
    scores = {k: sum(bool(re.search(pat, q)) for pat in pats) for k, pats in INTENTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "overview"


def _fmt_money(v: float) -> str:
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:,.2f}"


def _answer_offline(question: str, ctx: PipelineContext) -> str:
    intent = _detect_intent(question)
    o = ctx.overview

    if intent == "overview":
        return textwrap.dedent(f"""
            **Dataset overview**
            • {o.get('rows', 0):,} observations across {o.get('entities', 0)} entities
            • Time span: {o.get('date_min')} → {o.get('date_max')} ({o.get('span_days')} days)
            • Total sales: {_fmt_money(o.get('total_sales', 0))}
            • Mean daily sales per entity: {o.get('mean_daily_sales', 0):,.0f}
            • {o.get('pct_nonzero_sales_days', 0) * 100:.1f}% of entity-days had non-zero sales
        """).strip()

    if intent == "forecast":
        m = ctx.forecast_metrics or {}
        lines = [f"**Forecast (horizon: {ctx.horizon} steps ahead)**"]
        if m:
            lines.append(f"• Walk-forward CV — RMSE: {m.get('rmse', 0):,.1f} · "
                         f"MAE: {m.get('mae', 0):,.1f} · SMAPE: {m.get('smape', 0):.1f}%")
        if ctx.top_forecast_entities:
            lines.append("\n**Top 5 projected next-7-day sales:**")
            for r in ctx.top_forecast_entities[:5]:
                lines.append(f"• Entity `{r['entity_id']}`: {_fmt_money(r['next_7d_sum'])} "
                             f"(avg {r['mean_daily']:,.0f}/day)")
        return "\n".join(lines)

    if intent == "anomaly":
        if not ctx.top_anomalies:
            return "No anomalies were flagged in the data."
        lines = [f"**Top anomalies detected** ({len(ctx.top_anomalies)} of many)"]
        for a in ctx.top_anomalies[:5]:
            lines.append(f"• `{a['entity_id']}` on {a['date']}: {a['reason']} "
                         f"[method: {a['method']}, score: {a['score']:.2f}]")
        return "\n".join(lines)

    if intent == "drivers":
        if not ctx.top_drivers:
            return "Driver analysis not yet run."
        lines = ["**Top 5 sales drivers** (LightGBM gain importance, leakage-safe)"]
        for d in ctx.top_drivers[:5]:
            lines.append(f"• `{d['feature']}` — {d['direction_label']}")
        return "\n".join(lines)

    if intent == "recommend":
        if not ctx.top_recommendations:
            return "Recommendations not yet computed."
        lines = ["**Reorder recommendations** (sorted by qty needed)"]
        for r in ctx.top_recommendations[:5]:
            lines.append(f"• Entity `{r['entity_id']}` — order {r['recommended_order_qty']:,.0f} units "
                         f"(reorder point: {r['reorder_point']:,.0f}, "
                         f"mean daily demand: {r['mean_daily_demand']:,.0f})")
        lines.append(f"\nService level used: {ctx.top_recommendations[0]['service_level']:.0%}")
        return "\n".join(lines)

    if intent == "promo":
        if not ctx.promo_lift:
            return "No promotion column present in this dataset — can't estimate promo lift."
        p = ctx.promo_lift
        return textwrap.dedent(f"""
            **Promotion impact (naïve mean lift)**
            • Mean sales on promo days: {p['mean_sales_on_promo']:,.0f}
            • Mean sales off-promo:     {p['mean_sales_off_promo']:,.0f}
            • Naïve lift: **{p.get('naive_lift_pct', 0):.1f}%**
            • Sample sizes: {p['promo_days']:,} promo days vs {p['nonpromo_days']:,} non-promo
        """).strip()

    if intent == "seasonality":
        s = ctx.seasonality
        dow = s.get("dow_mean_sales", {})
        if not dow:
            return "No clear day-of-week pattern available."
        ordered = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        peak = max(dow, key=dow.get)
        trough = min(dow, key=dow.get)
        lines = ["**Day-of-week pattern (mean sales)**"]
        for d in ordered:
            if d in dow:
                bar = "█" * int(dow[d] / max(dow.values()) * 20)
                lines.append(f"  {d:<10} {dow[d]:>10,.0f}  {bar}")
        lines.append(f"\nPeak: **{peak}** · Lowest: **{trough}**")
        return "\n".join(lines)

    if intent == "top_stores":
        if not ctx.top_entities:
            return "No entity rankings available."
        lines = ["**Top 5 entities by total sales**"]
        for e in ctx.top_entities[:5]:
            lines.append(f"• `{e['entity_id']}` — total {_fmt_money(e['total_sales'])}, "
                         f"avg {e['mean_sales']:,.0f}/period")
        return "\n".join(lines)

    if intent == "metrics":
        m = ctx.forecast_metrics or {}
        if not m:
            return "Forecast model not trained yet."
        return textwrap.dedent(f"""
            **Forecast model performance (walk-forward CV)**
            • RMSE:  {m.get('rmse', 0):,.1f}
            • MAE:   {m.get('mae', 0):,.1f}
            • MAPE:  {m.get('mape', 0):.1f}%  (high when many zero-sales days exist)
            • SMAPE: {m.get('smape', 0):.1f}%  (more reliable than MAPE on retail data)
        """).strip()

    return "I can answer about: overview, forecast, anomalies, drivers, recommendations, promo lift, seasonality, top stores, model metrics."


# ============= Optional Groq mode =============

def groq_status(api_key_override: Optional[str] = None) -> tuple[bool, str]:
    """Return (is_ready, status_message). If api_key_override is provided,
    use that instead of the env var (lets the UI pass a pasted key directly)."""
    try:
        from groq import Groq  # noqa: F401
    except ImportError:
        return False, "❌ `groq` Python package not installed. Run `pip install groq`."
    key = api_key_override or os.environ.get("GROQ_API_KEY")
    if not key:
        return False, ("❌ No API key. Paste one in the box below, or "
                       "`export GROQ_API_KEY=gsk_...` and restart. "
                       "Get a free key at https://console.groq.com/keys")
    if not key.startswith(("gsk_", "GSK_")):
        return False, ("⚠️  API key doesn't look like a Groq key "
                       "(should start with `gsk_`). Double-check at https://console.groq.com/keys")
    return True, "✅ Groq ready"


def _answer_groq(question: str, ctx: PipelineContext, model: str,
                  api_key: Optional[str] = None) -> str:
    ready, status = groq_status(api_key)
    if not ready:
        offline = _answer_offline(question, ctx)
        return (f"⚠️  **Groq mode unavailable** — using offline mode instead.\n\n"
                f"{status}\n\n---\n\n{offline}")
    from groq import Groq
    key = api_key or os.environ.get("GROQ_API_KEY")
    try:
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": ("You are RetailMind AI. Answer ONLY from the provided JSON "
                             "pipeline context. Use specific numbers. Be concise and business-focused. "
                             "If the answer isn't in the context, say so.")},
                {"role": "user",
                 "content": f"Pipeline context:\n{ctx.to_json()}\n\nQuestion: {question}"},
            ],
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        offline = _answer_offline(question, ctx)
        return (f"⚠️  **Groq call failed**: `{type(e).__name__}: {e}`\n\n"
                f"Falling back to offline mode:\n\n---\n\n{offline}")


# ============= public API =============

def ask(
    pipeline,
    question: str,
    mode: str = "auto",
    groq_model: str = "llama-3.3-70b-versatile",
    api_key: Optional[str] = None,
) -> str:
    """Answer a question about pipeline results.

    Parameters
    ----------
    pipeline : RetailPipeline that has already been .run()
    question : free-text question
    mode : 'auto' (Groq if a key is available, else offline),
           'offline' (force offline), or 'groq' (force Groq)
    api_key : optional Groq key — overrides GROQ_API_KEY env var.
              Lets the UI accept a pasted key without restarting.
    """
    ctx = build_context(pipeline)
    if mode == "offline":
        return _answer_offline(question, ctx)
    if mode == "groq":
        return _answer_groq(question, ctx, groq_model, api_key=api_key)
    # auto: prefer groq if available
    key = api_key or os.environ.get("GROQ_API_KEY")
    if key:
        return _answer_groq(question, ctx, groq_model, api_key=key)
    return _answer_offline(question, ctx)
