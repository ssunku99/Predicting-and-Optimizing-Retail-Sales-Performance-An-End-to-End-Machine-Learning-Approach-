"""EDA — universal exploratory summary on canonical data.

All inputs are canonical (column names are 'date', 'entity_id', 'sales', ...).
Outputs are JSON-serializable dicts so they can be shown in the UI or pickled.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def overview(canon: pd.DataFrame) -> dict:
    """Top-line summary of the canonical dataframe."""
    n_entities = canon["entity_id"].nunique()
    date_min, date_max = canon["date"].min(), canon["date"].max()
    span_days = (date_max - date_min).days + 1
    total_sales = float(canon["sales"].sum())
    nonzero_rate = float((canon["sales"] > 0).mean())
    return {
        "rows": int(len(canon)),
        "entities": int(n_entities),
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "span_days": int(span_days),
        "total_sales": total_sales,
        "mean_daily_sales": float(canon["sales"].mean()),
        "median_daily_sales": float(canon["sales"].median()),
        "pct_nonzero_sales_days": nonzero_rate,
        "sales_p95": float(canon["sales"].quantile(0.95)),
        "sales_max": float(canon["sales"].max()),
    }


def missingness(canon: pd.DataFrame) -> pd.DataFrame:
    miss = canon.isna().mean().sort_values(ascending=False)
    return miss[miss > 0].rename("pct_missing").to_frame()


def entity_stats(canon: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Per-entity sales summary."""
    g = canon.groupby("entity_id")["sales"]
    summary = pd.DataFrame({
        "n_obs": g.size(),
        "total_sales": g.sum(),
        "mean_sales": g.mean(),
        "median_sales": g.median(),
        "std_sales": g.std(),
        "pct_zero_days": (canon.groupby("entity_id")["sales"].apply(lambda s: (s == 0).mean())),
    }).sort_values("total_sales", ascending=False)
    return summary.head(top)


def seasonality(canon: pd.DataFrame) -> dict:
    """Day-of-week, month-of-year, and week-of-month sales seasonality."""
    df = canon.copy()
    df["dow"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month_name()
    df["year"] = df["date"].dt.year
    return {
        "dow_mean_sales": df.groupby("dow")["sales"].mean().round(2).to_dict(),
        "month_mean_sales": df.groupby("month")["sales"].mean().round(2).to_dict(),
        "yearly_total": df.groupby("year")["sales"].sum().round(2).to_dict(),
    }


def promo_lift(canon: pd.DataFrame) -> Optional[dict]:
    """If a promo column exists, estimate naïve mean-sales lift on promo vs non-promo days."""
    if "promo" not in canon.columns:
        return None
    on = canon.loc[canon["promo"] > 0, "sales"]
    off = canon.loc[canon["promo"] == 0, "sales"]
    if on.empty or off.empty:
        return None
    lift = (on.mean() / off.mean() - 1) if off.mean() else np.nan
    return {
        "promo_days": int((canon["promo"] > 0).sum()),
        "nonpromo_days": int((canon["promo"] == 0).sum()),
        "mean_sales_on_promo": float(on.mean()),
        "mean_sales_off_promo": float(off.mean()),
        "naive_lift_pct": float(lift * 100) if not np.isnan(lift) else None,
    }


def aggregate_total(canon: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Sum across all entities to produce a single global time series."""
    total = (
        canon.groupby(pd.Grouper(key="date", freq=freq))["sales"]
        .sum()
        .reset_index()
    )
    return total


def full_report(canon: pd.DataFrame) -> dict:
    """One-shot EDA report combining everything above."""
    return {
        "overview": overview(canon),
        "missing": missingness(canon).to_dict()["pct_missing"] if not missingness(canon).empty else {},
        "top_entities": entity_stats(canon).reset_index().to_dict(orient="records"),
        "seasonality": seasonality(canon),
        "promo_lift": promo_lift(canon),
    }
