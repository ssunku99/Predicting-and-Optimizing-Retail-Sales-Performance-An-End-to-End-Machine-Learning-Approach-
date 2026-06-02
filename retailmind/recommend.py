"""Order recommendations — turn forecasts into reorder decisions.

Uses the **periodic-review (R, S) inventory model**, which matches how
small retailers actually operate: every `review_period_days`, check stock
and order enough to cover demand until the next review + safety margin.

For each entity:
    L  = lead_time_days        (days between placing order and receipt)
    R  = review_period_days    (how often you re-evaluate stock)
    z  = z-score for service_level (e.g. 95% → 1.645)

    mean_demand = mean of forecast over the next (L + R) days
    std_demand  = std  of forecast over the next (L + R) days

    target_stock_S    = mean_demand * (L + R) + z * std_demand * sqrt(L + R)
    reorder_qty       = max(0, target_stock_S - on_hand)

    reorder_point_ROP = mean_demand * L + z * std_demand * sqrt(L)
                       (informational — order when stock drops below this)
    days_of_cover     = on_hand / mean_demand_per_day
    urgency           = "urgent" if days_of_cover < L else "ok"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RecommendationParams:
    lead_time_days: int = 7
    service_level: float = 0.95
    review_period_days: int = 7
    default_on_hand: float = 0.0   # used if no on_hand provided per-entity


def recommend_orders(
    forecast: pd.DataFrame,
    on_hand: Optional[pd.DataFrame] = None,
    params: RecommendationParams = RecommendationParams(),
) -> pd.DataFrame:
    """Convert per-entity forecast into actionable reorder decisions.

    Parameters
    ----------
    forecast : DataFrame with columns ['entity_id', 'date', 'yhat'], one row
               per future period per entity.
    on_hand  : optional ['entity_id', 'on_hand']. Missing entries default to
               `params.default_on_hand`.
    params   : RecommendationParams.

    Returns DataFrame sorted by urgency then volume.
    """
    z = float(norm.ppf(params.service_level))
    L = int(params.lead_time_days)
    R = int(params.review_period_days)
    window = L + R

    rows = []
    for ent, g in forecast.sort_values("date").groupby("entity_id"):
        # Use the next (L + R) days of forecast for demand statistics — not the
        # full horizon. This matches the inventory cycle, not the user's horizon.
        h = g["yhat"].values[:window] if len(g) >= window else g["yhat"].values
        if len(h) == 0:
            continue
        mean_d = float(np.mean(h))                 # per-day mean
        std_d = float(np.std(h, ddof=1)) if len(h) > 1 else 0.0

        # Periodic-review target stock (the S in (R, S) model)
        target_S = mean_d * window + z * std_d * np.sqrt(window)
        # Continuous reorder point (informational)
        rop = mean_d * L + z * std_d * np.sqrt(L)

        # On-hand lookup
        oh = float(params.default_on_hand)
        if on_hand is not None and not on_hand.empty:
            row = on_hand[on_hand["entity_id"].astype(str) == str(ent)]
            if not row.empty:
                oh = float(row["on_hand"].iloc[0])

        order_qty = max(0.0, target_S - oh)
        days_cover = oh / mean_d if mean_d > 0 else float("inf")
        urgency = ("🟢 stocked" if days_cover >= window
                   else "🟡 reorder soon" if days_cover >= L
                   else "🔴 urgent")

        rows.append({
            "entity_id": str(ent),
            "urgency": urgency,
            "on_hand": round(oh, 1),
            "days_of_cover": round(days_cover, 1) if np.isfinite(days_cover) else None,
            "mean_daily_demand": round(mean_d, 1),
            "demand_std": round(std_d, 1),
            "lead_time": L,
            "review_period": R,
            "service_level": params.service_level,
            "reorder_point": round(rop, 1),
            "target_stock_level": round(target_S, 1),
            "recommended_order_qty": round(order_qty, 1),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    urgency_rank = {"🔴 urgent": 0, "🟡 reorder soon": 1, "🟢 stocked": 2}
    df["_rank"] = df["urgency"].map(urgency_rank)
    df = df.sort_values(["_rank", "recommended_order_qty"], ascending=[True, False]).drop(columns="_rank").reset_index(drop=True)
    return df


def summary_metrics(recs: pd.DataFrame) -> dict:
    """Top-level summary for the UI."""
    if recs is None or recs.empty:
        return {}
    return {
        "total_entities": int(len(recs)),
        "n_urgent": int((recs["urgency"] == "🔴 urgent").sum()),
        "n_reorder_soon": int((recs["urgency"] == "🟡 reorder soon").sum()),
        "n_stocked": int((recs["urgency"] == "🟢 stocked").sum()),
        "total_order_qty": float(recs["recommended_order_qty"].sum()),
        "total_target_stock": float(recs["target_stock_level"].sum()),
        "total_on_hand": float(recs["on_hand"].sum()),
    }
