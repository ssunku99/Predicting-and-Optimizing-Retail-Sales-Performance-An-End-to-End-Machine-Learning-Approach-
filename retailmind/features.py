"""Feature engineering on canonical data.

Builds the matrix that ML models consume:
  - Calendar features (year, month, dow, week, is_weekend, is_month_end)
  - Lag features (sales lag-1, lag-7, lag-14, lag-28)
  - Rolling-window stats (rolling mean / std over 7, 14, 28 days)
  - Trend / time-since-start
  - One-hot or target-encoded categoricals
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 7, 14, 28)
DEFAULT_ROLLS = (7, 14, 28)


def freq_to_lags(freq: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (lags, rolls) appropriate for the given aggregation frequency.

    Using daily lags (1, 7, 14, 28) on monthly data wastes 58% of the
    available history and adds near-zero signal at the wrong granularity.
    This helper maps each canonical frequency to a proportionally equivalent
    lag/roll set.

    D  → (1, 7, 14, 28)  rolls (7, 14, 28)   — daily default
    W  → (1, 2,  4,  8)  rolls (2,  4,  8)   — weekly (~daily scaled by 7)
    MS → (1, 2,  3, 12)  rolls (2,  3,  6)   — monthly, includes year-ago lag
    M  → same as MS
    """
    f = (freq or "D").upper()
    if f in ("MS", "M"):
        return (1, 2, 3, 12), (2, 3, 6)
    if f == "W":
        return (1, 2, 4, 8), (2, 4, 8)
    return (1, 7, 14, 28), (7, 14, 28)


def add_calendar_features(
    df: pd.DataFrame, date_col: str = "date", freq: str = "D"
) -> pd.DataFrame:
    """Add calendar features appropriate for the given aggregation frequency.

    For monthly data (MS/M) the canonical date is always the 1st of the
    month, so day-of-week and week-of-year features are near-constant and
    become pure noise that dominates feature importance. They are suppressed
    based on ``freq``:

    freq="D"        → all features (default, backward-compatible)
    freq="W"        → drop is_weekend, dayofweek, dow_sin/cos, day
    freq="MS"/"M"   → additionally drop weekofyear
    """
    out = df.copy()
    d = out[date_col]
    f = (freq or "D").upper()
    monthly = f in ("MS", "M")
    weekly = f == "W"

    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["dayofyear"] = d.dt.dayofyear
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    # cyclical month encoding is valid at all frequencies
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # Sub-monthly features: meaningless when all dates in a period share the
    # same week number (monthly canonical date is always the 1st).
    if not monthly:
        out["weekofyear"] = d.dt.isocalendar().week.astype(int)

    # Daily-only features: day-of-week is constant within a weekly/monthly
    # period so including it only adds noise for those aggregations.
    if not monthly and not weekly:
        out["day"] = d.dt.day
        out["dayofweek"] = d.dt.dayofweek
        out["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
        out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
        out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str = "sales",
    entity_col: str = "entity_id",
    lags: Iterable[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    out = df.sort_values([entity_col, "date"]).copy()
    g = out.groupby(entity_col, sort=False)[target]
    for lag in lags:
        out[f"{target}_lag_{lag}"] = g.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "sales",
    entity_col: str = "entity_id",
    windows: Iterable[int] = DEFAULT_ROLLS,
    base_lag: int = 1,
) -> pd.DataFrame:
    """Rolling stats computed on shifted target so we don't leak the current value."""
    out = df.sort_values([entity_col, "date"]).copy()
    shifted = out.groupby(entity_col, sort=False)[target].shift(base_lag)
    for w in windows:
        out[f"{target}_rmean_{w}"] = (
            shifted.groupby(out[entity_col], sort=False).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        out[f"{target}_rstd_{w}"] = (
            shifted.groupby(out[entity_col], sort=False).rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
        )
    return out


def encode_categoricals(
    df: pd.DataFrame,
    cols: Iterable[str],
    method: str = "ordinal",
) -> pd.DataFrame:
    """Simple, leak-safe categorical encoding. Ordinal by sorted unique values."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if method == "ordinal":
            out[c] = out[c].astype("category").cat.codes.astype(int)
        else:
            raise ValueError(f"Unknown method {method}")
    return out


def build_feature_matrix(
    canon: pd.DataFrame,
    target: str = "sales",
    lags: Iterable[int] = DEFAULT_LAGS,
    rolls: Iterable[int] = DEFAULT_ROLLS,
    drop_initial_nans: bool = True,
    freq: str = "D",
) -> pd.DataFrame:
    """End-to-end feature build for forecasting / regression."""
    out = add_calendar_features(canon, freq=freq)
    out = add_lag_features(out, target=target, lags=lags)
    out = add_rolling_features(out, target=target, windows=rolls)
    # Preserve signal from leaky contemporaneous columns by adding their lags.
    # Use the same freq-aware lags as the main target so aux lag semantics
    # stay consistent: (1, 7) on weekly data means 1-week and 7-week lags,
    # not the intended 1-day and 7-day lags.  Taking the two shortest lags
    # from the main lag set keeps the feature count manageable.
    aux_to_lag = [c for c in LEAKY_CONTEMPORANEOUS if c in out.columns and c != target]
    if aux_to_lag:
        _aux_lags = tuple(sorted(lags))[:2]  # two shortest lags, freq-appropriate
        out = add_aux_lag_features(out, aux_to_lag, lags=_aux_lags)

    # encode any object columns (entity_id, category names, etc.) as ordinals
    obj_cols = [c for c in out.columns if out[c].dtype == object and c != "date"]
    out = encode_categoricals(out, obj_cols, method="ordinal")

    if drop_initial_nans:
        max_lag = max(list(lags) + list(rolls))
        out = out.dropna(subset=[f"{target}_lag_{max(lags)}"])
    return out


# Columns that must NEVER be used as same-day predictors of `sales` because they
# are contemporaneous with sales — you do not know them at forecast time:
#
#   - quantity, unit_price, discount, profit
#       Accounting identity: sales ≈ quantity × unit_price − discount + profit.
#       Including them just measures arithmetic, not actual drivers.
#
#   - customers
#       Same-day foot traffic. By the time you know today's customer count,
#       today's sales are already realised. Letting the model see it leaks
#       the answer (R² jumps to ~0.98 — a clear sign of leakage).
#
# Their LAGGED versions (e.g. customers_lag_7) are legitimate and added separately
# via `add_aux_lag_features`.
LEAKY_CONTEMPORANEOUS = {"quantity", "unit_price", "discount", "profit", "customers"}
# Backwards-compat alias
SALES_TAUTOLOGY = LEAKY_CONTEMPORANEOUS


def add_aux_lag_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    entity_col: str = "entity_id",
    lags: Iterable[int] = (1, 7),
) -> pd.DataFrame:
    """Add lagged versions of contemporaneous variables (customers, etc.) so the
    signal is preserved without leakage. Original same-day column is left in
    place but `feature_columns` excludes it via LEAKY_CONTEMPORANEOUS.
    """
    out = df.sort_values([entity_col, "date"]).copy()
    for c in columns:
        if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        g = out.groupby(entity_col, sort=False)[c]
        for lag in lags:
            out[f"{c}_lag_{lag}"] = g.shift(lag)
    return out


# Column-name patterns that indicate transaction / row identifiers.  After
# daily aggregation these become mean(order_id_for_the_day) — a noisy proxy
# for time that duplicates the calendar features already in the matrix.
# Lagged versions (e.g. "order_id_lag_1") are equally useless and are also
# excluded via the prefix check below.
_TXN_ID_PATTERNS = (
    "order id", "order_id", "orderid",
    "sale id",  "sale_id",  "saleid",
    "invoice id", "invoice_id", "invoiceid",
    "transaction id", "transaction_id", "transactionid",
    "receipt id", "receipt_id", "receiptid",
    "ticket id", "ticket_id", "ticketid",
)


def feature_columns(matrix: pd.DataFrame, target: str = "sales") -> list[str]:
    """Return the list of usable feature columns.

    Excludes:
      - date, the target itself
      - leaky contemporaneous columns (quantity, unit_price, …)
      - non-numeric / datetime columns
      - transaction-ID columns (order_id, sale_id, invoice_id, …): after
        daily aggregation these are mean(row_id) — a noisy time-proxy that
        duplicates calendar features and causes overfitting.  Their lagged
        variants are equally uninformative and are also excluded.

    Lagged versions of legitimately excluded columns such as customers_lag_7
    are NOT excluded — they are legitimate predictors.
    """
    excluded = {"date", target}
    if target == "sales":
        excluded |= LEAKY_CONTEMPORANEOUS
    cols = []
    for c in matrix.columns:
        if c in excluded:
            continue
        if pd.api.types.is_datetime64_any_dtype(matrix[c]):
            continue
        if not pd.api.types.is_numeric_dtype(matrix[c]):
            continue
        # Drop transaction-ID columns and their lags
        c_lower = c.lower()
        if any(pat in c_lower for pat in _TXN_ID_PATTERNS):
            continue
        cols.append(c)
    return cols
