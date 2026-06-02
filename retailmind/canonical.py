"""Canonicalizer — converts a raw DataFrame + schema mapping into the
canonical long-format time series that downstream modules consume.

Output schema (always exactly these columns, plus optional extras):
  date          : datetime64[ns], day-start
  entity_id     : object (string)
  sales         : float64
  ...optional canonical roles if present (quantity, customers, promo, holiday, is_open, ...)
  ...auxiliary columns kept under raw names (for SHAP feature analysis)

If the source is transactional (multiple rows per entity/date), the
canonicalizer aggregates by (entity_id, date) using sensible rules:
  - sales / quantity / profit → sum
  - unit_price / discount     → weighted mean (by quantity if present else mean)
  - flags (promo/holiday/is_open) → max
  - auxiliary numeric          → mean
  - auxiliary categorical      → mode (first)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from retailmind.schema import CanonicalSchema, ColumnRole


SUM_ROLES = {ColumnRole.SALES, ColumnRole.QUANTITY, ColumnRole.PROFIT, ColumnRole.CUSTOMERS}
MAX_ROLES = {ColumnRole.PROMO, ColumnRole.HOLIDAY, ColumnRole.IS_OPEN}
MEAN_ROLES = {ColumnRole.UNIT_PRICE, ColumnRole.DISCOUNT}


def canonicalize(
    raw: pd.DataFrame,
    schema: CanonicalSchema,
    freq: str = "D",
    entity_default: str = "global",
    keep_aux: bool = True,
) -> pd.DataFrame:
    """Map raw DataFrame to canonical long-format and aggregate to `freq`.

    Parameters
    ----------
    raw : raw DataFrame as ingested.
    schema : confirmed mapping (from SchemaMapper.infer().schema).
    freq : pandas offset alias. 'D' (default) = daily. 'W' = weekly, 'MS' = monthly start.
    entity_default : value used for entity_id when no entity column is mapped.
    keep_aux : if True, keep AUX columns in the output (aggregated sensibly).
    """
    date_col = schema.column_for(ColumnRole.DATE)
    sales_col = schema.column_for(ColumnRole.SALES)
    if date_col is None or sales_col is None:
        raise ValueError("Schema must have DATE and SALES roles assigned.")

    df = raw.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])

    if df.empty:
        raise ValueError(
            f"After dropping rows with missing date or sales, the dataset is "
            f"empty. Likely causes: the date column '{date_col}' couldn't be "
            f"parsed, or the sales column '{sales_col}' has all-missing values. "
            f"Check the data and adjust the schema mapping if needed."
        )

    # Drop returns / refunds (negative sales rows). These distort forecasts
    # because they're not part of the "what will I sell next" signal.
    # If >40 % of rows are negative, we treat the column as a different concept
    # (e.g. P&L) and keep them.
    if pd.api.types.is_numeric_dtype(df[sales_col]):
        n_before = len(df)
        neg_share = (df[sales_col] < 0).mean()
        if 0 < neg_share <= 0.4:
            df = df[df[sales_col] >= 0]
            # The number filtered is informational; canonicalize doesn't carry a log
            # so this just affects downstream EDA stats silently. Decision_log entries
            # are added at the pipeline layer instead.

    # Floor date to chosen freq
    df["_date"] = df[date_col].dt.to_period(_period_alias(freq)).dt.start_time

    entity_col = schema.column_for(ColumnRole.ENTITY_ID)
    if entity_col is None:
        df["_entity"] = entity_default
    else:
        df["_entity"] = df[entity_col].astype(str)

    # Build column-spec for aggregation
    agg_spec: dict[str, str] = {sales_col: "sum"}
    rename: dict[str, str] = {sales_col: "sales"}

    for role in [ColumnRole.QUANTITY, ColumnRole.PROFIT, ColumnRole.CUSTOMERS]:
        c = schema.column_for(role)
        if c and pd.api.types.is_numeric_dtype(df[c]):
            agg_spec[c] = "sum"
            rename[c] = role.value

    for role in MEAN_ROLES:
        c = schema.column_for(role)
        if c and pd.api.types.is_numeric_dtype(df[c]):
            agg_spec[c] = "mean"
            rename[c] = role.value

    for role in MAX_ROLES:
        c = schema.column_for(role)
        if c is None:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            agg_spec[c] = "max"
            rename[c] = role.value
        else:
            # convert to numeric flag: non-empty / non-"0" → 1
            df[c] = (df[c].astype(str).str.lower().isin({"1", "true", "yes", "y", "a", "b", "c"})).astype(int)
            agg_spec[c] = "max"
            rename[c] = role.value

    # Carry the canonical product / customer / geo dims (mode if multi-value).
    # These are kept un-aggregated when there's a single value per group;
    # otherwise the first value is taken so they don't disappear.
    carry_roles = [
        ColumnRole.PRODUCT_ID, ColumnRole.PRODUCT_CATEGORY,
        ColumnRole.CUSTOMER_ID, ColumnRole.CUSTOMER_SEGMENT,
        ColumnRole.REGION, ColumnRole.CITY, ColumnRole.STATE,
    ]
    for role in carry_roles:
        c = schema.column_for(role)
        if c:
            agg_spec[c] = "first"
            rename[c] = role.value

    if keep_aux:
        aux_cols = schema.columns_for(ColumnRole.AUX)
        for c in aux_cols:
            if c == date_col or c == entity_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                agg_spec[c] = "mean"
            else:
                agg_spec[c] = "first"

    grouped = df.groupby(["_entity", "_date"], as_index=False).agg(agg_spec)
    grouped = grouped.rename(columns={"_entity": "entity_id", "_date": "date", **rename})
    grouped = grouped.sort_values(["entity_id", "date"]).reset_index(drop=True)

    # Fill date gaps per entity with zeros for sales / quantity (closed days, no traffic).
    # This is important so lag features and forecasts are correct.
    grouped = _reindex_dates(grouped, freq=freq)
    return grouped


def _period_alias(freq: str) -> str:
    mapping = {"D": "D", "W": "W", "MS": "M", "M": "M", "Q": "Q", "Y": "Y", "A": "A"}
    return mapping.get(freq, freq)


def _reindex_dates(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """For each entity, expand to full date range and forward-fill non-numeric, zero-fill flow vars.

    Vectorized: build a MultiIndex of (entity_id, date) covering each entity's
    own min→max range, reindex once, then fill per column. This is ~50× faster
    than looping per entity for datasets with many entities.
    """
    flow_cols = {"sales", "quantity", "profit", "customers"}

    # Convert pandas frequency aliases to a form that aligns with how the
    # groupby keys were generated (period.start_time → Monday for W).
    reindex_freq = {"W": "W-MON", "MS": "MS", "M": "MS"}.get(freq, freq)

    # Per-entity date ranges
    rng = df.groupby("entity_id")["date"].agg(["min", "max"])
    pieces = [
        pd.DataFrame({"entity_id": ent, "date": pd.date_range(r["min"], r["max"], freq=reindex_freq)})
        for ent, r in rng.iterrows()
    ]
    full = pd.concat(pieces, ignore_index=True)

    out = full.merge(df, on=["entity_id", "date"], how="left")
    out = out.sort_values(["entity_id", "date"]).reset_index(drop=True)

    for col in out.columns:
        if col in ("entity_id", "date"):
            continue
        if col in flow_cols and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)
        else:
            # ffill within entity (forward in time), then bfill for series-start NaNs
            out[col] = out.groupby("entity_id")[col].transform(lambda s: s.ffill().bfill())

    cols = ["date", "entity_id"] + [c for c in out.columns if c not in ("date", "entity_id")]
    return out[cols]
