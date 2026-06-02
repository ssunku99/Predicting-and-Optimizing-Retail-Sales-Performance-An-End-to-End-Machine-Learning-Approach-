"""Fixups that run between the schema mapper and the canonicalizer.

The mapper does name-based detection. These functions handle the cases
where name detection isn't enough — e.g. a dataset has Quantity and
UnitPrice but no Sales column, so we should compute Sales = Qty * Price.

Each function returns (df, schema, log). The log captures what we did
so the user/app can see why a decision was made.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from retailmind.schema import CanonicalSchema, ColumnRole
from retailmind.profiler import DataProfile, DecisionLog


SYNTH_SALES_COL = "_retailmind_synth_revenue"


def derive_synthetic_sales(df: pd.DataFrame, schema: CanonicalSchema,
                            log: DecisionLog) -> tuple[pd.DataFrame, CanonicalSchema]:
    """If both QUANTITY and UNIT_PRICE are mapped but no SALES, derive
    `_retailmind_synth_revenue = quantity × unit_price` and assign it to SALES.

    This handles the UCI Online Retail case (and many e-commerce exports)
    where revenue isn't stored directly but can be computed.
    """
    if schema.has(ColumnRole.SALES):
        return df, schema

    qty_col = schema.column_for(ColumnRole.QUANTITY)
    price_col = schema.column_for(ColumnRole.UNIT_PRICE)
    if not (qty_col and price_col):
        return df, schema

    df = df.copy()
    qty = pd.to_numeric(df[qty_col], errors="coerce")
    price = pd.to_numeric(df[price_col], errors="coerce")
    df[SYNTH_SALES_COL] = qty * price

    new_mapping = dict(schema.mapping)
    new_mapping[SYNTH_SALES_COL] = ColumnRole.SALES
    # Keep qty and price as their own roles (they're still useful as lagged features)
    schema = CanonicalSchema(mapping=new_mapping)

    pct_pos = float((df[SYNTH_SALES_COL] > 0).mean() * 100)
    log.log(
        f"Derived revenue = '{qty_col}' × '{price_col}'",
        f"No explicit sales column existed, but both quantity and unit_price did. "
        f"Synthetic revenue column ({pct_pos:.0f}% positive) is now the forecast target. "
        f"This gives $-denominated forecasts instead of just unit volume.",
        severity="fix",
    )
    return df, schema


def promote_product_to_entity(df: pd.DataFrame, schema: CanonicalSchema,
                               log: DecisionLog) -> tuple[pd.DataFrame, CanonicalSchema]:
    """If no entity_id and product_category exists with low cardinality, use it.

    This handles e-commerce / influencer datasets where the natural business
    unit is the product line, not a geographic location.  We prefer
    product_category over product_id because categories give denser per-entity
    time series (fewer zeros) and more interpretable forecasts.

    Cardinality guard: only promote if 2 ≤ n_unique ≤ 20.  More than 20
    product categories would produce very sparse per-category time series that
    are no better than the geo fallback.  This function runs BEFORE
    promote_geo_to_entity so a clean product split wins over a noisy city split.
    """
    if schema.has(ColumnRole.ENTITY_ID):
        return df, schema

    for role in (ColumnRole.PRODUCT_CATEGORY, ColumnRole.PRODUCT_ID):
        c = schema.column_for(role)
        if not c:
            continue
        n_unique = df[c].nunique(dropna=True)
        if n_unique < 2 or n_unique > 20:
            continue
        new_mapping = dict(schema.mapping)
        new_mapping[c] = ColumnRole.ENTITY_ID
        schema = CanonicalSchema(mapping=new_mapping)
        log.log(
            f"Promoted '{c}' ({role.value}) → entity_id",
            f"No store/outlet column found. '{c}' has {n_unique} unique values — "
            f"forecasting per-{role.value} gives denser, more meaningful series than "
            f"a high-cardinality geo split.",
            severity="fix",
        )
        return df, schema
    return df, schema


def promote_geo_to_entity(df: pd.DataFrame, schema: CanonicalSchema,
                            log: DecisionLog) -> tuple[pd.DataFrame, CanonicalSchema]:
    """If no entity_id and we have a geo column, promote it (Region > State > Country > City).

    Per-region forecasts are vastly more useful than a single global series.
    """
    if schema.has(ColumnRole.ENTITY_ID):
        return df, schema

    for role in (ColumnRole.REGION, ColumnRole.STATE, ColumnRole.CITY):
        c = schema.column_for(role)
        if not c:
            continue
        n_unique = df[c].nunique(dropna=True)
        if n_unique < 2 or n_unique > 200:
            # too few (one region) or too many (basically free text) — skip
            continue
        new_mapping = dict(schema.mapping)
        new_mapping[c] = ColumnRole.ENTITY_ID
        schema = CanonicalSchema(mapping=new_mapping)
        log.log(
            f"Promoted '{c}' ({role.value}) → entity_id",
            f"No store/outlet column was detected, but '{c}' has {n_unique} unique values. "
            f"Forecasting per-{role.value} gives separate models for each, instead of one "
            f"low-quality global series.",
            severity="fix",
        )
        return df, schema
    return df, schema


def choose_aggregation_freq(df: pd.DataFrame, schema: CanonicalSchema,
                              profile: DataProfile, requested_freq: Optional[str],
                              log: DecisionLog) -> str:
    """Pick D / W / MS. If user passed a freq explicitly, respect it.
    Otherwise use the profiler's suggestion and log the reason."""
    if requested_freq and requested_freq != "auto":
        return requested_freq

    freq = profile.suggested_freq
    reasons = {
        "D": "daily aggregation works (≥ 5 obs/entity/week on average)",
        "W": "daily was too sparse; switched to weekly to smooth noise (1-5 obs/entity/week)",
        "MS": "very sparse data — escalated to monthly aggregation (< 1 obs/entity/week)",
    }
    if freq != "D":
        log.log(f"Auto-aggregation: {freq}", reasons[freq], severity="fix")
    return freq


def adaptive_forecast_params(canon: pd.DataFrame, freq: str = "D") -> dict:
    """Decide LightGBM hyperparameters & feature config based on data size and frequency.

    Tiny data → fewer lags, fewer leaves, fewer rounds → won't overfit.
    Big data → richer model.
    Lag values are now chosen based on ``freq`` so that daily lags are not
    blindly applied to weekly or monthly data (where lag-28 means 28 months).
    """
    from retailmind.features import freq_to_lags

    n_rows = len(canon)
    n_entities = canon["entity_id"].nunique()
    rows_per_entity = n_rows / max(n_entities, 1)

    # Start from frequency-appropriate lags instead of hardcoded daily defaults
    default_lags, default_rolls = freq_to_lags(freq)

    params = {"lags": default_lags, "rolls": default_rolls,
              "n_estimators": 400, "learning_rate": 0.05,
              "num_leaves": 63, "min_data_in_leaf": 20,
              "cv_folds": 3}

    decisions = []
    if rows_per_entity < 60:
        # Shrink from the freq-appropriate base; take only the two shortest lags
        params["lags"] = default_lags[:2]
        params["rolls"] = (default_rolls[0],)
        params["n_estimators"] = 100
        params["num_leaves"] = 15
        params["min_data_in_leaf"] = 5
        params["cv_folds"] = 2
        decisions.append(("Reduced LightGBM complexity for small data",
                          f"Only ~{rows_per_entity:.0f} rows/entity — using "
                          f"lag={params['lags']}, 100 trees, 15 leaves to avoid overfitting"))
    elif rows_per_entity < 200:
        params["lags"] = default_lags[:3]   # drop the longest lag
        params["n_estimators"] = 200
        params["num_leaves"] = 31
        params["cv_folds"] = 2
        decisions.append(("Reduced lag horizon for moderate data",
                          f"~{rows_per_entity:.0f} rows/entity — dropped longest lag "
                          f"(kept {params['lags']})"))
    params["_decisions"] = decisions
    return params


def apply_smart_inference(df: pd.DataFrame, schema: CanonicalSchema,
                            profile: DataProfile, requested_freq: Optional[str],
                            log: DecisionLog) -> tuple[pd.DataFrame, CanonicalSchema, str]:
    """Run all smart-inference layers in order. Returns (df, schema, freq)."""
    df, schema = derive_synthetic_sales(df, schema, log)
    df, schema = promote_product_to_entity(df, schema, log)   # runs first (denser series)
    df, schema = promote_geo_to_entity(df, schema, log)       # fallback if no product split
    freq = choose_aggregation_freq(df, schema, profile, requested_freq, log)
    return df, schema, freq
