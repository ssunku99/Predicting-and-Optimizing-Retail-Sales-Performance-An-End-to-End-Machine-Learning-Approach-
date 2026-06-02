"""Quick profile of a raw dataset before the rest of the pipeline runs.

Outputs:
  - quality_score (0-100): rough sanity check based on missing %, duplicates,
    presence of a date column, and presence of a numeric column that could
    plausibly be sales.
  - suggested_freq ('D' / 'W' / 'MS'): based on row density per entity. If
    each entity gets less than ~5 observations per week, the pipeline switches
    to weekly aggregation.
  - candidate_sales_cols / candidate_entity_cols: lists used by the schema
    mapper as fallbacks when name detection isn't decisive.

The `DecisionLog` is just a list of (decision, reason, severity) tuples that
the pipeline appends to as it makes auto-choices, so the UI can show them
to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DecisionLog:
    """A running log of every auto-decision the pipeline made.
    Shown in the UI so the reviewer sees what was inferred and why.
    """
    entries: list[dict] = field(default_factory=list)

    def log(self, decision: str, reason: str, severity: str = "info") -> None:
        self.entries.append({"decision": decision, "reason": reason, "severity": severity})

    def render(self) -> str:
        if not self.entries:
            return "_(no auto-decisions made — schema was clean)_"
        lines = []
        icons = {"info": "ℹ️", "warn": "⚠️", "fix": "🔧", "skip": "⏭️"}
        for e in self.entries:
            icon = icons.get(e["severity"], "•")
            lines.append(f"{icon} **{e['decision']}** — {e['reason']}")
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        return list(self.entries)


@dataclass
class DataProfile:
    """Quantitative profile of a raw dataset (pre-mapping)."""
    rows: int
    cols: int
    pct_missing: float
    duplicate_rows: int
    date_columns: list[str]
    numeric_columns: list[str]
    text_columns: list[str]
    candidate_sales_cols: list[str]      # plausible $ / volume targets
    candidate_entity_cols: list[str]      # low-cardinality categoricals
    has_quantity: bool
    has_unit_price: bool
    has_explicit_sales: bool
    suggested_freq: str                   # 'D' / 'W' / 'MS'
    quality_score: int                    # 0-100

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# Heuristic name lists kept in sync with mapper.py
_SALES_NAME_HINTS = ("sales", "revenue", "amount", "total", "value", "payment",
                      "income", "earnings")
_QTY_NAME_HINTS = ("quantity", "qty", "units", "volume")
_PRICE_NAME_HINTS = ("price", "rate", "msrp", "cost")
_ENTITY_NAME_HINTS = ("store", "outlet", "branch", "shop", "location",
                       "warehouse", "site", "seller", "merchant")
_GEO_NAME_HINTS = ("region", "country", "state", "city", "province")
_DATE_NAME_HINTS = ("date", "time", "dt", "day", "timestamp", "period")
_ID_NAME_HINTS = ("id", "code", "key", "uuid", "guid", "sku", "barcode")


def profile(df: pd.DataFrame) -> DataProfile:
    """Compute a quantitative profile of the raw DataFrame."""
    if df.empty:
        return DataProfile(0, 0, 0, 0, [], [], [], [], [], False, False, False, "D", 0)

    rows, cols = df.shape
    pct_missing = float(df.isna().mean().mean() * 100)
    duplicate_rows = int(df.duplicated().sum())

    date_columns = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not date_columns:
        # name-based fallback for object cols that look like dates
        for c in df.columns:
            if df[c].dtype != object:
                continue
            if any(h in c.lower() for h in _DATE_NAME_HINTS):
                sample = df[c].dropna().head(200)
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    date_columns.append(c)

    numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text_columns = [c for c in df.columns if df[c].dtype == object]

    # Candidate sales columns: numeric, not ID-like, has variance, mostly positive
    candidate_sales_cols = []
    for c in numeric_columns:
        name_lc = c.lower()
        if any(h in name_lc for h in _ID_NAME_HINTS):
            continue
        s = df[c].dropna()
        if s.empty or s.nunique() <= 2:
            continue
        if (s < 0).mean() > 0.15:
            continue
        candidate_sales_cols.append(c)

    # Candidate entity columns: low cardinality categoricals
    candidate_entity_cols = []
    for c in df.columns:
        n_unique = df[c].nunique(dropna=True)
        if 1 < n_unique <= max(100, int(np.sqrt(rows))):
            if df[c].dtype == object or pd.api.types.is_integer_dtype(df[c]):
                candidate_entity_cols.append(c)

    has_quantity = any(any(h in c.lower() for h in _QTY_NAME_HINTS) for c in numeric_columns)
    has_unit_price = any(
        any(h in c.lower() for h in _PRICE_NAME_HINTS) and not any(s in c.lower() for s in _SALES_NAME_HINTS)
        for c in numeric_columns
    )
    has_explicit_sales = any(any(h in c.lower() for h in _SALES_NAME_HINTS) for c in numeric_columns)

    # Suggested aggregation frequency: based on data density per "natural" entity
    suggested_freq = _suggest_frequency(df, date_columns, candidate_entity_cols)

    # Quality score
    quality_score = _score_quality(rows, pct_missing, duplicate_rows,
                                    bool(date_columns), bool(candidate_sales_cols))

    return DataProfile(
        rows=rows, cols=cols,
        pct_missing=round(pct_missing, 2),
        duplicate_rows=duplicate_rows,
        date_columns=date_columns,
        numeric_columns=numeric_columns,
        text_columns=text_columns,
        candidate_sales_cols=candidate_sales_cols,
        candidate_entity_cols=candidate_entity_cols,
        has_quantity=has_quantity,
        has_unit_price=has_unit_price,
        has_explicit_sales=has_explicit_sales,
        suggested_freq=suggested_freq,
        quality_score=quality_score,
    )


def _suggest_frequency(df: pd.DataFrame, date_cols: list[str],
                        entity_cols: list[str]) -> str:
    """Pick D / W / MS based on observed density on the *top-decile* entities.

    Rationale: tail entities with very little data will be filtered later
    anyway; what matters is whether the entities we actually forecast have
    enough density for daily granularity. Using the 90th-percentile of
    per-entity row count instead of median avoids being dragged down by a
    long tail of low-volume entities.
    """
    if not date_cols:
        return "D"
    date_col = date_cols[0]
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return "D"
    span_days = (dates.max() - dates.min()).days + 1
    if span_days <= 0:
        return "D"

    if entity_cols:
        ent_col = entity_cols[0]
        per_entity = df.groupby(ent_col)[date_col].count()
        # use 90th percentile so the freq fits the entities we will actually keep
        p90_obs = float(per_entity.quantile(0.90))
        obs_per_week = p90_obs / (span_days / 7)
    else:
        obs_per_week = len(df) / (span_days / 7)

    if obs_per_week >= 5:
        return "D"
    if obs_per_week >= 1:
        return "W"
    return "MS"


def _score_quality(rows: int, pct_missing: float, duplicates: int,
                    has_date: bool, has_sales: bool) -> int:
    """0–100 quality score. Rough but useful as a sanity check at the top."""
    score = 100
    if not has_date: score -= 50
    if not has_sales: score -= 50
    if rows < 30: score -= 30
    elif rows < 200: score -= 10
    score -= int(min(pct_missing, 50))
    score -= int(min(duplicates / max(rows, 1) * 100, 20))
    return max(0, min(100, score))
