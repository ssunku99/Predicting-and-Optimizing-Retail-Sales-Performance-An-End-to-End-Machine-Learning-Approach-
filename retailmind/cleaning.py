"""Data cleaning — the explicit clean stage of the pipeline.

This used to live inline inside canonicalize(). It is pulled out into its own
module so that cleaning is a named, testable, loggable stage rather than a
side effect of aggregation.

The single entry point is clean_dataframe(). It is deliberately *idempotent*:
running it twice on the same frame produces the same result and logs nothing
the second time, so canonicalize() can still call it defensively even when the
pipeline has already run it as a separate stage.

What it does, in order:
  1. Coerce the date column to real datetimes; unparseable dates become NaT.
  2. Drop rows with a missing date or a missing sales value (cannot forecast).
  3. Handle returns / refunds: drop negative-sales rows, UNLESS more than 40%
     of rows are negative, in which case the column is probably a profit / P&L
     measure and the rows are kept untouched.

It does NOT remove statistical outliers. Large spikes are surfaced later by
the anomaly stage and reported, not deleted, because in retail a big spike is
often a real bulk order rather than an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from retailmind.schema import CanonicalSchema, ColumnRole


@dataclass
class CleaningReport:
    """What the clean stage did, for the decision log and the notebook."""
    rows_in: int
    rows_out: int
    bad_dates_dropped: int
    missing_sales_dropped: int
    returns_dropped: int
    returns_kept_as_pnl: bool

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_removed": self.rows_removed,
            "bad_dates_dropped": self.bad_dates_dropped,
            "missing_sales_dropped": self.missing_sales_dropped,
            "returns_dropped": self.returns_dropped,
            "returns_kept_as_pnl": self.returns_kept_as_pnl,
        }


def clean_dataframe(
    raw: pd.DataFrame,
    schema: CanonicalSchema,
    log=None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw frame against its schema. Returns (cleaned_df, report).

    Parameters
    ----------
    raw : raw DataFrame as ingested / smart-inferred.
    schema : confirmed mapping with at least DATE and SALES roles.
    log : optional DecisionLog; when provided, each non-trivial action is
          recorded so it surfaces in the app and notebook.
    """
    date_col = schema.column_for(ColumnRole.DATE)
    sales_col = schema.column_for(ColumnRole.SALES)
    if date_col is None or sales_col is None:
        raise ValueError("Schema must have DATE and SALES roles assigned.")

    df = raw.copy()
    rows_in = len(df)

    # 1. Coerce dates. Anything unparseable becomes NaT.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    bad_dates = int(df[date_col].isna().sum())

    # 2. Drop rows with no date or no sales value.
    missing_sales = int(df[sales_col].isna().sum())
    df = df.dropna(subset=[date_col, sales_col])

    if df.empty:
        raise ValueError(
            f"After dropping rows with missing date or sales, the dataset is "
            f"empty. Likely causes: the date column '{date_col}' couldn't be "
            f"parsed, or the sales column '{sales_col}' has all-missing values. "
            f"Check the data and adjust the schema mapping if needed."
        )

    # 3. Returns / refunds.
    returns_dropped = 0
    returns_kept_as_pnl = False
    if pd.api.types.is_numeric_dtype(df[sales_col]):
        neg_mask = df[sales_col] < 0
        neg_share = float(neg_mask.mean())
        if 0 < neg_share <= 0.4:
            returns_dropped = int(neg_mask.sum())
            df = df[~neg_mask]
        elif neg_share > 0.4:
            returns_kept_as_pnl = True

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(df),
        bad_dates_dropped=bad_dates,
        missing_sales_dropped=missing_sales,
        returns_dropped=returns_dropped,
        returns_kept_as_pnl=returns_kept_as_pnl,
    )

    # Log only when something actually happened (keeps idempotent re-runs quiet).
    if log is not None and report.rows_removed > 0:
        bits = []
        if bad_dates:
            bits.append(f"{bad_dates:,} unparseable dates")
        if missing_sales:
            bits.append(f"{missing_sales:,} missing-sales rows")
        if returns_dropped:
            bits.append(f"{returns_dropped:,} return/refund rows (negative sales)")
        log.log(
            f"Cleaned data: removed {report.rows_removed:,} of {rows_in:,} rows",
            "Dropped " + ", ".join(bits) + "." if bits else
            f"Removed {report.rows_removed:,} rows.",
            severity="fix",
        )
    if log is not None and returns_kept_as_pnl:
        log.log(
            "Kept negative-sales rows",
            "More than 40% of rows have negative sales, so the column is most "
            "likely a profit / P&L measure rather than units sold. Rows retained.",
            severity="info",
        )

    return df, report
