"""File ingestion — accepts CSV, Excel, Parquet, or multiple files.

Handles type inference for dates and numerics. Returns plain DataFrames;
schema mapping happens in mapper.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def load_file(path: PathLike, **read_kwargs) -> pd.DataFrame:
    """Read a single file by extension. Infers dates and numerics."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p, low_memory=False, **read_kwargs)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p, **read_kwargs)
    elif ext == ".parquet":
        df = pd.read_parquet(p, **read_kwargs)
    elif ext in (".tsv", ".txt"):
        df = pd.read_csv(p, sep="\t", low_memory=False, **read_kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return _infer_types(df)


def load_dataset(
    main_path: PathLike,
    auxiliary_paths: Optional[Iterable[PathLike]] = None,
    join_on: Optional[str] = None,
) -> pd.DataFrame:
    """Load a main file and optionally merge auxiliary lookups (e.g. Rossmann store.csv).

    If `join_on` is None and an auxiliary file shares exactly one column name
    with the main, that shared column is used as the join key.
    """
    main = load_file(main_path)
    if not auxiliary_paths:
        return main
    for aux_path in auxiliary_paths:
        aux = load_file(aux_path)
        if join_on is None:
            shared = [c for c in aux.columns if c in main.columns]
            if len(shared) != 1:
                raise ValueError(
                    f"Cannot auto-join {aux_path}: shared columns = {shared}. "
                    "Pass join_on explicitly."
                )
            key = shared[0]
        else:
            key = join_on
        main = main.merge(aux, on=key, how="left", suffixes=("", "_aux"))
    return main


_CURRENCY_PATTERN = re.compile(r"[\$€£¥₹₽¢฿,_\s]")
_PARENS_NEG = re.compile(r"^\((.+)\)$")


def _strip_currency(s: pd.Series) -> pd.Series:
    """Remove currency symbols, thousands separators, and convert
    accounting-style negatives like '(123.45)' → '-123.45'."""
    out = s.astype(str).str.strip()
    # Accounting negatives: (123) → -123
    out = out.str.replace(_PARENS_NEG, r"-\1", regex=True)
    # Strip currency symbols + separators
    out = out.str.replace(_CURRENCY_PATTERN, "", regex=True)
    # Common 'NaN-like' tokens
    out = out.replace({"": None, "-": None, "nan": None, "NaN": None,
                        "NULL": None, "N/A": None, "na": None})
    return out


def _infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort date and numeric parsing for object columns."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        sample = out[col].dropna().head(200)
        if sample.empty:
            continue
        # Try date parsing
        if _looks_date(col, sample):
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                out[col] = parsed
                continue
        # Try numeric — first directly
        numeric = pd.to_numeric(sample, errors="coerce")
        if numeric.notna().mean() > 0.95:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            continue
        # Then with currency/comma stripping for things like "$1,234.56" or "(123.45)"
        stripped_sample = _strip_currency(sample)
        numeric2 = pd.to_numeric(stripped_sample, errors="coerce")
        if numeric2.notna().mean() > 0.95:
            out[col] = pd.to_numeric(_strip_currency(out[col]), errors="coerce")
    return out


_DATE_NAME_HINTS = ("date", "dt", "day", "time", "timestamp", "period")


def _looks_date(col: str, sample: pd.Series) -> bool:
    name_hit = any(h in col.lower() for h in _DATE_NAME_HINTS)
    if name_hit:
        return True
    # Quick value-shape check: dates often have 8-19 chars with digits/separators
    s = sample.astype(str)
    avg_len = s.str.len().mean()
    if 6 <= avg_len <= 30 and s.str.contains(r"\d{2,4}[-/]\d{1,2}", regex=True).mean() > 0.5:
        return True
    return False
