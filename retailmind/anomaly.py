"""Anomaly detection on canonical data.

Three independent detectors are combined into a single ranked score:
  1. Isolation Forest on engineered features (multivariate)
  2. STL residual z-score (seasonal-trend decomposition; per-entity)
  3. IQR rule on raw sales (per-entity)

Each anomaly comes with a 'reason' string so the user knows *why* it was flagged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from statsmodels.tsa.seasonal import STL
    HAVE_STL = True
except ImportError:
    HAVE_STL = False

from retailmind.features import build_feature_matrix, feature_columns


def detect_iqr(
    canon: pd.DataFrame, target: str = "sales", k: float = 3.0
) -> pd.DataFrame:
    """Per-entity IQR outliers. Robust and explainable."""
    rows = []
    for ent, g in canon.groupby("entity_id"):
        q1, q3 = g[target].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        out_mask = (g[target] < lo) | (g[target] > hi)
        for _, r in g[out_mask].iterrows():
            rows.append({
                "entity_id": ent,
                "date": r["date"],
                "value": r[target],
                "lower_bound": lo,
                "upper_bound": hi,
                "method": "iqr",
                "score": float(abs(r[target] - g[target].median()) / max(iqr, 1)),
                "reason": (
                    f"{target}={r[target]:.0f} outside IQR window [{lo:.0f}, {hi:.0f}]"
                ),
            })
    return pd.DataFrame(rows)


def detect_stl_residual(
    canon: pd.DataFrame, target: str = "sales", season: int = 7, z_thresh: float = 3.5
) -> pd.DataFrame:
    """STL decomposition + z-score on residuals (handles seasonality before flagging)."""
    if not HAVE_STL:
        return pd.DataFrame()
    rows = []
    for ent, g in canon.sort_values("date").groupby("entity_id"):
        if len(g) < 2 * season + 1:
            continue
        series = g.set_index("date")[target].astype(float)
        try:
            stl = STL(series, period=season, robust=True).fit()
        except Exception:
            continue
        resid = stl.resid.dropna()
        if resid.std() == 0 or resid.empty:
            continue
        z = (resid - resid.mean()) / resid.std()
        flagged = z[z.abs() > z_thresh]
        for d, score in flagged.items():
            rows.append({
                "entity_id": ent,
                "date": d,
                "value": float(series.loc[d]),
                "method": "stl_residual",
                "score": float(abs(score)),
                "reason": (
                    f"STL residual z={score:.1f} (seasonally adjusted, expected≈{stl.trend.loc[d] + stl.seasonal.loc[d]:.0f})"
                ),
            })
    return pd.DataFrame(rows)


def detect_isolation_forest(
    canon: pd.DataFrame, contamination: float = 0.01, target: str = "sales",
    seed: int = 42, freq: str = "D",
) -> pd.DataFrame:
    """Multivariate IsolationForest on engineered features (catches subtle joint anomalies)."""
    matrix = build_feature_matrix(canon, target=target, freq=freq)
    feats = feature_columns(matrix, target=target)
    if matrix.empty:
        return pd.DataFrame()
    X = matrix[feats].fillna(0)
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    raw = model.fit_predict(X)
    scores = -model.score_samples(X)  # higher = more anomalous
    is_out = raw == -1
    out = matrix.loc[is_out, ["entity_id", "date"]].copy()
    out["value"] = matrix.loc[is_out, target].values
    out["score"] = scores[is_out]
    out["method"] = "isolation_forest"
    out["reason"] = [
        f"isolation_forest score={s:.2f} (top-{contamination * 100:.1f}% tail)"
        for s in scores[is_out]
    ]
    # Restore original entity_id strings when the matrix ordinal-encoded them.
    # Previously this used a date-only merge which produced a 1115× fanout
    # (every anomaly matched all stores sharing that date).  The correct fix
    # is to decode the integer codes back to the original category labels using
    # the same category order that encode_categoricals produced.
    # Restore original entity_id strings if the matrix ordinal-encoded them.
    # build_feature_matrix encodes object-dtype categoricals to int via cat.codes,
    # but ArrowStringArray-backed columns (dtype="str") are NOT encoded and stay
    # as strings already.  Only run the decode when entity_id is truly numeric.
    if "entity_id" in canon.columns and pd.api.types.is_numeric_dtype(out["entity_id"]):
        try:
            int_codes = out["entity_id"].astype("int64")
            cat_codes = canon["entity_id"].astype("category")
            # Use explicit int keys to avoid numpy-int vs Python-int hash mismatches
            code_map = {int(k): v for k, v in enumerate(cat_codes.cat.categories)}
            mapped = int_codes.map(code_map)
            if mapped.notna().any():   # only apply if mapping actually succeeded
                out["entity_id"] = mapped
        except (ValueError, TypeError):
            pass  # entity_id is already a usable label — nothing to decode
    return out.reset_index(drop=True)


def detect_all(
    canon: pd.DataFrame,
    target: str = "sales",
    iqr_k: float = 3.0,
    stl_z: float = 3.5,
    if_contamination: float = 0.01,
    freq: str = "D",
) -> pd.DataFrame:
    """Combine all detectors. Sort by score desc."""
    pieces = [
        detect_iqr(canon, target=target, k=iqr_k),
        detect_stl_residual(canon, target=target, z_thresh=stl_z),
        detect_isolation_forest(canon, target=target, contamination=if_contamination, freq=freq),
    ]
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values("score", ascending=False).reset_index(drop=True)
