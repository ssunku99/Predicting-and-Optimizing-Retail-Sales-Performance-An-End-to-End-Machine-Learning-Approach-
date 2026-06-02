"""Sales-driver regression — quantifies what moves the needle.

Improvements over a simple 80/20 split:
  - **log_target=True** (default for skewed retail data): trains on log1p(sales)
    and reports metrics on the original scale. Often boosts R² by 10–40 points
    on real-world retail data, where sales are log-normally distributed.
  - **walk-forward CV** instead of a single split: averages R² across multiple
    time-ordered folds, which is fairer when sales trend over time.
  - **baseline comparison**: reports R²-improvement vs a seasonal-naïve baseline
    so the reviewer sees how much *new* signal the model is capturing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False

from retailmind.features import build_feature_matrix, feature_columns


@dataclass
class RegressionReport:
    model_name: str
    r2: float
    rmse: float
    importance: pd.DataFrame
    holdout_size: int
    # Diagnostics
    log_target_used: bool = False
    cv_folds: int = 0
    baseline_r2: Optional[float] = None       # seasonal-naïve R² on the same holdout
    r2_lift_vs_baseline: Optional[float] = None
    per_fold: list[dict] = field(default_factory=list)
    # Safety-net: True when mean R² < 0 across all CV folds, meaning the model
    # is worse than simply predicting the grand mean.
    use_baseline: bool = False
    why_baseline: str = ""


def _r2_safe(y_true, y_pred) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _seasonal_naive_pred(train: pd.DataFrame, test: pd.DataFrame,
                          target: str, season: int = 7) -> np.ndarray:
    """y_hat[t] = y[t - season] per entity, then extract the test rows.

    The naive global shift is wrong for multi-entity feature matrices: rows
    from different entities are interleaved when sorted by date, so a global
    shift(season) crosses entity boundaries and returns values from the wrong
    entity.  Grouping by entity_id before shifting fixes this.
    """
    fallback = float(train[target].mean())
    combined = pd.concat([train, test], ignore_index=True)

    if "entity_id" in combined.columns:
        # Per-entity shift; groupby preserves the combined integer index
        shifted = combined.groupby("entity_id", sort=False)[target].shift(season)
        combined = combined.copy()
        combined["_sn_shifted"] = shifted.values
        yhat = combined["_sn_shifted"].values[len(train):]
    else:
        yhat = combined[target].shift(season).values[len(train):]

    yhat = np.where(np.isnan(yhat), fallback, yhat)
    return yhat


def fit_driver_model(
    canon: pd.DataFrame,
    target: str = "sales",
    cv_folds: int = 3,
    log_target: Optional[bool] = None,
    seed: int = 42,
    lags: Optional[tuple] = None,
    rolls: Optional[tuple] = None,
    freq: str = "D",
) -> RegressionReport:
    """Fit LightGBM (or Ridge fallback) with walk-forward CV.

    Parameters
    ----------
    log_target : if True, train on log1p(sales); if None, auto-enable when the
                 sales distribution is right-skewed (skew > 1.5). Default None.
    lags : lag offsets to build as features. Defaults to freq-appropriate values
           via freq_to_lags(freq) so weekly/monthly data doesn't get daily lags.
    rolls : rolling window sizes. Same default logic as lags.
    """
    # Default lags/rolls based on frequency so weekly data doesn't inherit
    # daily lag semantics (lag_28 on weekly data = 28 weeks = 7 months back).
    from retailmind.features import freq_to_lags as _freq_to_lags
    _default_lags, _default_rolls = _freq_to_lags(freq)
    if lags is None:
        lags = _default_lags
    if rolls is None:
        rolls = _default_rolls

    # Auto-shrink lags for small datasets
    n_rows = len(canon)
    n_ents = canon["entity_id"].nunique() if "entity_id" in canon.columns else 1
    rows_per_ent = int(n_rows / max(n_ents, 1))
    if rows_per_ent < max(lags) + 5:
        lags = tuple(int(l) for l in lags if l < rows_per_ent - 5) or (1,)
        rolls = tuple(int(r) for r in rolls if r < rows_per_ent - 5) or (max(int(min(rows_per_ent // 2, 7)), 2),)

    matrix = build_feature_matrix(canon, target=target, lags=lags, rolls=rolls, freq=freq)
    feats = feature_columns(matrix, target=target)
    matrix = matrix.sort_values("date").reset_index(drop=True)
    if matrix.empty or not feats:
        # last-ditch fallback: lag_1 only
        matrix = build_feature_matrix(canon, target=target, lags=(1,), rolls=(2,), freq=freq)
        feats = feature_columns(matrix, target=target)
        matrix = matrix.sort_values("date").reset_index(drop=True)
    if matrix.empty or not feats:
        raise ValueError(
            f"Not enough data to fit driver model: {n_rows} rows / {n_ents} entities. "
            f"Try coarser aggregation."
        )

    # Auto-enable log-transform for skewed targets (most retail data)
    if log_target is None:
        y = matrix[target].dropna()
        # Apply log1p transform only when data is mostly positive (≥ 85 % non-zero).
        # Sparse zero-inflated series (e.g. per-country weekly online retail with
        # 50 %+ zeros) are NOT log-normal; log1p hurts rather than helps there.
        # log1p handles the remaining ≤ 15 % structural zeros cleanly (log1p(0)=0).
        log_target = (y >= 0).all() and float((y > 0).mean()) >= 0.85 and float(y.skew()) > 1.5

    # Walk-forward CV: split by DATE (not row index) with non-overlapping test windows.
    # Row-index splits on a multi-entity matrix create overlapping test sets —
    # fold 1 tests on 40 % of rows, fold 2 on 25 %, fold 2 is a strict subset of fold 1.
    # Date-based non-overlapping windows give comparable, unbiased fold estimates.
    dates_sorted = np.sort(matrix["date"].unique())
    fold_reports: list[dict] = []
    all_train_for_importance = matrix  # we refit on full data at the end for importance

    if cv_folds < 2 or len(dates_sorted) < 10:
        split_date = pd.Timestamp(dates_sorted[int(len(dates_sorted) * 0.8)])
        fold_edges = [(split_date, None)]
        cv_folds = 1
    else:
        # Each test window covers an equal slice of the date range so windows
        # do not overlap.  linspace(0.5, 0.9, cv_folds) gives the split points;
        # n_test_dates is the fixed window length for every fold.
        n_test_dates = max(1, int(len(dates_sorted) * 0.20 / cv_folds))
        fold_edges = []
        for q in np.linspace(0.5, 0.9, cv_folds):
            start_idx = int(len(dates_sorted) * q)
            end_idx = min(start_idx + n_test_dates, len(dates_sorted))
            fold_edges.append(
                (pd.Timestamp(dates_sorted[start_idx]),
                 pd.Timestamp(dates_sorted[end_idx - 1]))
            )

    for split_date, end_date in fold_edges:
        train = matrix[matrix["date"] < split_date]
        test = (matrix[matrix["date"] >= split_date]
                if end_date is None
                else matrix[(matrix["date"] >= split_date) & (matrix["date"] <= end_date)])
        if train.empty or test.empty:
            continue

        y_train = np.log1p(train[target]) if log_target else train[target]
        y_test = test[target].values

        if HAVE_LGB:
            model = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=63,
                random_state=seed, verbose=-1,
            )
            mname = "lightgbm"
        else:
            model = Ridge(alpha=1.0, random_state=seed)
            mname = "ridge"
        X_train = train[feats].fillna(0)
        X_test = test[feats].fillna(0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if log_target:
            preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)

        # Baseline: per-entity seasonal-naïve on the same holdout
        bl_preds = _seasonal_naive_pred(train, test, target, season=7)

        fold_reports.append({
            "split_date": str(split_date.date()),
            "n_train": int(len(train)), "n_test": int(len(test)),
            "r2": _r2_safe(y_test, preds),
            "rmse": float(np.sqrt(np.mean((y_test - preds) ** 2))),
            "baseline_r2": _r2_safe(y_test, bl_preds),
            "baseline_rmse": float(np.sqrt(np.mean((y_test - bl_preds) ** 2))),
        })

    if not fold_reports:
        raise ValueError("Not enough data for any CV fold.")

    # Aggregate across folds
    mean_r2 = float(np.mean([f["r2"] for f in fold_reports]))
    mean_rmse = float(np.mean([f["rmse"] for f in fold_reports]))
    mean_baseline_r2 = float(np.mean([f["baseline_r2"] for f in fold_reports]))
    total_holdout = int(sum(f["n_test"] for f in fold_reports))

    # Safety net: if mean R² < 0 the model is worse than predicting the grand
    # mean on every fold — a clear sign of noise or frequency-mismatched features.
    use_baseline = False
    why_baseline = ""
    if mean_r2 < 0:
        import logging as _logging
        use_baseline = True
        why_baseline = (
            f"Mean CV R²={mean_r2:.3f} < 0 across {len(fold_reports)} fold(s). "
            f"The driver model performs worse than predicting the overall mean — "
            f"likely caused by too few observations, noisy data, or calendar "
            f"features that are meaningless at the '{freq}' aggregation level. "
            f"Baseline R²={mean_baseline_r2:.3f}. Inspect the per-fold results "
            f"and consider using a coarser or finer aggregation frequency."
        )
        _logging.warning("[RetailMind] driver model: %s", why_baseline)

    # Final fit on full data for stable feature importance
    y_full = np.log1p(matrix[target]) if log_target else matrix[target]
    if HAVE_LGB:
        final = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            random_state=seed, verbose=-1,
        )
        final.fit(matrix[feats].fillna(0), y_full)
        importance = pd.DataFrame({
            "feature": feats,
            "importance": final.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        model_name = "lightgbm"
    else:
        final = Ridge(alpha=1.0, random_state=seed)
        final.fit(matrix[feats].fillna(0), y_full)
        importance = pd.DataFrame({
            "feature": feats, "importance": np.abs(final.coef_),
        }).sort_values("importance", ascending=False)
        model_name = "ridge"

    # Direction: correlation sign between feature and target (on full data)
    directions = []
    for f in importance["feature"]:
        col = matrix[f]
        if col.std() == 0:
            directions.append(0.0)
        else:
            directions.append(float(np.corrcoef(col, matrix[target])[0, 1]))
    importance["direction"] = directions
    importance["direction_label"] = np.where(
        importance["direction"] > 0.05, "↑ increases sales",
        np.where(importance["direction"] < -0.05, "↓ decreases sales", "≈ neutral"))

    return RegressionReport(
        model_name=model_name,
        r2=mean_r2, rmse=mean_rmse,
        importance=importance.head(25).reset_index(drop=True),
        holdout_size=total_holdout,
        log_target_used=bool(log_target),
        cv_folds=len(fold_reports),
        baseline_r2=mean_baseline_r2,
        r2_lift_vs_baseline=mean_r2 - mean_baseline_r2,
        per_fold=fold_reports,
        use_baseline=use_baseline,
        why_baseline=why_baseline,
    )
