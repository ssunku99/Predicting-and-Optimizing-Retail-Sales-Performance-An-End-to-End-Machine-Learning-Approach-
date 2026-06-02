"""Forecasting with proper time-series cross-validation.

Two model families are supported out of the box:
  - LightGBM on engineered features (fast, accurate global model)
  - Seasonal-naïve baseline (sanity check)

Walk-forward CV: train on [t0, t1], validate on [t1, t1+h], slide h forward.
Reports RMSE, MAE, MAPE, SMAPE per fold and aggregated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False

from retailmind.features import build_feature_matrix, feature_columns


# ----- metrics -----

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Mean absolute percent error. eps avoids divide-by-zero on closed-store days."""
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def all_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


# ----- baselines -----

def seasonal_naive_forecast(
    canon: pd.DataFrame, horizon: int, season: int = 7, target: str = "sales"
) -> pd.DataFrame:
    """Predict y_hat[t] = y[t - season] per entity. Robust baseline."""
    out = []
    for ent, g in canon.sort_values("date").groupby("entity_id"):
        history = g[target].values
        if len(history) < season:
            continue
        last_date = g["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)
        # take the last `season` observations and tile to horizon
        cycle = history[-season:]
        preds = np.resize(cycle, horizon)
        out.append(pd.DataFrame({
            "entity_id": ent,
            "date": future_dates,
            "yhat": preds,
            "model": "seasonal_naive",
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----- LightGBM global model -----

@dataclass
class ForecastModel:
    """Trained LightGBM forecaster with metadata + safety net.

    `use_baseline` is set to True when LightGBM lost to seasonal-naive on
    the majority of CV folds — predict_future will then return the baseline
    instead. This guarantees the pipeline never ships a model that's worse
    than the obvious benchmark.
    """
    booster: object
    feature_cols: list[str]
    target: str = "sales"
    cv_metrics: dict = field(default_factory=dict)
    log_target: bool = False
    freq: str = "D"            # aggregation frequency; passed to predict_future for calendar features
    use_baseline: bool = False
    why_baseline: str = ""


def train_lgbm(
    canon: pd.DataFrame,
    target: str = "sales",
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_data_in_leaf: int = 20,
    seed: int = 42,
    cv_folds: int = 3,
    log_target: Optional[bool] = None,
    lags: tuple = (1, 7, 14, 28),
    rolls: tuple = (7, 14, 28),
    freq: str = "D",
) -> ForecastModel:
    """Train a global LightGBM forecaster with walk-forward CV.

    Auto-enables log-target if the sales distribution is right-skewed (skew > 1.5),
    which is typical for retail. Boosts R² and reduces RMSE on skewed targets.
    Also reports a seasonal-naïve baseline so the user sees % improvement.
    """
    if not HAVE_LGB:
        raise RuntimeError("lightgbm not installed")

    # Auto-shrink lags if not enough data per entity
    n_rows = len(canon)
    n_ents = canon["entity_id"].nunique() if "entity_id" in canon.columns else 1
    rows_per_ent = int(n_rows / max(n_ents, 1))
    if rows_per_ent < max(lags) + 5:
        lags = tuple(int(l) for l in lags if l < rows_per_ent - 5) or (1,)
        rolls = tuple(int(r) for r in rolls if r < rows_per_ent - 5) or (max(int(min(rows_per_ent // 2, 7)), 2),)

    matrix = build_feature_matrix(canon, target=target, lags=lags, rolls=rolls, freq=freq)
    if matrix.empty:
        # Fallback: use just lag_1 if possible
        matrix = build_feature_matrix(canon, target=target, lags=(1,), rolls=(2,), freq=freq)
    feats = feature_columns(matrix, target=target)
    matrix = matrix.sort_values("date").reset_index(drop=True)
    if matrix.empty or not feats:
        raise ValueError(
            f"Not enough data to train: only {n_rows} rows across {n_ents} entities. "
            f"Try aggregating to a coarser frequency or adding more historical data."
        )

    # Auto-decide log-transform
    if log_target is None:
        y = matrix[target].dropna()
        log_target = (y >= 0).all() and float((y > 0).mean()) >= 0.85 and float(y.skew()) > 1.5

    # Walk-forward CV by date quantiles + baseline on the same folds
    dates_sorted = np.sort(matrix["date"].unique())
    fold_metrics = []
    if cv_folds >= 2:
        edges = np.linspace(0.5, 0.95, cv_folds)
        for q in edges:
            split_date = pd.Timestamp(dates_sorted[int(len(dates_sorted) * q)])
            # Freq-appropriate validation window (~4 periods in each frequency)
            _valid_offsets = {
                "MS": pd.DateOffset(months=4), "M": pd.DateOffset(months=4),
                "W": pd.Timedelta(weeks=8), "D": pd.Timedelta(days=28),
            }
            _valid_window = _valid_offsets.get((freq or "D").upper(), pd.Timedelta(days=28))
            train = matrix[matrix["date"] < split_date]
            valid = matrix[(matrix["date"] >= split_date)
                           & (matrix["date"] < split_date + _valid_window)]
            if train.empty or valid.empty:
                continue
            y_train = np.log1p(train[target]) if log_target else train[target]
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate,
                num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf,
                random_state=seed, verbose=-1,
            )
            model.fit(train[feats], y_train)
            preds = model.predict(valid[feats])
            if log_target:
                preds = np.expm1(preds)
            preds = np.clip(preds, 0, None)

            # Baseline: per-entity seasonal-naïve over the same valid window.
            # A global shift(7) is wrong for multi-entity matrices because rows
            # from different entities are interleaved by date.
            fallback_bl = float(train[target].mean())
            combined_t = pd.concat([train, valid], ignore_index=True)
            if "entity_id" in combined_t.columns:
                _sn = combined_t.groupby("entity_id", sort=False)[target].shift(7)
                combined_t = combined_t.copy()
                combined_t["_sn"] = _sn.values
                bl_preds = combined_t["_sn"].values[len(train):]
            else:
                bl_preds = combined_t[target].shift(7).values[len(train):]
            bl_preds = np.where(np.isnan(bl_preds), fallback_bl, bl_preds)

            m = all_metrics(valid[target].values, preds)
            bm = all_metrics(valid[target].values, bl_preds)
            m["split_date"] = str(split_date.date())
            m["n_train"] = int(len(train)); m["n_valid"] = int(len(valid))
            m["baseline_rmse"] = bm["rmse"]; m["baseline_smape"] = bm["smape"]
            m["rmse_lift_pct"] = (bm["rmse"] - m["rmse"]) / max(bm["rmse"], 1) * 100
            fold_metrics.append(m)

    # Final fit on full data
    y_full = np.log1p(matrix[target]) if log_target else matrix[target]
    final = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf,
        random_state=seed, verbose=-1,
    )
    final.fit(matrix[feats], y_full)

    cv_summary = {}
    use_baseline = False
    why_baseline = ""
    if fold_metrics:
        df = pd.DataFrame(fold_metrics)
        agg = df[["rmse", "mae", "mape", "smape",
                   "baseline_rmse", "baseline_smape", "rmse_lift_pct"]].mean().to_dict()
        cv_summary = {"folds": fold_metrics, "mean": agg, "log_target": bool(log_target)}

        # Safety net: if LightGBM loses to baseline on MOST folds, fall back
        # to baseline. A senior DS would never ship a model that's worse than
        # the obvious benchmark — the pipeline auto-detects this.
        n_folds = len(fold_metrics)
        n_lost = sum(1 for f in fold_metrics if f["rmse"] >= f["baseline_rmse"])
        if n_lost >= max(2, n_folds // 2 + 1):
            use_baseline = True
            why_baseline = (
                f"LightGBM lost to seasonal-naïve on {n_lost}/{n_folds} CV folds. "
                f"Auto-fallback to baseline so we never ship a worse-than-naive model. "
                f"This usually means the dataset is too small or too noisy for a "
                f"trained model to add value over simple repetition."
            )

    return ForecastModel(
        booster=final, feature_cols=feats, target=target, cv_metrics=cv_summary,
        log_target=bool(log_target), freq=freq,
        use_baseline=use_baseline, why_baseline=why_baseline,
    )


def predict_future(
    model: ForecastModel,
    canon: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Recursive multi-step forecast — extends each entity's series by `horizon` steps.

    If `model.use_baseline` is True (safety-net triggered), returns the
    seasonal-naïve forecast instead of the LightGBM predictions.
    """
    # Safety net: pipeline decided the trained model is worse than baseline → use baseline
    if getattr(model, "use_baseline", False):
        out = seasonal_naive_forecast(canon, horizon=horizon, target=model.target)
        out["model"] = "seasonal_naive_safety_net"
        return out

    from retailmind.features import add_calendar_features, LEAKY_CONTEMPORANEOUS
    import re

    target = model.target
    feats = model.feature_cols
    # Derive the lags/rolls the model was actually trained on
    lag_set, roll_set = set(), set()
    for f in feats:
        m = re.match(rf"{target}_lag_(\d+)$", f)
        if m: lag_set.add(int(m.group(1)))
        m = re.match(rf"{target}_rmean_(\d+)$", f)
        if m: roll_set.add(int(m.group(1)))
    LAGS_USED = tuple(sorted(lag_set)) or (1,)
    ROLLS_USED = tuple(sorted(roll_set)) or (7,)
    history = canon.copy().sort_values(["entity_id", "date"]).reset_index(drop=True)
    last_dates = history.groupby("entity_id")["date"].max()
    # Aux columns (customers, etc.) whose lags the model expects
    aux_to_lag = [c for c in LEAKY_CONTEMPORANEOUS if c in history.columns and c != target]

    # Build future scaffolding: one row per entity per future date with NaN sales.
    # Track _is_future so we can return ONLY those rows later (entities can have
    # different last_dates, so a global date-mask would over-count).
    future_rows = []
    for ent, last in last_dates.items():
        future = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
        block = pd.DataFrame({"entity_id": ent, "date": future, target: np.nan,
                               "_is_future": True})
        last_row = history[history["entity_id"] == ent].iloc[-1]
        for c in history.columns:
            if c in ("entity_id", "date", target):
                continue
            block[c] = last_row[c]
        future_rows.append(block)
    future_df = pd.concat(future_rows, ignore_index=True)

    history = history.copy()
    history["_is_future"] = False
    combined = pd.concat([history, future_df], ignore_index=True)
    combined = combined.sort_values(["entity_id", "date"]).reset_index(drop=True)
    combined = add_calendar_features(combined, freq=getattr(model, "freq", "D"))

    # Ordinal-encode any object columns (entity_id, categorical aux) using the full set
    obj_cols = [c for c in combined.columns if combined[c].dtype == object and c != "date"]
    for c in obj_cols:
        combined[c] = combined[c].astype("category").cat.codes.astype(int)

    # Pre-compute group structure
    grp_key = combined["entity_id"].values
    n = len(combined)

    # Iterate future dates sorted ascending, fill predictions step by step
    future_dates_sorted = np.sort(future_df["date"].unique())

    # Aux lag features are static across the recursive loop (they don't depend on predicted sales),
    # so compute them once up-front. Derive the required lags PER aux column from the model's
    # actual feature_cols — training uses freq-aware lags (e.g. (1, 2) at daily), but earlier
    # versions of this loop hardcoded (1, 7) which caused KeyError on datasets with multiple
    # numeric aux columns (unit_price, quantity, discount, profit).
    aux_lags_needed: dict[str, set[int]] = {}
    for f in feats:
        for aux in aux_to_lag:
            m = re.match(rf"{re.escape(aux)}_lag_(\d+)$", f)
            if m:
                aux_lags_needed.setdefault(aux, set()).add(int(m.group(1)))
    for c, lags_for_c in aux_lags_needed.items():
        g = combined.groupby("entity_id", sort=False)[c]
        for lag in sorted(lags_for_c):
            combined[f"{c}_lag_{lag}"] = g.shift(lag)

    for d in future_dates_sorted:
        idx_mask = (combined["date"] == d).values
        if not idx_mask.any():
            continue
        # Recompute sales lag/rolling each step — they depend on the running predictions
        for lag in LAGS_USED:
            combined[f"{target}_lag_{lag}"] = (
                combined.groupby("entity_id", sort=False)[target].shift(lag)
            )
        shifted = combined.groupby("entity_id", sort=False)[target].shift(1)
        for w in ROLLS_USED:
            combined[f"{target}_rmean_{w}"] = (
                shifted.groupby(combined["entity_id"], sort=False)
                       .rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            combined[f"{target}_rstd_{w}"] = (
                shifted.groupby(combined["entity_id"], sort=False)
                       .rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            )
        X = combined.loc[idx_mask, feats].fillna(0)
        preds = model.booster.predict(X)
        if model.log_target:
            preds = np.expm1(preds)
        combined.loc[idx_mask, target] = np.clip(preds, 0, None)

    # Return ONLY the rows we explicitly added as future (avoids over-counting when
    # entities have different last_dates and future windows overlap).
    fut = combined.loc[combined["_is_future"] == True,
                        ["entity_id", "date", target]].copy()
    fut = fut.rename(columns={target: "yhat"})
    fut["model"] = "lightgbm"
    return fut.reset_index(drop=True)
