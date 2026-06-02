"""AutoML wrappers for the driver regression and forecasting stages.

These are opt in. The default pipeline still uses the hand tuned LightGBM
plus seasonal naive baseline. Turning AutoML on swaps those for a search
across multiple model families.

Why this exists:
    Reviewer feedback on v1 asked "did you try multiple models". This
    module answers that. It picks among LightGBM, XGBoost, RandomForest,
    ExtraTrees and Linear via FLAML for the regression side, and among
    ARIMA, Prophet, ETS and a dozen others via AutoTS for the forecast
    side. Same evaluation discipline as the hand tuned versions (walk
    forward CV, baseline comparison).

Usage:
    from retailmind.automl import fit_flaml_driver_model, train_autots_forecaster
    rep = fit_flaml_driver_model(canon, time_budget=30)
    fc_model = train_autots_forecaster(canon, horizon=14)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from retailmind.regression import RegressionReport, _seasonal_naive_pred, _r2_safe
from retailmind.features import build_feature_matrix, feature_columns
from retailmind.forecast import (ForecastModel, all_metrics, smape,
                                  seasonal_naive_forecast)


try:
    from flaml import AutoML
    HAVE_FLAML = True
except ImportError:
    HAVE_FLAML = False

try:
    from autots import AutoTS
    HAVE_AUTOTS = True
except ImportError:
    HAVE_AUTOTS = False


def fit_flaml_driver_model(
    canon: pd.DataFrame,
    target: str = "sales",
    time_budget: int = 30,
    cv_folds: int = 3,
    estimator_list: Optional[list[str]] = None,
    log_target: Optional[bool] = None,
    seed: int = 42,
    freq: str = "D",
) -> RegressionReport:
    """Run FLAML AutoML search and return the standard RegressionReport.

    FLAML tries multiple model families (LightGBM, XGBoost, RandomForest,
    ExtraTrees, Linear) and reports the best. The output object matches
    fit_driver_model so the UI and notebooks work without changes.

    Parameters
    ----------
    time_budget : seconds FLAML is allowed to search. 30 is enough for most
                  retail datasets. Raise to 120 for a deeper search.
    estimator_list : which estimators to consider. Default covers the five
                     most common tabular learners.
    """
    if not HAVE_FLAML:
        raise RuntimeError("flaml not installed. Run: pip install flaml")

    estimators = estimator_list or ["lgbm", "xgboost", "rf", "extra_tree"]

    from retailmind.features import freq_to_lags as _freq_to_lags
    _lags, _rolls = _freq_to_lags(freq)
    matrix = build_feature_matrix(canon, target=target, freq=freq, lags=_lags, rolls=_rolls)
    feats = feature_columns(matrix, target=target)
    matrix = matrix.sort_values("date").reset_index(drop=True)
    if matrix.empty or not feats:
        raise ValueError("Not enough data or features to fit AutoML driver model.")

    if log_target is None:
        y = matrix[target].dropna()
        log_target = (y >= 0).all() and float((y > 0).mean()) >= 0.85 and float(y.skew()) > 1.5

    # Walk-forward CV with date-based non-overlapping test windows
    dates_sorted = np.sort(matrix["date"].unique())
    fold_reports: list[dict] = []
    best_estimator_per_fold: list[str] = []

    if cv_folds < 2 or len(dates_sorted) < 10:
        split_date = pd.Timestamp(dates_sorted[int(len(dates_sorted) * 0.8)])
        fold_edges = [(split_date, None)]
        cv_folds = 1
    else:
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

        am = AutoML()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am.fit(
                X_train=train[feats], y_train=y_train,
                task="regression", metric="rmse",
                time_budget=time_budget,
                estimator_list=estimators,
                eval_method="cv", n_splits=3,
                verbose=0, seed=seed,
            )

        preds = am.predict(test[feats])
        if log_target:
            preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)

        baseline_preds = _seasonal_naive_pred(train, test, target, season=7)

        fold_reports.append({
            "split_date": str(split_date.date()),
            "n_train": int(len(train)), "n_test": int(len(test)),
            "r2": _r2_safe(y_test, preds),
            "rmse": float(np.sqrt(np.mean((y_test - preds) ** 2))),
            "baseline_r2": _r2_safe(y_test, baseline_preds),
            "baseline_rmse": float(np.sqrt(np.mean((y_test - baseline_preds) ** 2))),
            "best_estimator": am.best_estimator,
            "best_config": str(am.best_config),
        })
        best_estimator_per_fold.append(am.best_estimator)

    if not fold_reports:
        raise ValueError("No CV folds were able to run.")

    mean_r2 = float(np.mean([f["r2"] for f in fold_reports]))
    mean_rmse = float(np.mean([f["rmse"] for f in fold_reports]))
    mean_baseline_r2 = float(np.mean([f["baseline_r2"] for f in fold_reports]))
    total_holdout = int(sum(f["n_test"] for f in fold_reports))

    # Refit FLAML on the full data once for the importance plot
    y_full = np.log1p(matrix[target]) if log_target else matrix[target]
    final = AutoML()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(
            X_train=matrix[feats], y_train=y_full,
            task="regression", metric="rmse",
            time_budget=time_budget,
            estimator_list=estimators,
            eval_method="cv", n_splits=3,
            verbose=0, seed=seed,
        )

    # Pull feature importance if the winning estimator supports it
    try:
        importances = final.model.estimator.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feats, "importance": importances,
        }).sort_values("importance", ascending=False)
    except Exception:
        # Fallback: linear coefs or correlation magnitude
        try:
            coefs = np.abs(final.model.estimator.coef_)
            importance_df = pd.DataFrame({
                "feature": feats, "importance": coefs,
            }).sort_values("importance", ascending=False)
        except Exception:
            corrs = [abs(float(np.corrcoef(matrix[f], matrix[target])[0, 1]))
                     if matrix[f].std() > 0 else 0.0 for f in feats]
            importance_df = pd.DataFrame({
                "feature": feats, "importance": corrs,
            }).sort_values("importance", ascending=False)

    # Add directions
    directions = []
    for f in importance_df["feature"]:
        col = matrix[f]
        if col.std() == 0:
            directions.append(0.0)
        else:
            directions.append(float(np.corrcoef(col, matrix[target])[0, 1]))
    importance_df["direction"] = directions
    importance_df["direction_label"] = np.where(
        importance_df["direction"] > 0.05, "increases sales",
        np.where(importance_df["direction"] < -0.05, "decreases sales", "neutral"))

    use_baseline = mean_r2 < 0
    why_baseline = (
        f"Mean CV R²={mean_r2:.3f} < 0 — model worse than grand-mean predictor."
        if use_baseline else ""
    )
    return RegressionReport(
        model_name=f"flaml ({final.best_estimator})",
        r2=mean_r2, rmse=mean_rmse,
        importance=importance_df.head(25).reset_index(drop=True),
        holdout_size=total_holdout,
        log_target_used=bool(log_target),
        cv_folds=len(fold_reports),
        baseline_r2=mean_baseline_r2,
        r2_lift_vs_baseline=mean_r2 - mean_baseline_r2,
        per_fold=fold_reports,
        use_baseline=use_baseline,
        why_baseline=why_baseline,
    )


# ============================================================
# AutoTS forecasting
# ============================================================

@dataclass
class AutoTSForecastResult:
    """Stripped down AutoTS wrapper that quacks like ForecastModel."""
    forecast: pd.DataFrame              # columns: entity_id, date, yhat
    best_model_name: str
    cv_metrics: dict = field(default_factory=dict)
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)


def train_autots_forecaster(
    canon: pd.DataFrame,
    horizon: int = 14,
    target: str = "sales",
    max_generations: int = 5,
    model_list: str = "fast",
    frequency: str = "infer",
    seed: int = 42,
) -> AutoTSForecastResult:
    """Run AutoTS search and produce per entity forecasts.

    AutoTS trains many time series models (ARIMA, Prophet, ETS, SeasonalNaive,
    GLM, etc.) and picks the ensemble or single model that wins on its own
    internal validation.

    Parameters
    ----------
    max_generations : how many genetic search rounds (3 is fast, 10 is thorough).
    model_list : 'fast', 'default', 'all'. 'fast' is plenty for a demo.
    frequency : 'infer' for auto, else 'D', 'W', or 'MS'.
    """
    if not HAVE_AUTOTS:
        raise RuntimeError("autots not installed. Run: pip install autots")

    long = canon.rename(columns={"date": "datetime", "sales": "value",
                                  "entity_id": "series_id"})
    long = long[["datetime", "value", "series_id"]].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = AutoTS(
            forecast_length=horizon,
            frequency=frequency,
            max_generations=max_generations,
            num_validations=2,
            model_list=model_list,
            ensemble=None,
            random_seed=seed,
            no_negatives=True,
            verbose=0,
        )
        am = am.fit(long, date_col="datetime", value_col="value", id_col="series_id")

    pred = am.predict()
    fc_df = pred.forecast.reset_index().melt(
        id_vars="datetime" if "datetime" in pred.forecast.reset_index().columns else "index",
        var_name="entity_id", value_name="yhat",
    )
    fc_df = fc_df.rename(columns={fc_df.columns[0]: "date"})
    fc_df["model"] = "autots"

    # Best model name
    try:
        best_model_name = str(am.best_model_name)
    except Exception:
        best_model_name = "autots_ensemble"

    # Leaderboard
    try:
        lb = am.initial_results.model_results
        leaderboard = lb[["Model", "smape_weighted", "rmse_weighted"]] \
            .sort_values("smape_weighted").head(15).reset_index(drop=True)
    except Exception:
        leaderboard = pd.DataFrame()

    return AutoTSForecastResult(
        forecast=fc_df[["entity_id", "date", "yhat", "model"]],
        best_model_name=best_model_name,
        cv_metrics={"best_model": best_model_name,
                     "models_tried": int(len(leaderboard)) if not leaderboard.empty else None},
        leaderboard=leaderboard,
    )
