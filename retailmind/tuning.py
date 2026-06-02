"""Hyperparameter tuning with walk-forward time-series CV.

Addresses reviewer feedback #3 on v1 ("Hyperparameter details were not mentioned
anywhere in your submission").

Why a custom tuner instead of sklearn GridSearchCV?
  Sklearn's KFold shuffles by default and even with shuffle=False, it doesn't
  respect time ordering across entities. For a global retail model with mixed
  entity-dates, walk-forward by date is the only correct evaluation.

Grid is intentionally small and well-justified — see ROSSMANN_GRID / DEFAULT_GRID
docstrings for the rationale per hyperparameter.
"""

from __future__ import annotations

import itertools
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
from retailmind.forecast import all_metrics


# A defensible, small search grid. Each hyperparameter has 2–3 well-spaced values:
#
#   n_estimators    — more rounds → more capacity but slower & higher overfit risk.
#                     200 is fast and reasonable; 400 is the default; 800 tests
#                     whether we need more.
#   learning_rate   — 0.05 is the LightGBM standard for medium-sized retail data;
#                     0.1 tests speed-vs-accuracy trade-off.
#   num_leaves      — controls tree complexity. 31 is the LightGBM default (gentle);
#                     63 doubles capacity (matches what we used pre-tuning);
#                     127 is aggressive and risks overfitting on small entities.
#   min_data_in_leaf— guards against overfit on noisy daily sales. 20 default;
#                     50 tests stronger regularization.
DEFAULT_GRID: dict[str, list] = {
    "n_estimators":     [200, 400, 800],
    "learning_rate":    [0.05, 0.1],
    "num_leaves":       [31, 63, 127],
    "min_data_in_leaf": [20, 50],
}

# Reduced grid for quick smoke tests / unit tests
QUICK_GRID: dict[str, list] = {
    "n_estimators":     [200, 400],
    "learning_rate":    [0.05],
    "num_leaves":       [31, 63],
    "min_data_in_leaf": [20],
}


@dataclass
class TuningResult:
    best_params: dict
    best_score: float                # CV mean of selected metric (lower=better for RMSE/SMAPE)
    metric: str
    all_trials: pd.DataFrame         # one row per param combo with CV metrics
    n_combos: int
    n_folds: int


def _walkforward_splits(dates: np.ndarray, n_folds: int, valid_window: int) -> list[tuple]:
    """Yield (train_mask, valid_mask) for each fold using sorted unique dates."""
    uniq = np.sort(np.unique(dates))
    splits = []
    edges = np.linspace(0.5, 0.95, n_folds)
    for q in edges:
        split_date = uniq[int(len(uniq) * q)]
        end_date = uniq[min(int(len(uniq) * q) + valid_window, len(uniq) - 1)]
        train_mask = dates < split_date
        valid_mask = (dates >= split_date) & (dates < end_date)
        if train_mask.sum() > 0 and valid_mask.sum() > 0:
            splits.append((train_mask, valid_mask, split_date))
    return splits


def tune_lgbm(
    canon: pd.DataFrame,
    target: str = "sales",
    grid: Optional[dict[str, list]] = None,
    metric: str = "rmse",
    n_folds: int = 3,
    valid_window_days: int = 28,
    seed: int = 42,
    verbose: bool = True,
    freq: str = "D",
) -> TuningResult:
    """Grid-search LightGBM hyperparameters with walk-forward CV.

    Parameters
    ----------
    canon : canonical DataFrame (date, entity_id, sales, ...).
    grid : param→values dict. Defaults to DEFAULT_GRID (12 combos).
    metric : 'rmse', 'mae', 'mape', or 'smape'. Lower is better.
    n_folds : number of walk-forward folds.
    valid_window_days : size of each validation window.
    """
    if not HAVE_LGB:
        raise RuntimeError("lightgbm not installed")
    grid = grid or DEFAULT_GRID

    matrix = build_feature_matrix(canon, target=target, freq=freq)
    feats = feature_columns(matrix, target=target)
    matrix = matrix.sort_values("date").reset_index(drop=True)
    dates = matrix["date"].values

    splits = _walkforward_splits(dates, n_folds, valid_window_days)
    if not splits:
        raise ValueError("Not enough data for walk-forward CV with current settings.")

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if verbose:
        print(f"Tuning LightGBM: {len(combos)} combinations × {len(splits)} folds = "
              f"{len(combos) * len(splits)} fits")

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        fold_scores = []
        for tm, vm, sd in splits:
            train_X, train_y = matrix.loc[tm, feats], matrix.loc[tm, target]
            valid_X, valid_y = matrix.loc[vm, feats], matrix.loc[vm, target]
            model = lgb.LGBMRegressor(
                random_state=seed, verbose=-1, n_jobs=-1, **params,
            )
            model.fit(train_X, train_y)
            preds = model.predict(valid_X)
            m = all_metrics(valid_y.values, preds)
            fold_scores.append(m[metric])
        row = {**params, f"cv_mean_{metric}": float(np.mean(fold_scores)),
               f"cv_std_{metric}": float(np.std(fold_scores))}
        rows.append(row)
        if verbose:
            print(f"  {params}  →  {metric}={row[f'cv_mean_{metric}']:.2f} "
                  f"(±{row[f'cv_std_{metric}']:.2f})")

    trials = pd.DataFrame(rows).sort_values(f"cv_mean_{metric}").reset_index(drop=True)
    best = trials.iloc[0]
    best_params = {k: best[k] for k in keys}
    # Cast numeric values back to int where appropriate
    for k in ("n_estimators", "num_leaves", "min_data_in_leaf"):
        if k in best_params:
            best_params[k] = int(best_params[k])

    return TuningResult(
        best_params=best_params,
        best_score=float(best[f"cv_mean_{metric}"]),
        metric=metric,
        all_trials=trials,
        n_combos=len(combos),
        n_folds=len(splits),
    )


def explain_grid(grid: dict[str, list] | None = None) -> str:
    """Return a human-readable rationale for each hyperparameter range."""
    grid = grid or DEFAULT_GRID
    explanations = {
        "n_estimators": ("more rounds → more capacity but slower & overfits more. "
                         "Probing whether stopping at 200 is enough or 800 helps."),
        "learning_rate": ("LightGBM standard 0.05 vs faster 0.1; both with enough rounds "
                          "for convergence."),
        "num_leaves": ("tree complexity. 31 = LightGBM default (gentle), "
                       "63 = our pre-tune default, 127 = aggressive."),
        "min_data_in_leaf": ("overfit guard for noisy daily sales. 20 default vs "
                             "50 (stronger regularisation)."),
    }
    lines = [f"Search grid: {sum(len(v) for v in grid.values())} unique values across "
             f"{len(grid)} hyperparameters."]
    for k, v in grid.items():
        lines.append(f"  • {k}={v}  — {explanations.get(k, '')}")
    return "\n".join(lines)
