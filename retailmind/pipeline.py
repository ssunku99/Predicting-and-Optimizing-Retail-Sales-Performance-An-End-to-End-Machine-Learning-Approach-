"""Orchestrator that ties the modules together.

Usage:
    from retailmind import RetailPipeline
    p = RetailPipeline.from_files('train.csv', auxiliary_paths=['store.csv'])
    p.run()           # canonicalize + EDA + forecast + anomaly + drivers + recs
    p.forecast.head()
    p.recommendations.head()

You can also call the stages individually if you want to inspect intermediate
results:
    p.profile_and_infer()
    p.canonicalize_()
    p.eda_()
    p.forecast_(sample_entities=50, cv_folds=3)
    p.anomalies_()
    p.drivers_()
    p.recommendations_()

The pipeline does a few things automatically before training:
  - profiles the raw data and picks an aggregation frequency
  - derives a sales column from quantity * unit_price if needed
  - promotes Region/Country to entity_id if there's no store column
  - shrinks model complexity if the dataset is small
  - falls back to seasonal-naive if LightGBM can't beat it on CV folds
These are all logged in `p.decision_log` so you can see what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

import re

import numpy as np
import pandas as pd

from retailmind.ingest import load_dataset, load_file
from retailmind.mapper import SchemaMapper, MappingResult
from retailmind.schema import CanonicalSchema, ColumnRole
from retailmind.canonical import canonicalize
from retailmind.cleaning import clean_dataframe, CleaningReport
from retailmind import eda
from retailmind.forecast import (train_lgbm, predict_future, ForecastModel,
                                  seasonal_naive_forecast)
from retailmind.anomaly import detect_all
from retailmind.regression import fit_driver_model, RegressionReport
from retailmind.recommend import recommend_orders, RecommendationParams
from retailmind.profiler import profile, DataProfile, DecisionLog
from retailmind.smart_inference import (apply_smart_inference,
                                          adaptive_forecast_params)


PathLike = Union[str, Path]


@dataclass
class RetailPipeline:
    """Smart, adaptive end-to-end pipeline."""

    raw: pd.DataFrame
    mapping: MappingResult
    profile_: Optional[DataProfile] = None
    decision_log: DecisionLog = field(default_factory=DecisionLog)

    cleaned: Optional[pd.DataFrame] = None
    cleaning_report: Optional[CleaningReport] = None
    canonical: Optional[pd.DataFrame] = None
    eda_report: Optional[dict] = None
    forecast_model: Optional[ForecastModel] = None
    forecast: Optional[pd.DataFrame] = None
    baseline_forecast: Optional[pd.DataFrame] = None
    anomalies: Optional[pd.DataFrame] = None
    driver_report: Optional[RegressionReport] = None
    recommendations: Optional[pd.DataFrame] = None
    per_entity_metrics: Optional[pd.DataFrame] = None

    freq: str = "auto"          # 'auto' lets profiler pick
    horizon: int = 28
    max_entities_for_full_forecast: int = 25
    chosen_freq: Optional[str] = None

    # --------- factory ----------
    @classmethod
    def from_files(cls, main_path, auxiliary_paths=None, join_on=None, overrides=None):
        if auxiliary_paths:
            raw = load_dataset(main_path, auxiliary_paths=auxiliary_paths, join_on=join_on)
        else:
            raw = load_file(main_path)
        mapping = SchemaMapper().infer(raw, overrides=overrides)
        return cls(raw=raw, mapping=mapping)

    # --------- stages ----------
    def profile_and_infer(self) -> dict:
        """Run profiler + smart-inference. Updates self.mapping, self.freq, self.decision_log."""
        self.profile_ = profile(self.raw)

        # Apply smart-inference on top of mapper output
        df, schema, freq = apply_smart_inference(
            self.raw, self.mapping.schema, self.profile_, self.freq, self.decision_log
        )
        self.raw = df
        # Wrap updated schema in a new MappingResult preserving warnings/guesses
        self.mapping = MappingResult(schema=schema, guesses=self.mapping.guesses,
                                      warnings=self.mapping.warnings)
        self.chosen_freq = freq
        return {"profile": self.profile_.to_dict(),
                "decisions": self.decision_log.to_list(),
                "freq": freq}

    def clean_(self) -> pd.DataFrame:
        """Explicit data-cleaning stage. Coerces dates, drops rows with no
        date or sales, and handles returns/refunds. Logs what it removed."""
        if self.chosen_freq is None:
            self.profile_and_infer()
        self.cleaned, self.cleaning_report = clean_dataframe(
            self.raw, self.mapping.schema, log=self.decision_log
        )
        return self.cleaned

    def canonicalize_(self) -> pd.DataFrame:
        if self.chosen_freq is None:
            self.profile_and_infer()
        if self.cleaned is None:
            self.clean_()
        self.canonical = canonicalize(self.cleaned, self.mapping.schema, freq=self.chosen_freq)
        # Auto-filter tiny entities: drop those with < 10% of the median row count.
        # Small entities ruin global model training and per-entity reporting.
        if self.canonical["entity_id"].nunique() > 5:
            counts = self.canonical.groupby("entity_id").size()
            threshold = max(int(counts.median() * 0.1), 5)
            tiny = counts[counts < threshold].index.tolist()
            if tiny:
                self.canonical = self.canonical[~self.canonical["entity_id"].isin(tiny)].copy()
                self.decision_log.log(
                    f"Filtered {len(tiny)} tiny entities",
                    f"Entities with <{threshold} observations were dropped to avoid "
                    f"polluting global training (kept {self.canonical['entity_id'].nunique()} entities).",
                    severity="skip",
                )
        return self.canonical

    def eda_(self) -> dict:
        assert self.canonical is not None
        self.eda_report = eda.full_report(self.canonical)
        return self.eda_report

    def forecast_(self, sample_entities=None, cv_folds=None):
        """Adaptive forecast: picks model complexity based on data size."""
        assert self.canonical is not None
        df = self.canonical
        if sample_entities and df["entity_id"].nunique() > sample_entities:
            top = df.groupby("entity_id")["sales"].sum().nlargest(sample_entities).index
            df_train = df[df["entity_id"].isin(top)].copy()
        else:
            df_train = df

        # Adaptive params (freq-aware lags)
        _freq = self.chosen_freq or "D"
        params = adaptive_forecast_params(df_train, freq=_freq)
        for d, r in params.pop("_decisions", []):
            self.decision_log.log(d, r, severity="fix")
        if cv_folds is None:
            cv_folds = params.pop("cv_folds", 3)
        else:
            params.pop("cv_folds", None)

        self.forecast_model = train_lgbm(
            df_train, cv_folds=cv_folds,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_data_in_leaf=params["min_data_in_leaf"],
            lags=tuple(params["lags"]),
            rolls=tuple(params["rolls"]),
            freq=_freq,
        )

        # Limit recursive-forecast scope
        if df["entity_id"].nunique() > self.max_entities_for_full_forecast:
            top = df.groupby("entity_id")["sales"].sum().nlargest(self.max_entities_for_full_forecast).index
            df_fcst = df[df["entity_id"].isin(top)]
        else:
            df_fcst = df
        self.forecast = predict_future(self.forecast_model, df_fcst, horizon=self.horizon)
        self.baseline_forecast = seasonal_naive_forecast(df_fcst, horizon=self.horizon)

        # Per-entity metrics: compute SMAPE per entity on a holdout
        self.per_entity_metrics = self._compute_per_entity_metrics(df_train)
        return self.forecast

    def _compute_per_entity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each entity, compute LightGBM vs baseline SMAPE on the last 14 days.
        Surfaces best/worst entities for stratified reporting.
        """
        from retailmind.forecast import smape
        rows = []
        # Scale the min-rows threshold with data size: at least 14 + minimum to fit lags
        min_rows = max(14, max(re.search(r"_lag_(\d+)$", c).group(1) and int(re.search(r"_lag_(\d+)$", c).group(1))
                                  for c in self.forecast_model.feature_cols
                                  if re.search(r"_lag_(\d+)$", c)) + 14)
        for ent, g in df.groupby("entity_id"):
            g = g.sort_values("date")
            if len(g) < min_rows:
                continue
            train, test = g.iloc[:-14], g.iloc[-14:]
            if train["sales"].sum() == 0:
                continue
            # Seasonal-naïve baseline
            bl = pd.concat([train, test])["sales"].shift(7).values[-14:]
            bl = np.where(np.isnan(bl), train["sales"].mean(), bl)
            # Use the trained global model for predictions (cheap — already fit)
            from retailmind.features import build_feature_matrix, feature_columns
            try:
                # build features for the whole entity timeline, then take the last 14 rows
                full_mat = build_feature_matrix(g, drop_initial_nans=False, freq=self.chosen_freq or "D")
                pred_rows = full_mat.iloc[-14:][self.forecast_model.feature_cols].fillna(0)
                preds = self.forecast_model.booster.predict(pred_rows)
                if self.forecast_model.log_target:
                    preds = np.expm1(preds)
                preds = np.clip(preds, 0, None)
            except Exception:
                continue
            rows.append({
                "entity_id": ent,
                "total_sales": float(g["sales"].sum()),
                "n_obs": len(g),
                "lgbm_smape": smape(test["sales"].values, preds),
                "baseline_smape": smape(test["sales"].values, bl),
            })
        if not rows:
            return pd.DataFrame()
        df_pe = pd.DataFrame(rows)
        df_pe["beats_baseline"] = df_pe["lgbm_smape"] < df_pe["baseline_smape"]
        df_pe["smape_lift"] = df_pe["baseline_smape"] - df_pe["lgbm_smape"]
        return df_pe.sort_values("total_sales", ascending=False).reset_index(drop=True)

    def anomalies_(self, sample_entities=50) -> pd.DataFrame:
        assert self.canonical is not None
        df = self.canonical
        if sample_entities and df["entity_id"].nunique() > sample_entities:
            top = df.groupby("entity_id")["sales"].sum().nlargest(sample_entities).index
            df = df[df["entity_id"].isin(top)].copy()
        self.anomalies = detect_all(df, freq=self.chosen_freq or "D")
        return self.anomalies

    def drivers_(self, sample_entities=50, automl: bool = False,
                  automl_time_budget: int = 30) -> RegressionReport:
        """Fit driver model. If automl=True, FLAML searches across LightGBM,
        XGBoost, RandomForest and ExtraTrees and reports the winner."""
        assert self.canonical is not None
        df = self.canonical
        if sample_entities and df["entity_id"].nunique() > sample_entities:
            top = df.groupby("entity_id")["sales"].sum().nlargest(sample_entities).index
            df = df[df["entity_id"].isin(top)].copy()
        _freq = self.chosen_freq or "D"
        params = adaptive_forecast_params(df, freq=_freq)
        cv_folds = params.get("cv_folds", 3)
        if automl:
            from retailmind.automl import fit_flaml_driver_model
            self.driver_report = fit_flaml_driver_model(
                df, cv_folds=cv_folds, time_budget=automl_time_budget, freq=_freq,
            )
            self.decision_log.log(
                "FLAML AutoML for driver model",
                f"Searched LGBM, XGBoost, RandomForest, ExtraTrees within "
                f"{automl_time_budget}s. Winner: {self.driver_report.model_name}.",
                severity="info",
            )
        else:
            self.driver_report = fit_driver_model(
                df, cv_folds=cv_folds,
                lags=tuple(params["lags"]), rolls=tuple(params["rolls"]),
                freq=_freq,
            )
        return self.driver_report

    def recommendations_(self, params=None) -> pd.DataFrame:
        assert self.forecast is not None
        self.recommendations = recommend_orders(
            self.forecast, params=params or RecommendationParams()
        )
        return self.recommendations

    def run(self, **forecast_kwargs):
        """Smart, full end-to-end run."""
        self.profile_and_infer()
        self.clean_()
        self.canonicalize_()
        self.eda_()
        self.forecast_(**forecast_kwargs)
        self.anomalies_()
        self.drivers_()
        self.recommendations_()
        return self

    # --------- reporting helpers ----------
    def stratified_summary(self) -> dict:
        """Volume-weighted, success-rate breakdown of per-entity metrics."""
        if self.per_entity_metrics is None or self.per_entity_metrics.empty:
            return {}
        pe = self.per_entity_metrics
        weights = pe["total_sales"] / pe["total_sales"].sum()
        return {
            "n_entities_evaluated": int(len(pe)),
            "pct_beating_baseline": round(float(pe["beats_baseline"].mean() * 100), 1),
            "volume_weighted_smape_lgbm": round(float((pe["lgbm_smape"] * weights).sum()), 1),
            "volume_weighted_smape_baseline": round(float((pe["baseline_smape"] * weights).sum()), 1),
            "best_entities": pe.nsmallest(5, "lgbm_smape")[
                ["entity_id", "lgbm_smape", "baseline_smape", "smape_lift"]
            ].to_dict(orient="records"),
            "worst_entities": pe.nlargest(5, "lgbm_smape")[
                ["entity_id", "lgbm_smape", "baseline_smape", "smape_lift"]
            ].to_dict(orient="records"),
        }
