"""Stage-by-stage diagnostic for the universal pipeline.

Use this when a dataset "doesn't work" to find out EXACTLY where it broke.

Usage:
    python diagnose.py path/to/your_dataset.csv

It runs each pipeline stage independently, catches exceptions, and prints a
green / yellow / red status for each stage with the actual error and a
suggested fix.

Stages checked:
    1. File read
    2. Type inference
    3. Profile (quality score, density, column candidates)
    4. Schema mapping (auto-detection)
    5. Smart inference (derive revenue, promote geo, pick freq)
    6. Canonicalization (aggregate by entity-date)
    7. EDA
    8. Feature engineering
    9. Forecast training (walk-forward CV)
   10. Forecast prediction
   11. Anomaly detection
   12. Driver regression
   13. Recommendations
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))


GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class Diagnostic:
    def __init__(self, path: Path):
        self.path = path
        self.results: list[dict] = []
        self.context: dict[str, Any] = {}

    def stage(self, name: str, fn: Callable, hint_on_fail: str = "") -> bool:
        """Run a stage. Record outcome. Return True if it passed."""
        t0 = time.time()
        try:
            out = fn()
            elapsed = time.time() - t0
            self.results.append({
                "stage": name, "status": "ok",
                "elapsed": elapsed, "output": out, "error": None,
            })
            print(f"  {GREEN}✓{RESET}  {name:<32} {DIM}({elapsed:.1f}s){RESET}")
            return True
        except Exception as e:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            self.results.append({
                "stage": name, "status": "fail",
                "elapsed": elapsed, "error": str(e), "tb": tb,
                "hint": hint_on_fail,
            })
            print(f"  {RED}✗{RESET}  {name:<32} {DIM}({elapsed:.1f}s){RESET}")
            print(f"     {RED}{type(e).__name__}: {e}{RESET}")
            if hint_on_fail:
                print(f"     {YELLOW}→ {hint_on_fail}{RESET}")
            return False

    def run(self) -> int:
        print(f"\n{BOLD}Diagnosing {self.path}{RESET}\n")
        print(f"{BOLD}{'─' * 70}{RESET}")
        print(f"{BOLD}STAGE 1-3: Read + profile{RESET}")

        # 1. File read
        def _read():
            from retailmind.ingest import load_file
            df = load_file(self.path)
            self.context["raw"] = df
            return f"{df.shape[0]:,} rows × {df.shape[1]} columns"
        if not self.stage("File read", _read,
                           "Unsupported file format or corrupt file. "
                           "Supported: .csv, .xlsx, .xls, .parquet, .tsv"):
            return self._summary()

        # 2. Type inference summary
        def _types():
            df = self.context["raw"]
            n_date = sum(1 for c in df.columns
                          if hasattr(df[c].dtype, 'name') and 'date' in df[c].dtype.name)
            n_num = df.select_dtypes(include='number').shape[1]
            n_obj = df.select_dtypes(include='object').shape[1]
            return f"{n_date} datetime, {n_num} numeric, {n_obj} text columns"
        self.stage("Type inference", _types,
                    "Date / numeric autoparse couldn't categorize the columns. "
                    "Manual schema editor in the Streamlit app can fix this.")

        # 3. Profile
        def _profile():
            from retailmind.profiler import profile
            p = profile(self.context["raw"])
            self.context["profile"] = p
            return (f"quality={p.quality_score}/100  freq={p.suggested_freq}  "
                    f"sales_candidates={len(p.candidate_sales_cols)}  "
                    f"entity_candidates={len(p.candidate_entity_cols)}")
        self.stage("Data profile", _profile,
                    "Profiler couldn't compute basic stats — dataset may be empty or have no numeric columns.")

        print(f"\n{BOLD}STAGE 4-6: Schema + canonicalize{RESET}")

        # 4. Schema mapping (don't fail on missing roles yet — smart_inference may fix)
        def _schema():
            from retailmind.mapper import SchemaMapper
            r = SchemaMapper().infer(self.context["raw"])
            self.context["mapping"] = r
            mapped = {c: role.value for c, role in r.schema.mapping.items()
                      if role.value not in ("aux", "ignore")}
            return f"mapped {len(mapped)} roles: {list(mapped.values())[:6]}…"
        if not self.stage("Schema mapping", _schema,
                           "Mapper crashed (rare). Usually a non-tabular file."):
            return self._summary()

        # 5. Smart inference (this is where revenue is derived, geo promoted, etc.)
        def _smart():
            from retailmind.smart_inference import apply_smart_inference
            from retailmind.profiler import DecisionLog
            log = DecisionLog()
            df, schema, freq = apply_smart_inference(
                self.context["raw"], self.context["mapping"].schema,
                self.context["profile"], None, log,
            )
            self.context["raw"] = df
            self.context["schema"] = schema
            self.context["freq"] = freq
            self.context["decision_log"] = log
            # Validate AFTER smart_inference — only fail if it still can't make schema valid
            errors = schema.validate()
            if errors:
                raise ValueError(
                    f"Schema still invalid after smart-inference: {errors}. "
                    f"Use the Streamlit app's manual schema editor."
                )
            return f"freq={freq}, {len(log.entries)} auto-decisions made"
        if not self.stage("Smart inference + validate", _smart,
                           "Smart-inference couldn't fill the schema. The dataset is missing "
                           "a date column or a numeric column to use as sales."):
            return self._summary()

        # 6. Canonicalize
        def _canon():
            from retailmind.canonical import canonicalize
            canon = canonicalize(
                self.context["raw"], self.context["schema"],
                freq=self.context.get("freq", "D"),
            )
            self.context["canon"] = canon
            if canon.empty:
                raise ValueError("Canonical dataframe is empty after aggregation.")
            return (f"{len(canon):,} rows · {canon['entity_id'].nunique()} entities · "
                    f"{canon['date'].min().date()} → {canon['date'].max().date()}")
        if not self.stage("Canonicalize", _canon,
                           "Aggregation produced empty result. Date column may be unparseable, "
                           "or all sales values are NaN."):
            return self._summary()

        print(f"\n{BOLD}STAGE 7-9: EDA + features + forecast{RESET}")

        # 7. EDA
        def _eda():
            from retailmind import eda
            r = eda.full_report(self.context["canon"])
            self.context["eda_report"] = r
            return f"overview keys: {list(r['overview'].keys())[:4]}…"
        self.stage("EDA report", _eda,
                    "EDA helper crashed — usually means canonical dataframe has unexpected dtypes.")

        # 8. Feature engineering
        def _feats():
            from retailmind.features import build_feature_matrix, feature_columns
            mat = build_feature_matrix(self.context["canon"])
            feats = feature_columns(mat)
            self.context["feature_matrix"] = mat
            if not feats:
                raise ValueError("No usable feature columns after dropping leakage + non-numeric.")
            return f"{len(mat):,} rows × {len(feats)} features"
        self.stage("Feature engineering", _feats,
                    "Not enough rows per entity to compute lag features. Try aggregating to a coarser frequency.")

        # 9. Forecast training
        def _forecast():
            from retailmind.forecast import train_lgbm
            df = self.context["canon"]
            # Cap on top entities for speed
            top = df.groupby("entity_id")["sales"].sum().nlargest(20).index
            df_t = df[df["entity_id"].isin(top)].copy()
            model = train_lgbm(df_t, cv_folds=2)
            self.context["model"] = model
            self.context["df_fcst_src"] = df_t
            m = model.cv_metrics.get("mean", {})
            note = " [safety-net: using baseline]" if model.use_baseline else ""
            return (f"SMAPE={m.get('smape', 0):.1f}% vs baseline {m.get('baseline_smape', 0):.1f}%"
                    f" (lift {m.get('rmse_lift_pct', 0):+.1f}%){note}")
        if not self.stage("Forecast training", _forecast,
                           "LightGBM training failed. Common causes: not enough rows for any CV fold, "
                           "all-NaN feature columns, or a categorical column LightGBM can't encode."):
            return self._summary()

        print(f"\n{BOLD}STAGE 10-13: Predict + downstream{RESET}")

        # 10. Forecast prediction
        def _predict():
            from retailmind.forecast import predict_future
            fc = predict_future(self.context["model"], self.context["df_fcst_src"], horizon=14)
            self.context["forecast"] = fc
            return f"{len(fc):,} forecast rows across {fc['entity_id'].nunique()} entities"
        self.stage("Forecast prediction", _predict,
                    "Recursive forecasting failed — usually a feature alignment issue between train/predict.")

        # 11. Anomaly detection
        def _anom():
            from retailmind.anomaly import detect_all
            a = detect_all(self.context["canon"].head(10000))  # cap for speed
            return f"{len(a)} anomalies flagged"
        self.stage("Anomaly detection", _anom,
                    "Anomaly detection failed — usually too few rows per entity for STL.")

        # 12. Driver regression
        def _drv():
            from retailmind.regression import fit_driver_model
            rep = fit_driver_model(self.context["canon"].head(20000), cv_folds=2)
            self.context["driver"] = rep
            return (f"R²={rep.r2:+.3f} (baseline {rep.baseline_r2:+.3f}, "
                    f"lift {rep.r2_lift_vs_baseline:+.3f})")
        self.stage("Driver regression", _drv,
                    "Driver regression failed — not enough data for CV folds.")

        # 13. Recommendations
        def _recs():
            from retailmind.recommend import recommend_orders, RecommendationParams
            r = recommend_orders(self.context["forecast"], params=RecommendationParams())
            return f"{len(r)} reorder recommendations generated"
        self.stage("Recommendations", _recs,
                    "Reorder model failed — forecast may be empty or non-numeric.")

        return self._summary()

    def _summary(self) -> int:
        n_ok = sum(1 for r in self.results if r["status"] == "ok")
        n_fail = sum(1 for r in self.results if r["status"] == "fail")
        total = len(self.results)
        elapsed = sum(r["elapsed"] for r in self.results)

        print(f"\n{BOLD}{'─' * 70}{RESET}")
        print(f"{BOLD}SUMMARY{RESET}")
        if n_fail == 0:
            print(f"  {GREEN}{n_ok}/{total} stages passed in {elapsed:.1f}s{RESET}")
            print(f"  {GREEN}The pipeline works end-to-end on this dataset.{RESET}")
            return 0
        else:
            print(f"  {RED}{n_fail}/{total} stages failed in {elapsed:.1f}s{RESET}")
            print(f"\n  {BOLD}Failed stages:{RESET}")
            for r in self.results:
                if r["status"] == "fail":
                    print(f"    {RED}✗ {r['stage']}{RESET}  — {r['error']}")
                    if r.get("hint"):
                        print(f"      {YELLOW}→ {r['hint']}{RESET}")
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-by-stage pipeline diagnostic")
    parser.add_argument("path", help="Dataset to diagnose")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show tracebacks for failures")
    args = parser.parse_args()

    p = Path(args.path).expanduser().resolve()
    if not p.exists():
        print(f"{RED}File not found: {p}{RESET}")
        return 2

    d = Diagnostic(p)
    code = d.run()

    if args.verbose:
        for r in d.results:
            if r["status"] == "fail" and r.get("tb"):
                print(f"\n{DIM}Traceback for {r['stage']}:{RESET}")
                print(r["tb"])
    return code


if __name__ == "__main__":
    sys.exit(main())
