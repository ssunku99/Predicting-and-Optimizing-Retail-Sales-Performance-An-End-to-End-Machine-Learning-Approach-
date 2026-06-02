"""RetailMind health check — verify the pipeline works on ANY dataset.

Run before submission / a demo to be sure no dataset blocks the pipeline.

Usage:
    # Check a single file
    python health_check.py path/to/sales.csv

    # Check every CSV/XLSX in a folder
    python health_check.py path/to/datasets/

    # Strict mode: also run forecast+CV (slow on big files)
    python health_check.py path/to/file.csv --full

The script prints a per-file report:
    ✅ ok       — mapper found the schema, pipeline ran end-to-end
    ⚠️  warn    — mapper succeeded with assumptions; check the warnings
    ❌ failed   — pipeline could not run; manual schema fix required
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retailmind.ingest import load_file
from retailmind.mapper import SchemaMapper
from retailmind.canonical import canonicalize
from retailmind.schema import ColumnRole


SUPPORTED_EXT = {".csv", ".xlsx", ".xls", ".parquet", ".tsv", ".txt"}


def check_one(path: Path, full: bool = False) -> dict:
    """Return a dict with status, mapping, warnings, errors, timing."""
    out = {"path": str(path), "status": "❌ failed", "errors": [],
           "warnings": [], "mapping": {}, "elapsed": 0.0}
    t0 = time.time()
    try:
        raw = load_file(path)
        out["rows"] = len(raw)
        out["columns"] = list(raw.columns)

        result = SchemaMapper().infer(raw)
        out["mapping"] = {raw_col: role.value for raw_col, role in result.schema.mapping.items()
                          if role not in (ColumnRole.AUX, ColumnRole.IGNORE)}
        out["warnings"] = result.warnings

        errors = result.schema.validate()
        if errors:
            out["errors"] = errors
            return out

        # Try canonicalize
        canon = canonicalize(raw, result.schema, freq="D")
        out["canon_rows"] = len(canon)
        out["entities"] = canon["entity_id"].nunique()
        out["status"] = "⚠️  warn" if result.warnings else "✅ ok"

        if full:
            from retailmind.forecast import train_lgbm, predict_future
            # Sample top-10 entities to keep it fast
            top = canon.groupby("entity_id")["sales"].sum().nlargest(10).index
            df_sub = canon[canon["entity_id"].isin(top)].copy()
            model = train_lgbm(df_sub, cv_folds=2)
            fc = predict_future(model, df_sub, horizon=7)
            out["cv_smape"] = model.cv_metrics.get("mean", {}).get("smape")
            out["forecast_rows"] = len(fc)

    except Exception as e:
        out["errors"] = [f"{type(e).__name__}: {e}"]
        out["status"] = "❌ failed"
        out["traceback"] = traceback.format_exc()
    finally:
        out["elapsed"] = round(time.time() - t0, 2)
    return out


def print_report(report: dict, verbose: bool = False) -> None:
    print(f"\n{report['status']}  {report['path']}  ({report['elapsed']}s)")
    print(f"   rows: {report.get('rows', '?')}   columns: {report.get('columns', [])}")
    if report.get("mapping"):
        print("   detected schema:")
        for raw, role in report["mapping"].items():
            print(f"      {raw!r:>30} → {role}")
    if report.get("warnings"):
        print("   warnings:")
        for w in report["warnings"]:
            print(f"      • {w}")
    if report.get("errors"):
        print("   ❌ errors:")
        for e in report["errors"]:
            print(f"      • {e}")
        if verbose and report.get("traceback"):
            print("\n" + report["traceback"])
    if "canon_rows" in report:
        print(f"   canonical: {report['canon_rows']:,} rows · {report['entities']} entities")
    if "cv_smape" in report:
        print(f"   forecast CV SMAPE: {report['cv_smape']:.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="RetailMind health check")
    parser.add_argument("path", help="File or folder to check")
    parser.add_argument("--full", action="store_true",
                        help="Also run forecast + CV (slow on big files)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show tracebacks for failures")
    args = parser.parse_args()

    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        print(f"❌ Path not found: {target}")
        return 2

    files = []
    if target.is_file():
        files = [target]
    else:
        for ext in SUPPORTED_EXT:
            files.extend(target.glob(f"*{ext}"))
        files = sorted(files)

    if not files:
        print(f"No supported files (.csv/.xlsx/.parquet) found in {target}")
        return 1

    print(f"Checking {len(files)} file(s)…")
    reports = []
    for f in files:
        rep = check_one(f, full=args.full)
        print_report(rep, verbose=args.verbose)
        reports.append(rep)

    # Summary
    n_ok = sum(1 for r in reports if r["status"].startswith("✅"))
    n_warn = sum(1 for r in reports if r["status"].startswith("⚠️"))
    n_fail = sum(1 for r in reports if r["status"].startswith("❌"))
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: ✅ {n_ok} ok · ⚠️ {n_warn} warn · ❌ {n_fail} failed")
    print('=' * 60)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
