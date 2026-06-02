# Predicting and optimizing retail sales performance 

Data Science Practicum 2
Sai Teja Sunku, Regis University.

A pipeline that takes a retail sales CSV/Excel and gives you EDA, forecasts,
anomalies, sales drivers, and reorder recommendations. The whole point is
that the same code is supposed to work on different datasets without me
having to rewrite anything for each new schema, so I tested it on Rossmann
(store-day data) and Walmart (transactional order-line data) and added a
schema mapper in the middle that figures out which raw column is what.

## What's in the folder

```
retailmind/         # the actual Python package: one file per stage
  schema.py         # canonical column names every module agrees on
  ingest.py         # reads CSV / XLSX / Parquet
  mapper.py         # auto-detects which raw column is sales / date / etc.
  profiler.py       # quality score + suggested aggregation frequency
  smart_inference.py# fills schema gaps (e.g. derive revenue from qty * price)
  canonical.py      # converts mapped raw → uniform long-format dataframe
  eda.py            # overview, seasonality, promo lift, etc.
  features.py       # calendar + lag + rolling features (with leakage guard)
  forecast.py       # LightGBM with walk-forward CV + seasonal-naive baseline
  anomaly.py        # IsolationForest + STL residual + IQR
  regression.py     # sales-driver model with feature importance
  recommend.py      # periodic-review (R, S) reorder model
  assistant.py      # natural-language Q&A (offline + optional Groq)
  pipeline.py       # ties everything together
  report.py         # static HTML report generator (for GitHub Pages)

notebooks/          # one notebook per stage (run in order or jump around)
  01_eda.ipynb
  02_schema_mapping.ipynb
  03_feature_engineering.ipynb
  04_forecasting.ipynb
  05_anomaly_detection.ipynb
  06_regression_drivers.ipynb
  07_order_recommendations.ipynb
  08_assistant.ipynb
  09_universal_pipeline.ipynb       # end-to-end demo on both datasets
  10_senior_ds_audit.ipynb          # checks for leakage, calibration, etc.
  11_final_dashboard.ipynb          # all metrics in one place

app.py              # Streamlit app — upload any CSV and run the pipeline
tests/              # pytest suite that runs same checks on both datasets
docs/               # generated HTML reports for GitHub Pages
health_check.py     # quick script to validate any dataset before running
requirements.txt
```

## Datasets I used

- Rossmann store sales (Kaggle): 1.02 M rows, 1,115 stores, 2013-2015
- Walmart retail transactions (Kaggle): 8,399 order lines, 2012-2015

These are old datasets but the pipeline is supposed to work on any retail
data, not just these two. The Streamlit app lets you upload your own.

## How to run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# either run the notebooks
jupyter notebook notebooks/

# or run the interactive app
streamlit run app.py

# or run the pipeline directly from Python
python3 -c "
from retailmind import RetailPipeline
p = RetailPipeline.from_files('train.csv', auxiliary_paths=['store.csv'])
p.run()
print(p.forecast.head())
"
```

## How to verify it actually works on both datasets

```bash
pytest tests/test_universal.py -v
```

This runs 7 assertions against both Rossmann and Walmart (14 tests total).
If they all pass, the universality claim holds. Drop a third dataset into
`tests/test_universal.py:DATASETS` and the same 7 assertions run against
it without me touching anything else.

## Design notes (what I learned)

### Why the schema mapper

My first attempt hard-coded column names like `Store`, `Date`, `Sales`.
When I tried to add Walmart, half the pipeline broke because the columns
were called `Order Date`, `Region`, etc. So I made an enum of "canonical
roles" and made every downstream module read only those names. The mapper
guesses which raw column plays which role using name patterns + dtype.

### Why walk-forward CV

Random k-fold trains on future data and tests on past — useless for
forecasting. Walk-forward picks N split-dates and trains on [start, t],
tests on [t, t+28], for each split. That mimics how you'd actually use
the model in production.

### The leakage guard

When I first ran the driver model on Rossmann I got R² = 0.98 which was
suspicious. Turned out `customers` (foot traffic) was being used as a
same-day predictor of `sales` — but you don't know today's customer count
until the day is over. After excluding it, R² dropped to a more honest
0.95. I called the excluded set `LEAKY_CONTEMPORANEOUS` in features.py
and the audit notebook quantifies the difference.

### Why baseline lift matters more than absolute R²

A senior DS at a meetup told me "R² alone is meaningless — show me how
much better you are than the obvious baseline". So every metric in the
pipeline is paired with seasonal-naive (yhat[t] = sales[t-7]). For
Rossmann the model gets 18.8% SMAPE vs 44.1% baseline = +82% RMSE lift.
For Walmart it's 87% vs 106% = +35% lift. Walmart's absolute numbers
look bad but the model is still adding real value over the naive guess.

### Safety net

If the model loses to baseline on most CV folds (happens on tiny / noisy
data) the pipeline auto-falls back to the baseline. That way it never
ships a model that's worse than what you'd get by repeating last week.

## Deployment

- Static HTML reports → push `docs/` to GitHub Pages → free permanent URL
- Interactive app → push repo to GitHub, deploy via share.streamlit.io

The static report can be regenerated by running `notebooks/11_final_dashboard.ipynb`
which calls `retailmind.report.save_report()` on both datasets and writes
to `docs/`.

## Limitations I know about

- Very small datasets (<200 rows total, 1 entity) — the safety net kicks
  in and falls back to baseline because LightGBM can't outperform on
  that little data.
- Pure-noise data — same problem; no model can extract signal that isn't
  there. The lift metric correctly reports ~0% in those cases.
- Long-horizon recursive forecasting (>60 days) compounds error. The
  pipeline caps at 90.
- The schema mapper uses English column-name patterns. Non-English column
  names need the manual editor in the Streamlit app.

## Things I'd add if I had more time

- Prophet alongside LightGBM as a forecast alternative (Prophet handles
  yearly seasonality well, LightGBM doesn't see year-over-year patterns
  as easily without explicit features).
- Quantile regression to get prediction intervals instead of just point
  forecasts (would feed into the reorder model's safety stock).
- A per-entity ARIMA fallback for stores that have unusual patterns the
  global model misses.
- More tests on more datasets — I want to confirm the pipeline works on
  a Shopify export, a grocery dataset, and something with non-USD currency.
