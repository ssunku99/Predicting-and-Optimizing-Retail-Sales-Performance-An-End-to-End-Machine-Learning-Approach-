# Predicting and Optimizing Retail Sales Performance

My Data Science Practicum 2 project at Regis University.
**Sai Teja Sunku** · MS Data Science, 2026.

**Live demo:** https://ssunku99-predicting-and-optimizing-retail-sales-perf-app-cpybcd.streamlit.app/

## What it does

You drop a retail sales file (CSV or Excel) into the web app and the
pipeline gives you back:

1. an EDA summary (rows, entities, seasonality, etc.)
2. a 28-day forecast with walk-forward cross-validation
3. anomalies — spikes and drops worth a closer look
4. a driver model that says what's actually moving sales
5. reorder recommendations (how much to order, when, and why)
6. a chat tab where you can ask things in plain English

The whole point was to get this working on *any* retail dataset
without rewriting code for each one. I tested it on two pretty
different datasets — Rossmann (a million rows of daily store sales)
and Walmart (8k rows of transactional order lines) — and the trick
is a schema mapper in the middle that figures out which raw column
means what.

## Running it locally

```bash
git clone https://github.com/ssunku99/Predicting-and-Optimizing-Retail-Sales-Performance-An-End-to-End-Machine-Learning-Approach-.git
cd Predicting-and-Optimizing-Retail-Sales-Performance-An-End-to-End-Machine-Learning-Approach-
pip install -r requirements.txt
streamlit run app.py
```

It opens in your browser. There's a "Use bundled samples" option in
the sidebar that loads the Rossmann data shipped with the repo, so
you can try the whole thing without uploading anything.

## What's in the folder

```
retailmind/           # the actual Python package, one file per stage
  schema.py             # canonical column names every module agrees on
  ingest.py             # CSV / XLSX / Parquet reader
  mapper.py             # auto-detects which raw column is what
  profiler.py           # data-quality score + suggests aggregation freq
  smart_inference.py    # fills schema gaps (e.g. revenue = qty × price)
  canonical.py          # converts mapped raw into a uniform long table
  eda.py                # overview, seasonality, promo lift, etc.
  features.py           # calendar / lag / rolling features
  forecast.py           # LightGBM + walk-forward CV + naive baseline
  anomaly.py            # IsolationForest + STL residual + IQR ensemble
  regression.py         # driver model with feature importance
  recommend.py          # reorder recommendations (R, S inventory model)
  assistant.py          # chat — works offline, richer with a Groq key
  pipeline.py           # ties all the stages together

notebooks/              # one notebook per stage, plus end-to-end demos
app.py                  # the Streamlit web app
tests/                  # pytest — runs the same checks on both datasets
health_check.py         # quick CLI script to validate a dataset
```

## Datasets I tested on

- **Rossmann store sales** (Kaggle): 1.02M rows, 1,115 German drugstore
  stores, 2013-2015
- **Walmart retail data** (Kaggle): 8,399 rows across 3 product
  categories, 2012-2015

These are old datasets but the pipeline isn't tied to them. Anything
retail with a date + a sales/quantity column should work in the app.

## Tests

`tests/test_universal.py` runs 7 assertions against both datasets
(14 tests total). If they all pass, the "works on any retail data"
claim is holding up. To add a third dataset, append it to the
`DATASETS` list in that file and the same 7 checks run automatically.

## Notes on a few design choices

These are the things I had to actually think about, not just plug
together.

### Why the schema mapper exists

My first attempt hardcoded column names like `Store`, `Date`, `Sales`.
When I tried to add Walmart, half the pipeline broke because the
columns were called `Order Date`, `Region`, and so on. So I made an
enum of "canonical roles" and every downstream module reads only
those names. The mapper guesses which raw column plays which role
from the header text + the column's dtype.

### Walk-forward CV, not k-fold

Random k-fold trains on the future and tests on the past, which is
useless for forecasting. Walk-forward picks several split dates and
trains on `[start, t]`, tests on `[t, t+28]`, for each split. That's
closer to how you'd actually use the model.

### The leakage guard

First time I ran the driver model on Rossmann I got R² = 0.98, which
looked too good. Turned out `customers` (foot traffic) was being used
as a same-day predictor of `sales` — but you don't know today's
customer count at the start of the day. After excluding that whole
set of "same-day-leaky" features (I named them `LEAKY_CONTEMPORANEOUS`
in features.py) R² dropped to a more honest 0.95.

### Baseline lift over absolute R²

A senior DS at a meetup told me "R² on its own is meaningless — show
me how much better you are than the obvious baseline". So every
metric in the pipeline is paired with a seasonal-naive baseline
(yhat[t] = sales[t-7]). Rossmann gets 18.8% SMAPE vs 44.1% baseline,
which is roughly a +82% RMSE lift. Walmart gets 87% vs 106% = +35%.
The Walmart absolute numbers look bad on their own, but the lift
over naive is real.

### Safety net

If the model loses to the baseline on most CV folds (which happens
on small or very noisy data), the pipeline falls back to baseline
automatically and tells you why. That way it never ships a forecast
that's worse than "just repeat last week".

## Limitations

- Very small datasets (under ~200 rows, single entity) — the safety
  net trips and no real forecast is produced. By design.
- Pure-noise data — same thing. No model can find signal that isn't
  there. The lift metric correctly reports near zero in those cases.
- Long-horizon recursive forecasts compound error. The horizon is
  capped at 90 days in the app for that reason.
- The schema mapper relies on English column-name patterns. For
  non-English columns you'd use the manual editor in the Streamlit
  app's first tab.

